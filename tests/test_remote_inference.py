# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused protocol and queue tests for rollout remote inference."""

from __future__ import annotations

import math
import pickle  # nosec: exercises the trusted Async Inference wire protocol
from collections import deque
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch


class _TimedAction:
    def __init__(self, timestep: int, action: torch.Tensor):
        self.timestep = timestep
        self.action = action

    def get_timestep(self) -> int:
        return self.timestep

    def get_action(self) -> torch.Tensor:
        return self.action


def _bare_engine():
    from lerobot.rollout.inference.base import InferenceEngine
    from lerobot.rollout.inference.remote import RemoteInferenceEngine

    engine = RemoteInferenceEngine.__new__(RemoteInferenceEngine)
    InferenceEngine.__init__(engine)
    engine._channel = None
    engine._stub = None
    engine._worker = None
    engine._started = False
    engine._active = Event()
    engine._stop_event = Event()
    engine._request_event = Event()
    engine._action_available = Event()
    engine._failed = Event()
    engine._rpc_lock = Lock()
    engine._observation_lock = Lock()
    engine._queue_lock = Lock()
    engine._latest_observation = None
    engine._observation_timestep = 0
    engine._observation_pending = False
    engine._action_queue = deque()
    engine._latest_action_timestep = -1
    engine._has_received_action = False
    engine._worker_error = None
    engine._first_action_timeout_s = 0.01
    engine._global_shutdown_event = Event()
    return engine


def test_remote_inference_config_defaults_and_validation() -> None:
    from lerobot.rollout.inference.factory import RemoteInferenceConfig

    config = RemoteInferenceConfig(server_address="policy-host:8080")

    assert config.type == "remote"
    assert config.actions_per_chunk == 50
    assert config.chunk_size_threshold == 0.5
    assert config.aggregate_fn_name == "weighted_average"
    assert config.first_action_timeout_s == 120.0

    invalid = (
        {"server_address": ""},
        {"actions_per_chunk": 0},
        {"chunk_size_threshold": -0.01},
        {"chunk_size_threshold": 1.01},
        {"aggregate_fn_name": "unknown"},
        {"connect_timeout_s": 0.0},
        {"setup_timeout_s": 0.0},
        {"rpc_timeout_s": 0.0},
        {"first_action_timeout_s": 0.0},
    )
    for override in invalid:
        with pytest.raises(ValueError):
            RemoteInferenceConfig(**override)


@pytest.mark.parametrize(
    "field",
    [
        "actions_per_chunk",
        "chunk_size_threshold",
        "connect_timeout_s",
        "setup_timeout_s",
        "rpc_timeout_s",
        "first_action_timeout_s",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_remote_inference_config_rejects_non_finite_numbers(field: str, value: float) -> None:
    from lerobot.rollout.inference.factory import RemoteInferenceConfig

    with pytest.raises(ValueError):
        RemoteInferenceConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fps", 0),
        ("fps", False),
        ("fps", 25.0),
        ("inference_latency", math.nan),
        ("obs_queue_timeout", math.inf),
    ],
)
def test_policy_server_config_rejects_invalid_timing_values(field: str, value: object) -> None:
    from lerobot.async_inference.configs import PolicyServerConfig

    with pytest.raises(ValueError):
        PolicyServerConfig(**{field: value})


def test_remote_handshake_resets_session_before_sending_policy_spec() -> None:
    engine = _bare_engine()
    engine._stub = MagicMock()
    engine._services_pb2 = SimpleNamespace(
        Empty=lambda: "empty",
        PolicySetup=lambda *, data: SimpleNamespace(data=data),
    )
    engine._policy_specs = SimpleNamespace(policy_type="pi05", actions_per_chunk=50)
    engine._rpc_timeout_s = 3.0
    engine._setup_timeout_s = 120.0

    engine._handshake()

    assert engine._stub.method_calls == [
        call.Ready("empty", timeout=3.0),
        call.SendPolicyInstructions(engine._stub.SendPolicyInstructions.call_args.args[0], timeout=120.0),
    ]
    setup = engine._stub.SendPolicyInstructions.call_args.args[0]
    restored = pickle.loads(setup.data)  # nosec
    assert restored.policy_type == "pi05"
    assert restored.actions_per_chunk == 50


def test_remote_start_closes_channel_when_handshake_fails() -> None:
    engine = _bare_engine()
    channel = MagicMock()
    ready_future = MagicMock()
    ready_future.result.return_value = None
    engine._grpc = MagicMock()
    engine._grpc.insecure_channel.return_value = channel
    engine._grpc.channel_ready_future.return_value = ready_future
    engine._stub_class = MagicMock()
    engine._server_address = "policy-host:8080"
    engine._channel_options = ()
    engine._connect_timeout_s = 2.0
    engine._started = False
    engine._stop_event = Event()
    engine._failed = Event()
    engine._handshake = MagicMock(side_effect=RuntimeError("setup rejected"))

    with pytest.raises(ConnectionError, match="policy-host:8080"):
        engine.start()

    channel.close.assert_called_once_with()
    assert engine._channel is None
    assert engine._stub is None
    assert not engine._started


def test_remote_chunk_merge_retains_uncovered_actions_and_aggregates_overlap() -> None:
    engine = _bare_engine()
    engine._queue_lock = Lock()
    engine._latest_action_timestep = 4
    engine._aggregate_fn = lambda old, new: 0.25 * old + 0.75 * new
    engine._action_queue = deque(
        [_TimedAction(5, torch.tensor([10.0])), _TimedAction(6, torch.tensor([20.0]))]
    )

    engine._merge_action_chunk(
        [
            _TimedAction(4, torch.tensor([4.0])),
            _TimedAction(6, torch.tensor([6.0])),
            _TimedAction(7, torch.tensor([7.0])),
        ]
    )

    assert [action.get_timestep() for action in engine._action_queue] == [5, 6, 7]
    assert torch.equal(engine._action_queue[0].get_action(), torch.tensor([10.0]))
    assert torch.equal(engine._action_queue[1].get_action(), torch.tensor([9.5]))
    assert torch.equal(engine._action_queue[2].get_action(), torch.tensor([7.0]))


def test_remote_observation_schedules_once_at_or_below_refill_threshold() -> None:
    engine = _bare_engine()
    engine._queue_lock = Lock()
    engine._observation_lock = Lock()
    engine._started = True
    engine._active = Event()
    engine._active.set()
    engine._failed = Event()
    engine._request_event = Event()
    engine._observation_pending = False
    engine._task = "Stack blocks"
    engine._actions_per_chunk = 10
    engine._chunk_size_threshold = 0.5
    engine._latest_action_timestep = 3
    engine._action_queue = deque(range(6))

    first = {"left_j1.pos": 0.0}
    engine.notify_observation(first)
    assert not engine._request_event.is_set()

    engine._action_queue.pop()
    engine.notify_observation(first)

    assert engine._request_event.is_set()
    assert engine._latest_observation == {"left_j1.pos": 0.0, "task": "Stack blocks"}
    assert engine._observation_timestep == 4
    assert "task" not in first

    engine.notify_observation({"left_j1.pos": 1.0})
    assert engine._latest_observation["left_j1.pos"] == 0.0


def test_remote_worker_round_trip_enqueues_actions_and_clears_pending_flag() -> None:
    from lerobot.async_inference.helpers import TimedAction, TimedObservation

    engine = _bare_engine()
    engine._stop_event = Event()
    engine._active = Event()
    engine._active.set()
    engine._request_event = Event()
    engine._request_event.set()
    engine._failed = Event()
    engine._rpc_lock = Lock()
    engine._observation_lock = Lock()
    engine._queue_lock = Lock()
    engine._latest_observation = {"left_j1.pos": 0.25, "task": "Stack blocks"}
    engine._observation_timestep = 8
    engine._observation_pending = True
    engine._latest_action_timestep = 7
    engine._action_queue = deque()
    engine._aggregate_fn = lambda _old, new: new
    engine._global_shutdown_event = Event()
    engine._services_pb2 = SimpleNamespace(Observation=object, Empty=lambda: "empty")
    engine._send_bytes_in_chunks = lambda payload, *_args, **_kwargs: [payload]
    engine._stub = MagicMock()

    actions = [
        TimedAction(timestamp=1.0, timestep=8, action=torch.tensor([1.0, 2.0])),
        TimedAction(timestamp=1.1, timestep=9, action=torch.tensor([3.0, 4.0])),
    ]

    def get_actions(_request, *, timeout):
        assert timeout == 15.0
        engine._stop_event.set()
        return SimpleNamespace(data=pickle.dumps(actions))  # nosec

    engine._stub.GetActions.side_effect = get_actions
    engine._rpc_timeout_s = 5.0
    engine._first_action_timeout_s = 15.0

    engine._remote_loop()

    engine._stub.SendObservations.assert_called_once()
    serialized_observation = engine._stub.SendObservations.call_args.args[0][0]
    observation = pickle.loads(serialized_observation)  # nosec
    assert isinstance(observation, TimedObservation)
    assert observation.get_timestep() == 8
    assert observation.get_observation()["task"] == "Stack blocks"
    assert [item.get_timestep() for item in engine._action_queue] == [8, 9]
    telemetry = engine.get_inference_telemetry()
    assert telemetry["chunk_id"] == 0
    assert telemetry["inference_started_ns"] is not None
    assert telemetry["inference_finished_ns"] >= telemetry["inference_started_ns"]
    assert telemetry["inference_latency_ms"] >= 0
    assert telemetry["queue_size"] == 2
    assert not engine._observation_pending
    assert not engine.failed


def test_remote_worker_rpc_failure_requests_global_shutdown() -> None:
    engine = _bare_engine()
    engine._stop_event = Event()
    engine._active = Event()
    engine._active.set()
    engine._request_event = Event()
    engine._request_event.set()
    engine._failed = Event()
    engine._rpc_lock = Lock()
    engine._observation_lock = Lock()
    engine._queue_lock = Lock()
    engine._latest_observation = {"left_j1.pos": 0.25}
    engine._observation_timestep = 0
    engine._observation_pending = True
    engine._latest_action_timestep = -1
    engine._action_queue = deque()
    engine._aggregate_fn = lambda _old, new: new
    engine._global_shutdown_event = Event()
    engine._services_pb2 = SimpleNamespace(Observation=object, Empty=lambda: "empty")
    engine._send_bytes_in_chunks = lambda payload, *_args, **_kwargs: [payload]
    engine._stub = MagicMock()
    engine._stub.SendObservations.side_effect = RuntimeError("server unavailable")
    engine._rpc_timeout_s = 5.0

    engine._remote_loop()

    assert engine.failed
    assert engine._global_shutdown_event.is_set()
    assert not engine._active.is_set()
    assert not engine._observation_pending
    with pytest.raises(RuntimeError, match="server unavailable") as exc_info:
        engine.get_action(None)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_remote_first_action_timeout_fails_closed() -> None:
    engine = _bare_engine()
    engine._started = True
    engine._active.set()

    with pytest.raises(TimeoutError, match="first remote action chunk"):
        engine.get_action(None)

    assert engine.failed
    assert engine._global_shutdown_event.is_set()
    assert not engine._active.is_set()


def test_remote_first_action_wait_honors_external_shutdown() -> None:
    engine = _bare_engine()
    engine._started = True
    engine._active.set()
    engine._global_shutdown_event.set()

    assert engine.get_action(None) is None
    assert not engine.failed


def test_policy_server_reuses_matching_resident_policy(monkeypatch, tmp_path) -> None:
    import lerobot.async_inference.policy_server as policy_server_module
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.async_inference.policy_server import PolicyServer

    server = PolicyServer(PolicyServerConfig())
    policy = MagicMock()
    policy.to.return_value = policy
    preprocessor = MagicMock()
    postprocessor = MagicMock()
    policy_class = MagicMock()
    policy_class.from_pretrained.return_value = policy
    monkeypatch.setattr(policy_server_module, "get_policy_class", lambda _policy_type: policy_class)
    monkeypatch.setattr(
        policy_server_module,
        "make_pre_post_processors",
        lambda *_args, **_kwargs: (preprocessor, postprocessor),
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    specs = RemotePolicyConfig(
        policy_type="pi05",
        pretrained_name_or_path=str(checkpoint),
        lerobot_features={},
        actions_per_chunk=50,
        device="cpu",
        pretrained_revision="revision-1",
    )

    assert server._configure_policy(specs)
    assert not server._configure_policy(specs)

    policy_class.from_pretrained.assert_called_once_with(str(checkpoint), revision="revision-1")
    policy.to.assert_called_once_with("cpu")
    policy.eval.assert_called_once_with()
    policy.reset.assert_called_once_with()
    preprocessor.reset.assert_called_once_with()
    postprocessor.reset.assert_called_once_with()


def test_policy_server_inference_failure_aborts_rpc() -> None:
    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.async_inference.policy_server import PolicyServer

    server = PolicyServer(PolicyServerConfig(obs_queue_timeout=0.01, inference_latency=0.0))
    server.observation_queue.put(TimedObservation(timestamp=1.0, timestep=0, observation={}, must_go=True))
    server._predict_action_chunk = MagicMock(side_effect=RuntimeError("model crashed"))
    context = MagicMock()
    context.peer.return_value = "test-client"
    context.abort.side_effect = RuntimeError("rpc aborted")
    server._owner_peer = "test-client"

    with pytest.raises(RuntimeError, match="rpc aborted"):
        server.GetActions(None, context)

    context.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "Policy inference failed: model crashed")


def test_policy_server_rejects_second_client_across_all_rpcs() -> None:
    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    server = PolicyServer(PolicyServerConfig(obs_queue_timeout=0.0, inference_latency=0.0))
    owner_context = MagicMock()
    owner_context.peer.return_value = "ipv4:127.0.0.1:10001"
    server.Ready(None, owner_context)

    marker = object()
    server.last_processed_obs = marker
    intruder_context = MagicMock()
    intruder_context.peer.return_value = "ipv4:127.0.0.1:10002"

    server.Ready(None, intruder_context)
    server.SendPolicyInstructions(SimpleNamespace(data=b"must not be deserialized"), intruder_context)
    server.SendObservations(iter(()), intruder_context)
    response = server.GetActions(None, intruder_context)

    assert intruder_context.abort.call_count == 4
    assert all(
        call.args[0] is grpc.StatusCode.RESOURCE_EXHAUSTED for call in intruder_context.abort.call_args_list
    )
    assert all(
        "one server process per robot" in call.args[1] for call in intruder_context.abort.call_args_list
    )
    assert server._owner_peer == "ipv4:127.0.0.1:10001"
    assert server.last_processed_obs is marker
    assert response.data == b""


def test_policy_server_requires_ready_before_policy_rpcs() -> None:
    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.transport import services_pb2

    server = PolicyServer(PolicyServerConfig())
    context = MagicMock()
    context.peer.return_value = "test-client"

    response = server.SendPolicyInstructions(SimpleNamespace(data=b"must not be deserialized"), context)

    context.abort.assert_called_once_with(
        grpc.StatusCode.FAILED_PRECONDITION,
        "Call Ready before using policy RPCs.",
    )
    assert response == services_pb2.Empty()
