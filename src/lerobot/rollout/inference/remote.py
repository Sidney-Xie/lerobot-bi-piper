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

"""Remote asynchronous policy inference for the rollout engine."""

from __future__ import annotations

import logging
import pickle  # nosec: protocol compatibility; use only on a trusted network
import time
from collections import deque
from copy import copy
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import torch

from lerobot.utils.import_utils import require_package

from .base import InferenceEngine

if TYPE_CHECKING:
    from lerobot.configs import PreTrainedConfig

    from .factory import RemoteInferenceConfig

logger = logging.getLogger(__name__)

_REMOTE_JOIN_TIMEOUT_S = 3.0


class RemoteInferenceEngine(InferenceEngine):
    """Use the Async Inference gRPC server without giving it robot ownership."""

    def __init__(
        self,
        config: RemoteInferenceConfig,
        *,
        policy_config: PreTrainedConfig,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        rename_map: dict[str, str],
        shutdown_event: Event | None = None,
    ) -> None:
        super().__init__()
        require_package("grpcio", extra="async", import_name="grpc")

        # Keep optional gRPC imports out of module scope so local sync/RTC
        # rollouts do not require the async dependency group.
        import grpc

        from lerobot.async_inference.configs import get_aggregate_function
        from lerobot.async_inference.helpers import RemotePolicyConfig
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

        self._grpc = grpc
        self._services_pb2 = services_pb2
        self._stub_class = services_pb2_grpc.AsyncInferenceStub
        self._send_bytes_in_chunks = send_bytes_in_chunks
        self._channel_options = grpc_channel_options(initial_backoff=f"{1 / fps:.4f}s")

        server_policy_path = config.policy_path or policy_config.pretrained_path
        self._policy_specs = RemotePolicyConfig(
            policy_type=policy_config.type,
            pretrained_name_or_path=str(server_policy_path),
            lerobot_features=hw_features,
            actions_per_chunk=config.actions_per_chunk,
            device=device or "cpu",
            rename_map=rename_map,
            pretrained_revision=policy_config.pretrained_revision,
        )
        self._server_address = config.server_address
        self._actions_per_chunk = config.actions_per_chunk
        self._chunk_size_threshold = config.chunk_size_threshold
        self._aggregate_fn = get_aggregate_function(config.aggregate_fn_name)
        self._connect_timeout_s = config.connect_timeout_s
        self._setup_timeout_s = config.setup_timeout_s
        self._rpc_timeout_s = config.rpc_timeout_s
        self._first_action_timeout_s = config.first_action_timeout_s
        self._task = task
        self._global_shutdown_event = shutdown_event

        self._channel = None
        self._stub = None
        self._worker: Thread | None = None
        self._started = False
        self._active = Event()
        self._stop_event = Event()
        self._request_event = Event()
        self._action_available = Event()
        self._failed = Event()
        self._rpc_lock = Lock()
        self._observation_lock = Lock()
        self._queue_lock = Lock()
        self._latest_observation: dict[str, Any] | None = None
        self._observation_timestep = 0
        self._observation_pending = False
        self._action_queue: deque[Any] = deque()
        self._latest_action_timestep = -1
        self._has_received_action = False
        self._worker_error: Exception | None = None

    @property
    def ready(self) -> bool:
        return (
            self._started
            and self._worker is not None
            and self._worker.is_alive()
            and not self._failed.is_set()
        )

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def _clear_local_state(self) -> None:
        with self._queue_lock:
            self._action_queue.clear()
            self._latest_action_timestep = -1
            self._has_received_action = False
        with self._observation_lock:
            self._latest_observation = None
            self._observation_pending = False
        self._request_event.clear()
        self._action_available.clear()

    def _handshake(self) -> None:
        self._stub.Ready(self._services_pb2.Empty(), timeout=self._rpc_timeout_s)
        setup = self._services_pb2.PolicySetup(data=pickle.dumps(self._policy_specs))  # nosec
        self._stub.SendPolicyInstructions(setup, timeout=self._setup_timeout_s)

    def start(self) -> None:
        if self._started:
            return
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("Previous remote inference worker is still running")

        self._stop_event.clear()
        self._failed.clear()
        self._worker_error = None
        self._reset_inference_telemetry()
        self._clear_local_state()
        self._channel = self._grpc.insecure_channel(self._server_address, self._channel_options)
        try:
            self._grpc.channel_ready_future(self._channel).result(timeout=self._connect_timeout_s)
            self._stub = self._stub_class(self._channel)
            self._handshake()
        except Exception as exc:
            self._channel.close()
            self._channel = None
            self._stub = None
            raise ConnectionError(
                f"Could not initialize remote policy server at {self._server_address}"
            ) from exc

        self._started = True
        self._worker = Thread(target=self._remote_loop, daemon=True, name="RemotePolicyInference")
        self._worker.start()
        logger.info("Connected to remote policy server at %s", self._server_address)

    def stop(self) -> None:
        self._active.clear()
        self._stop_event.set()
        self._request_event.set()
        self._action_available.set()
        if self._channel is not None:
            # Closing the channel unblocks an in-flight GetActions call.
            self._channel.close()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=_REMOTE_JOIN_TIMEOUT_S)
            if worker.is_alive():
                logger.warning("Remote inference worker did not stop within %.1fs", _REMOTE_JOIN_TIMEOUT_S)
        self._worker = worker if worker is not None and worker.is_alive() else None
        self._stub = None
        self._channel = None
        self._started = False

    def reset(self) -> None:
        """Clear local actions and reset episode-scoped state on the server."""
        self._active.clear()
        self._reset_inference_telemetry()
        self._clear_local_state()
        if not self._started:
            return
        try:
            with self._rpc_lock:
                self._handshake()
            # Clear again so a response completed just before the handshake can
            # never leak an action from the previous episode.
            self._clear_local_state()
        except Exception as exc:
            self._fail("Could not reset remote policy session", exc)
            raise RuntimeError("Could not reset remote policy session") from exc

    def resume(self) -> None:
        self._active.set()

    def pause(self) -> None:
        self._active.clear()

    def notify_observation(self, obs: dict) -> None:
        if not self._started or not self._active.is_set() or self._failed.is_set():
            return

        with self._queue_lock:
            queue_size = len(self._action_queue)
            next_timestep = self._latest_action_timestep + 1
        if queue_size / self._actions_per_chunk > self._chunk_size_threshold:
            return

        with self._observation_lock:
            if self._observation_pending:
                return
            observation = copy(obs)
            observation["task"] = self._task
            self._latest_observation = observation
            self._observation_timestep = next_timestep
            self._observation_pending = True
        self._request_event.set()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        del obs_frame
        self._raise_if_failed()

        with self._queue_lock:
            timed_action = self._pop_action_locked()
            wait_for_first_action = (
                timed_action is None
                and not self._has_received_action
                and self._started
                and self._active.is_set()
            )

        if timed_action is None and wait_for_first_action:
            if not self._wait_for_first_action():
                self._raise_if_failed()
                if self._stop_event.is_set():
                    return None
                if self._global_shutdown_event is not None and self._global_shutdown_event.is_set():
                    return None
                error = TimeoutError(
                    "Timed out waiting for the first remote action chunk after "
                    f"{self._first_action_timeout_s:.1f}s"
                )
                self._fail("Remote policy produced no initial action", error)
                raise error

            self._raise_if_failed()
            if self._stop_event.is_set():
                return None
            with self._queue_lock:
                timed_action = self._pop_action_locked()

        if timed_action is None:
            return None
        return timed_action.get_action().detach().cpu()

    def _wait_for_first_action(self) -> bool:
        deadline = time.monotonic() + self._first_action_timeout_s
        while not self._stop_event.is_set():
            if self._global_shutdown_event is not None and self._global_shutdown_event.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._action_available.wait(timeout=min(remaining, 0.1)):
                return True
        return False

    def _pop_action_locked(self) -> Any | None:
        """Pop one action while ``_queue_lock`` is held."""
        if not self._action_queue:
            return None
        timed_action = self._action_queue.popleft()
        self._latest_action_timestep = timed_action.get_timestep()
        return timed_action

    def _inference_queue_size(self) -> int:
        with self._queue_lock:
            return len(self._action_queue)

    def _raise_if_failed(self) -> None:
        if not self._failed.is_set():
            return
        error = self._worker_error
        if error is None:
            raise RuntimeError("Remote inference worker failed")
        raise RuntimeError(f"Remote inference worker failed: {error}") from error

    def _merge_action_chunk(self, incoming_actions: list[Any]) -> None:
        with self._queue_lock:
            by_timestep = {
                action.get_timestep(): action
                for action in self._action_queue
                if action.get_timestep() > self._latest_action_timestep
            }
            for incoming in incoming_actions:
                timestep = incoming.get_timestep()
                if timestep <= self._latest_action_timestep:
                    continue
                previous = by_timestep.get(timestep)
                if previous is not None:
                    incoming.action = self._aggregate_fn(previous.get_action(), incoming.get_action())
                by_timestep[timestep] = incoming
            self._action_queue = deque(by_timestep[timestep] for timestep in sorted(by_timestep))
            if self._action_queue:
                self._has_received_action = True
                self._action_available.set()

    def _remote_loop(self) -> None:
        from lerobot.async_inference.helpers import TimedObservation

        while not self._stop_event.is_set():
            self._request_event.wait(timeout=0.1)
            if self._stop_event.is_set():
                break
            if not self._request_event.is_set() or not self._active.is_set():
                time.sleep(0.01)
                continue
            self._request_event.clear()

            with self._observation_lock:
                observation = self._latest_observation
                timestep = self._observation_timestep
            if observation is None:
                continue

            try:
                timed_observation = TimedObservation(
                    timestamp=time.time(),
                    observation=observation,
                    timestep=timestep,
                    must_go=True,
                )
                payload = pickle.dumps(timed_observation)  # nosec
                chunks = self._send_bytes_in_chunks(
                    payload,
                    self._services_pb2.Observation,
                    log_prefix="[ROLLOUT CLIENT] Observation",
                    silent=True,
                )
                with self._rpc_lock:
                    if not self._active.is_set():
                        continue
                    attempt = self._begin_inference()
                    self._stub.SendObservations(chunks, timeout=self._rpc_timeout_s)
                    with self._queue_lock:
                        first_request = not self._has_received_action
                    action_timeout_s = (
                        max(self._rpc_timeout_s, self._first_action_timeout_s)
                        if first_request
                        else self._rpc_timeout_s
                    )
                    response = self._stub.GetActions(self._services_pb2.Empty(), timeout=action_timeout_s)

                if not response.data:
                    raise RuntimeError("Remote policy server returned an empty action chunk")
                actions = pickle.loads(response.data)  # nosec
                if not actions:
                    raise RuntimeError("Remote policy server returned an empty action list")
                self._merge_action_chunk(actions)
                self._finish_inference(attempt)
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._fail("Remote policy request failed", exc)
                break
            finally:
                with self._observation_lock:
                    self._observation_pending = False

    def _fail(self, message: str, exc: Exception) -> None:
        logger.error("%s: %s", message, exc)
        if self._worker_error is None:
            self._worker_error = exc
        self._failed.set()
        self._active.clear()
        self._action_available.set()
        if self._global_shutdown_event is not None:
            self._global_shutdown_event.set()
