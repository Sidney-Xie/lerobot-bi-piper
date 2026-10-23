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

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc|remote``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from threading import Event

import draccus

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .openpi_websocket import OpenPIWebSocketInferenceEngine
from .remote import RemoteInferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


@InferenceEngineConfig.register_subclass("remote")
@dataclass
class RemoteInferenceConfig(InferenceEngineConfig):
    """Action-chunk inference through a persistent gRPC policy server."""

    server_address: str = "localhost:8080"
    # Override this when the checkpoint has a different path on the server.
    policy_path: str | None = None
    policy_device: str | None = None
    actions_per_chunk: int = 50
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "weighted_average"
    connect_timeout_s: float = 10.0
    setup_timeout_s: float = 600.0
    rpc_timeout_s: float = 30.0
    # Loading the first chunk can be substantially slower than steady-state
    # inference (CUDA initialization, processor caches, etc.).
    first_action_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not self.server_address:
            raise ValueError("Remote inference server_address cannot be empty")
        if (
            not isinstance(self.actions_per_chunk, Integral)
            or isinstance(self.actions_per_chunk, bool)
            or self.actions_per_chunk <= 0
        ):
            raise ValueError("Remote inference actions_per_chunk must be a positive integer")
        if not _is_finite_real(self.chunk_size_threshold) or not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("Remote inference chunk_size_threshold must be finite and in [0, 1]")
        if self.aggregate_fn_name not in {"weighted_average", "latest_only", "average", "conservative"}:
            raise ValueError(f"Unknown remote aggregate_fn_name: {self.aggregate_fn_name}")
        timeouts = {
            "connect_timeout_s": self.connect_timeout_s,
            "setup_timeout_s": self.setup_timeout_s,
            "rpc_timeout_s": self.rpc_timeout_s,
            "first_action_timeout_s": self.first_action_timeout_s,
        }
        invalid_timeouts = {
            name: value for name, value in timeouts.items() if not _is_finite_real(value) or value <= 0
        }
        if invalid_timeouts:
            raise ValueError(f"Remote inference timeouts must be positive and finite: {invalid_timeouts}")


@InferenceEngineConfig.register_subclass("openpi_websocket")
@dataclass
class OpenPIWebSocketInferenceConfig(InferenceEngineConfig):
    """Action-chunk inference through an OpenPI JAX policy server."""

    server_address: str = "localhost:8000"
    api_key: str | None = None
    action_horizon: int = 50
    open_loop_horizon: int = 1
    image_size: int = 224
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 30.0
    first_action_timeout_s: float = 60.0
    image_key: str = "cam_high"
    left_wrist_image_key: str = "cam_left_wrist"
    right_wrist_image_key: str = "cam_right_wrist"
    state_key: str = "state"
    prompt_key: str = "prompt"

    def __post_init__(self) -> None:
        if not self.server_address.strip():
            raise ValueError("OpenPI WebSocket server_address cannot be empty")
        positive_ints = {
            "action_horizon": self.action_horizon,
            "open_loop_horizon": self.open_loop_horizon,
            "image_size": self.image_size,
        }
        invalid_ints = {
            name: value
            for name, value in positive_ints.items()
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
        }
        if invalid_ints:
            raise ValueError(f"OpenPI WebSocket sizes must be positive integers: {invalid_ints}")
        timeouts = {
            "connect_timeout_s": self.connect_timeout_s,
            "request_timeout_s": self.request_timeout_s,
            "first_action_timeout_s": self.first_action_timeout_s,
        }
        invalid_timeouts = {
            name: value for name, value in timeouts.items() if not _is_finite_real(value) or value <= 0
        }
        if invalid_timeouts:
            raise ValueError(f"OpenPI WebSocket timeouts must be positive and finite: {invalid_timeouts}")
        keys = {
            "image_key": self.image_key,
            "left_wrist_image_key": self.left_wrist_image_key,
            "right_wrist_image_key": self.right_wrist_image_key,
            "state_key": self.state_key,
            "prompt_key": self.prompt_key,
        }
        if any(not isinstance(value, str) or not value.strip() for value in keys.values()):
            raise ValueError("OpenPI WebSocket observation keys must be non-empty strings")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy | None,
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    policy_config: PreTrainedConfig | None = None,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
    rename_map: dict[str, str] | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, SyncInferenceConfig):
        if policy is None or preprocessor is None or postprocessor is None:
            raise ValueError("Sync inference requires a locally loaded policy and processors")
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, RTCInferenceConfig):
        if policy is None or preprocessor is None or postprocessor is None:
            raise ValueError("RTC inference requires a locally loaded policy and processors")
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    if isinstance(config, RemoteInferenceConfig):
        if policy_config is None:
            raise ValueError("Remote inference requires the policy checkpoint configuration")
        return RemoteInferenceEngine(
            config,
            policy_config=policy_config,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=config.policy_device or device,
            rename_map=rename_map or {},
            shutdown_event=shutdown_event,
        )
    if isinstance(config, OpenPIWebSocketInferenceConfig):
        return OpenPIWebSocketInferenceEngine(
            config,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            fps=fps,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
