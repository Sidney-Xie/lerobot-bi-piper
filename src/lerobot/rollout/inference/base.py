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

"""Inference engine ABC.

Rollout strategies consume actions through this small interface so they
do not need to know whether inference happens inline on the control thread
or asynchronously in a background thread (RTC).
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from threading import Lock

import torch


@dataclass(frozen=True, slots=True)
class _InferenceAttempt:
    chunk_id: int
    generation: int
    started_perf_ns: int


class _InferenceTelemetryTracker:
    """Thread-safe timing state shared by all inference backends."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._generation = 0
        self._next_chunk_id = 0
        self._chunk_id: int | None = None
        self._started_ns: int | None = None
        self._finished_ns: int | None = None
        self._latency_ms: float | None = None

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._next_chunk_id = 0
            self._chunk_id = None
            self._started_ns = None
            self._finished_ns = None
            self._latency_ms = None

    def begin(self) -> _InferenceAttempt:
        started_ns = time.time_ns()
        started_perf_ns = time.perf_counter_ns()
        with self._lock:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            self._chunk_id = chunk_id
            self._started_ns = started_ns
            self._finished_ns = None
            self._latency_ms = None
            generation = self._generation
        return _InferenceAttempt(chunk_id, generation, started_perf_ns)

    def finish(self, attempt: _InferenceAttempt) -> None:
        finished_perf_ns = time.perf_counter_ns()
        finished_ns = time.time_ns()
        with self._lock:
            # Ignore a result that completed after an episode reset or after a
            # newer request superseded this one.
            if self._generation != attempt.generation or self._chunk_id != attempt.chunk_id:
                return
            self._finished_ns = finished_ns
            self._latency_ms = (finished_perf_ns - attempt.started_perf_ns) / 1_000_000.0

    def snapshot(self, queue_size: int) -> dict[str, int | float | None]:
        with self._lock:
            return {
                "inference_started_ns": self._started_ns,
                "inference_finished_ns": self._finished_ns,
                "inference_latency_ms": self._latency_ms,
                "queue_size": queue_size,
                "chunk_id": self._chunk_id,
            }


class InferenceEngine(abc.ABC):
    """Abstract backend for producing actions during rollout.

    Subclasses decide whether inference happens inline on the control
    thread or asynchronously in a background thread.  The contract is
    minimal so additional backends can be plugged in without touching
    rollout strategies.

    Lifecycle
    ---------
    ``start`` — prepare the backend (e.g. launch a background thread).
    ``stop`` — shut the backend down cleanly.
    ``reset`` — clear episode-scoped state (policy hidden state, queues…).

    Action production
    -----------------
    ``get_action(obs_frame)`` — return the next action tensor, or
    ``None`` if none is available (e.g. async queue empty).  Sync
    backends always compute from ``obs_frame``; async backends ignore
    it (they receive observations via ``notify_observation``).

    Optional hooks
    --------------
    ``notify_observation`` / ``pause`` / ``resume`` have a no-op default
    so rollout strategies can invoke them unconditionally.
    """

    def __init__(self) -> None:
        self._inference_telemetry = _InferenceTelemetryTracker()

    def _begin_inference(self) -> _InferenceAttempt:
        return self._inference_telemetry.begin()

    def _finish_inference(self, attempt: _InferenceAttempt) -> None:
        self._inference_telemetry.finish(attempt)

    def _reset_inference_telemetry(self) -> None:
        self._inference_telemetry.reset()

    def _inference_queue_size(self) -> int:
        return 0

    def get_inference_telemetry(self) -> dict[str, int | float | None]:
        """Return an atomic timing snapshot plus the backend's live queue depth."""
        return self._inference_telemetry.snapshot(self._inference_queue_size())

    @abc.abstractmethod
    def start(self) -> None:
        """Initialise the backend."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Tear the backend down."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episode-scoped state."""

    @abc.abstractmethod
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next action tensor, or ``None`` if unavailable."""

    def notify_observation(self, obs: dict) -> None:  # noqa: B027
        """Publish the latest processed observation.  Default: no-op."""

    def pause(self) -> None:  # noqa: B027
        """Pause background inference.  Default: no-op."""

    def resume(self) -> None:  # noqa: B027
        """Resume background inference.  Default: no-op."""

    @property
    def ready(self) -> bool:
        """True once the backend can produce actions (e.g. warmup done)."""
        return True

    @property
    def failed(self) -> bool:
        """True if an unrecoverable error occurred in the backend."""
        return False
