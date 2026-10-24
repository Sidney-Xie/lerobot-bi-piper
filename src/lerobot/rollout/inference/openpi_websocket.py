"""OpenPI JAX policy-server inference over WebSocket.

The robot process deliberately does not import or load a LeRobot policy.  It
formats the current observation, sends it to OpenPI, and consumes the returned
action chunk through the common rollout engine interface.
"""

from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import copy
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine

if TYPE_CHECKING:
    from .factory import OpenPIWebSocketInferenceConfig

logger = logging.getLogger(__name__)
_JOIN_TIMEOUT_S = 3.0


class _OpenPIClient:
    """Small timeout-aware implementation of OpenPI's msgpack protocol."""

    def __init__(self, host: str, port: int, api_key: str | None, timeout_s: float) -> None:
        try:
            import websockets.sync.client
            from openpi_client import msgpack_numpy
        except ModuleNotFoundError as exc:
            missing = exc.name or "required package"
            raise ImportError(
                "OpenPI WebSocket inference requires both 'websockets' and the local "
                "'openpi-client' package. Run `uv sync --locked` and then "
                "`uv pip install --no-deps -e openpi/packages/openpi-client`. "
                f"Missing module: {missing}"
            ) from exc

        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}"
        if port is not None and "://" in uri and uri.rsplit(":", 1)[-1] != str(port):
            uri = f"{uri}:{port}"
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._ws = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=timeout_s,
            close_timeout=timeout_s,
        )
        self._packer = msgpack_numpy.Packer()
        self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=timeout_s))
        logger.info("OpenPI server metadata: %s", self.metadata)

    def infer(self, observation: dict[str, Any], timeout_s: float) -> dict:
        from openpi_client import msgpack_numpy

        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv(timeout=timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server: {response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        return

    def close(self) -> None:
        self._ws.close()


def _as_hwc_uint8(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"OpenPI image must be HWC or CHW with 3 dimensions, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"OpenPI image must have 3 channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if np.nanmax(array) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255)
    return np.asarray(array, dtype=np.uint8)


def _resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    try:
        from openpi_client import image_tools

        return np.asarray(image_tools.resize_with_pad(image, size, size))
    except ImportError:
        # Keep the client usable with a minimal openpi-client install. PIL is
        # already a LeRobot image dependency and matches OpenPI's semantics.
        from PIL import Image

        height, width = image.shape[:2]
        ratio = max(width / size, height / size)
        resized = Image.fromarray(image).resize(
            (max(1, int(width / ratio)), max(1, int(height / ratio))), Image.BILINEAR
        )
        canvas = Image.new("RGB", (size, size), 0)
        canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
        return np.asarray(canvas, dtype=np.uint8)


def _as_chw_uint8(value: Any, size: int) -> np.ndarray:
    """Convert a camera frame to the CHW uint8 layout expected by AlohaInputs."""
    return np.moveaxis(_resize_with_pad(_as_hwc_uint8(value), size), -1, 0)


class OpenPIWebSocketInferenceEngine(InferenceEngine):
    """Consume OpenPI action chunks while keeping websocket I/O serialized."""

    def __init__(
        self,
        config: OpenPIWebSocketInferenceConfig,
        *,
        hw_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        shutdown_event: Event | None = None,
    ) -> None:
        super().__init__()
        try:
            import openpi_client  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenPI WebSocket inference requires the openpi-client package. "
                "Install it from openpi/packages/openpi-client or add it to PYTHONPATH."
            ) from exc
        self._config = config
        self._hw_features = hw_features
        self._server_address = config.server_address
        self._ordered_action_keys = list(ordered_action_keys)
        self._action_dim = len(self._ordered_action_keys)
        self._task = task
        self._shutdown_event = shutdown_event
        self._client: Any = None
        self._worker: Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._stop_event = Event()
        self._active = Event()
        self._request_event = Event()
        self._action_available = Event()
        self._failed = Event()
        self._started = False
        self._observation_lock = Lock()
        self._queue_lock = Lock()
        self._latest_observation: dict[str, Any] | None = None
        self._action_queue: deque[torch.Tensor] = deque()
        self._has_received_action = False
        self._worker_error: Exception | None = None

    @property
    def ready(self) -> bool:
        return self._started and self._worker is not None and self._worker.is_alive() and not self.failed

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def _inference_queue_size(self) -> int:
        with self._queue_lock:
            return len(self._action_queue)

    def _raise_if_failed(self) -> None:
        if self._failed.is_set():
            raise RuntimeError("OpenPI WebSocket inference failed") from self._worker_error

    def _fail(self, message: str, error: Exception) -> None:
        self._worker_error = error
        self._failed.set()
        self._active.clear()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        logger.error("%s: %s", message, error)
        self._action_available.set()

    def _clear_state(self) -> None:
        with self._queue_lock:
            self._action_queue.clear()
            self._has_received_action = False
        with self._observation_lock:
            self._latest_observation = None
        self._request_event.clear()
        self._action_available.clear()

    def _parse_address(self) -> tuple[str, int]:
        address = self._server_address.removeprefix("ws://").removeprefix("wss://")
        if ":" not in address:
            return address, 8000
        host, port = address.rsplit(":", 1)
        return host, int(port)

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._failed.clear()
        self._worker_error = None
        self._reset_inference_telemetry()
        self._clear_state()
        host, port = self._parse_address()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="OpenPIRequest")
        try:
            self._client = _OpenPIClient(host, port, self._config.api_key, self._config.connect_timeout_s)
        except Exception as exc:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
            raise ConnectionError(
                f"Could not connect to OpenPI policy server at {self._server_address}"
            ) from exc
        self._started = True
        self._active.set()
        self._worker = Thread(target=self._worker_loop, daemon=True, name="OpenPIWebSocketInference")
        self._worker.start()
        logger.info("Connected to OpenPI policy server at %s", self._server_address)

    def stop(self) -> None:
        self._active.clear()
        self._stop_event.set()
        self._request_event.set()
        self._action_available.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=_JOIN_TIMEOUT_S)
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Failed to close OpenPI websocket", exc_info=True)
        self._executor = None
        self._client = None
        self._worker = None
        self._started = False

    def reset(self) -> None:
        self._active.clear()
        self._reset_inference_telemetry()
        self._clear_state()
        if self._client is not None:
            reset = getattr(self._client, "reset", None)
            if callable(reset):
                reset()

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        if not self.failed:
            self._active.set()

    def _format_observation(self, obs: dict) -> dict[str, Any]:
        def get_image(name: str) -> np.ndarray:
            key = f"observation.images.{name}"
            if key not in obs:
                raise KeyError(f"OpenPI observation is missing {key!r}")
            return _as_chw_uint8(obs[key], self._config.image_size)

        state = obs.get("observation.state")
        if state is None:
            raise KeyError("OpenPI observation is missing 'observation.state'")
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.size != self._action_dim:
            raise ValueError(
                f"OpenPI state dimension {state_array.size} does not match action dimension {self._action_dim}"
            )
        payload = {
            "images": {
                self._config.image_key: get_image(self._config.image_key),
                self._config.left_wrist_image_key: get_image(self._config.left_wrist_image_key),
                self._config.right_wrist_image_key: get_image(self._config.right_wrist_image_key),
            },
            self._config.state_key: state_array,
            self._config.prompt_key: self._task,
        }
        return payload

    def notify_observation(self, obs: dict) -> None:
        if not self._started or not self._active.is_set() or self.failed:
            return
        with self._queue_lock:
            queue_size = len(self._action_queue)
        if queue_size >= self._config.open_loop_horizon:
            return
        # notify_observation receives hardware-named fields, just like RTC.
        # Aggregate joints in feature order and namespace camera keys before
        # adapting the frame to OpenPI's wire format.
        frame = build_dataset_frame(self._hw_features, copy(obs), prefix="observation")
        formatted = self._format_observation(frame)
        with self._observation_lock:
            self._latest_observation = formatted
        self._request_event.set()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        del obs_frame
        self._raise_if_failed()
        with self._queue_lock:
            if self._action_queue:
                return self._action_queue.popleft()
            wait = self._started and self._active.is_set() and not self._has_received_action
        if wait:
            self._action_available.wait(self._config.first_action_timeout_s)
            self._raise_if_failed()
            with self._queue_lock:
                if self._action_queue:
                    return self._action_queue.popleft()
            if self._stop_event.is_set() or not self._active.is_set():
                return None
            raise TimeoutError("Timed out waiting for the first OpenPI action chunk")
        return None

    def _request_actions(self, observation: dict[str, Any]) -> list[torch.Tensor]:
        if self._client is None or self._executor is None:
            raise RuntimeError("OpenPI client is not connected")
        attempt = self._begin_inference()
        future = self._executor.submit(self._client.infer, observation, self._config.request_timeout_s)
        try:
            result = future.result(timeout=self._config.request_timeout_s)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"OpenPI inference exceeded {self._config.request_timeout_s:.1f}s") from exc
        self._finish_inference(attempt)
        if not isinstance(result, dict) or "actions" not in result:
            raise ValueError("OpenPI server response must be a mapping containing 'actions'")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != self._action_dim:
            raise ValueError(f"OpenPI actions must have shape [T, {self._action_dim}], got {actions.shape}")
        if actions.shape[0] == 0 or not np.isfinite(actions).all():
            raise ValueError("OpenPI actions must be non-empty and finite")
        return [torch.from_numpy(row.copy()) for row in actions[: self._config.action_horizon]]

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._request_event.wait(0.1)
            self._request_event.clear()
            if self._stop_event.is_set() or not self._active.is_set():
                continue
            with self._observation_lock:
                observation = self._latest_observation
                self._latest_observation = None
            if observation is None:
                continue
            try:
                actions = self._request_actions(observation)
                with self._queue_lock:
                    self._action_queue.extend(actions)
                    self._has_received_action = True
                self._action_available.set()
            except Exception as exc:
                self._fail("OpenPI WebSocket request failed", exc)
                return
