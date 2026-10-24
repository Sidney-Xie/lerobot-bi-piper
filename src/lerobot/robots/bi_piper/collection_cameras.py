"""Fresh, coherent RealSense RGB(-D) samples for HDF5 collection.

Unlike independent ``read_latest`` calls, each sample contains color, aligned
depth, device identity and calibration from one frameset. Camera workers never
send robot commands and retain only their most recent frameset.
"""

from __future__ import annotations

import logging
import math
import time
from threading import Condition, Event, Thread
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _pyrealsense2_available, require_package

from .hdf5_recording import CameraSample

if TYPE_CHECKING or _pyrealsense2_available:
    import pyrealsense2 as rs
else:
    rs = None

logger = logging.getLogger(__name__)


def _intrinsics(value: Any) -> dict[str, Any]:
    return {
        "width": value.width,
        "height": value.height,
        "fx": value.fx,
        "fy": value.fy,
        "ppx": value.ppx,
        "ppy": value.ppy,
        "distortion_model": str(value.model),
        "distortion_coefficients": list(value.coeffs),
    }


class RealSenseCollectionCameras:
    """Read each camera at most once per device frame, with bounded waits."""

    def __init__(
        self,
        serials: dict[str, str],
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        use_depth: bool = True,
        frame_timeout_s: float = 0.5,
        exposure: int | None = None,
        gain: int | None = None,
    ) -> None:
        if not serials or any(not name or not serial for name, serial in serials.items()):
            raise ValueError("Camera names and serials must be non-empty")
        if len(set(serials.values())) != len(serials):
            raise ValueError("Camera serial numbers must be distinct")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (width, height, fps)
        ):
            raise ValueError("Camera dimensions and FPS must be positive integers")
        if not math.isfinite(frame_timeout_s) or frame_timeout_s <= 0:
            raise ValueError("Camera frame timeout must be positive and finite")
        if exposure is not None and (not isinstance(exposure, int) or exposure <= 0):
            raise ValueError("Exposure must be a positive integer")
        if gain is not None and (not isinstance(gain, int) or gain < 0):
            raise ValueError("Gain must be a non-negative integer")
        self.serials = dict(serials)
        self.width, self.height, self.fps = width, height, fps
        self.use_depth = use_depth
        self.frame_timeout_s = frame_timeout_s
        self.exposure, self.gain = exposure, gain
        self._pipelines: dict[str, Any] = {}
        self._threads: list[Thread] = []
        self._condition = Condition()
        self._stop = Event()
        self._latest: dict[str, CameraSample] = {}
        self._errors: dict[str, Exception] = {}
        self._last_numbers: dict[str, int] = {}

    def connect(self) -> None:
        if self._pipelines or any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Collection cameras are already connected")
        self._threads.clear()
        require_package("pyrealsense2", extra="intelrealsense")
        self._stop.clear()
        self._latest.clear()
        self._errors.clear()
        self._last_numbers.clear()
        try:
            for name, serial in self.serials.items():
                pipeline = rs.pipeline()
                self._pipelines[name] = pipeline
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
                if self.use_depth:
                    config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                profile = pipeline.start(config)
                device = profile.get_device()
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intrinsics = _intrinsics(color_profile.get_intrinsics())
                calibration: dict[str, Any] = {
                    "serial": serial,
                    "color_fps": color_profile.fps(),
                    "color_intrinsics": intrinsics,
                    "intrinsic_cv": np.array(
                        [
                            [intrinsics["fx"], 0, intrinsics["ppx"]],
                            [0, intrinsics["fy"], intrinsics["ppy"]],
                            [0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                }
                if self.use_depth:
                    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                    extrinsic = depth_profile.get_extrinsics_to(color_profile)
                    calibration.update(
                        depth_scale_m=float(device.first_depth_sensor().get_depth_scale()),
                        depth_intrinsics=_intrinsics(depth_profile.get_intrinsics()),
                        depth_to_color_rotation=np.asarray(extrinsic.rotation).reshape(3, 3, order="F"),
                        depth_to_color_translation_m=np.asarray(extrinsic.translation),
                    )
                if self.exposure is not None or self.gain is not None:
                    sensors = {
                        sensor.get_info(rs.camera_info.name): sensor for sensor in device.query_sensors()
                    }
                    if "RGB Camera" not in sensors:
                        raise RuntimeError("Manual exposure/gain requires a dedicated RGB Camera sensor")
                    sensor = sensors["RGB Camera"]
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                    if self.exposure is not None:
                        sensor.set_option(rs.option.exposure, self.exposure)
                    if self.gain is not None:
                        sensor.set_option(rs.option.gain, self.gain)
                thread = Thread(
                    target=self._capture,
                    args=(name, pipeline, calibration),
                    daemon=True,
                    name=f"bi-piper-collect-{name}",
                )
                self._threads.append(thread)
                thread.start()
            # All cameras must deliver before any robot is connected/enabled.
            self.read(timeout_s=max(5.0, self.frame_timeout_s))
        except BaseException:
            try:
                self.disconnect()
            except Exception:
                logger.exception("Camera cleanup after a failed connection also failed")
            raise

    def _capture(self, name: str, pipeline: Any, calibration: dict[str, Any]) -> None:
        last_depth_number: int | None = None
        try:
            align = rs.align(rs.stream.color) if self.use_depth else None
            while not self._stop.is_set():
                available, frames = pipeline.try_wait_for_frames(timeout_ms=200)
                if not available:
                    continue
                if align is not None:
                    frames = align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame() if self.use_depth else None
                if not color or (self.use_depth and not depth):
                    continue
                if depth is not None:
                    depth_number = depth.get_frame_number()
                    if last_depth_number is not None and depth_number <= last_depth_number:
                        continue
                    last_depth_number = depth_number
                sample = CameraSample(
                    rgb=np.asanyarray(color.get_data()).copy(),
                    depth_mm=(
                        np.asanyarray(depth.get_data()).astype(np.float64)
                        * calibration["depth_scale_m"]
                        * 1000
                        if depth is not None
                        else None
                    ),
                    timestamp_ns=time.monotonic_ns(),
                    frame_number=int(color.get_frame_number()),
                    device_timestamp_ms=float(color.get_timestamp()),
                    calibration=calibration,
                )
                with self._condition:
                    self._latest[name] = sample
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._errors[name] = error
                self._condition.notify_all()

    def read(self, *, timeout_s: float | None = None) -> dict[str, CameraSample]:
        timeout = self.frame_timeout_s if timeout_s is None else timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Camera timeout must be positive and finite")
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._stop.is_set():
                    raise TimeoutError("Collection cameras have been stopped")
                if self._errors:
                    name, error = next(iter(self._errors.items()))
                    raise RuntimeError(f"Collection camera {name} failed") from error
                now_ns = time.monotonic_ns()
                ready = all(
                    name in self._latest
                    and self._latest[name].frame_number > self._last_numbers.get(name, -1)
                    and 0 <= now_ns - self._latest[name].timestamp_ns <= timeout * 1e9
                    for name in self.serials
                )
                if ready:
                    result = {name: self._latest[name] for name in self.serials}
                    self._last_numbers.update({name: frame.frame_number for name, frame in result.items()})
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    raise TimeoutError("No fresh frames from every collection camera before the deadline")
                self._condition.wait(remaining)

    def disconnect(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        errors = []
        for name, pipeline in self._pipelines.items():
            try:
                pipeline.stop()
            except Exception as error:
                errors.append(f"{name}: {error}")
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    errors.append(f"capture worker did not terminate: {thread.name}")
        self._pipelines.clear()
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        if errors:
            raise RuntimeError(f"Failed to stop collection camera pipelines: {errors}")
