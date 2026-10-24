"""Incremental, atomic HDF5 recording shared by bi-Piper collection workflows.

Input joints and grippers use radians and meters. RoboTwin's ``joint_action``
contains measured follower feedback, while ``action`` stores an independently
supplied command. RGB images are JPEG byte strings, as expected by RoboTwin's
dataset converters. Only compressed JPEGs are buffered; depth and numeric
feedback are appended directly to a temporary HDF5 file.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .hdf5_replay import VECTOR_ORDER

_CAMERA_NAMES = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
_SIDES = ("left", "right")


@dataclass
class CameraSample:
    """One RGB frame, optional aligned depth in millimeters, and camera metadata."""

    rgb: np.ndarray
    depth_mm: np.ndarray | None = None
    timestamp_ns: int | None = None
    frame_number: int | None = None
    device_timestamp_ms: float | None = None
    calibration: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionSample:
    """One measured follower state and its optional source command and cameras."""

    position: np.ndarray
    timestamp_ns: int
    action: np.ndarray | None = None
    source_frame_index: int | None = None
    endposes: dict[str, np.ndarray] = field(default_factory=dict)
    cameras: dict[str, CameraSample] = field(default_factory=dict)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer.")
    result = int(value)
    if not minimum <= result <= np.iinfo(np.int64).max:
        raise ValueError(f"{label} must be in [{minimum}, {np.iinfo(np.int64).max}].")
    return result


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or infinity.")
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(value, default=_json_default, allow_nan=False, sort_keys=True)


def _metadata_attribute(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, np.generic)):
        return value
    if isinstance(value, (np.ndarray, list, tuple)):
        try:
            array = np.asarray(value)
        except ValueError:
            pass  # Ragged metadata still has a lossless JSON representation.
        else:
            if array.dtype.kind in "biuf":
                return array
    return _json(value)


def _safe_name(name: Any, label: str) -> str:
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise ValueError(f"{label} must be a nonempty HDF5 component name, got {name!r}.")
    return name


def _write_tree(group: Any, metadata: dict[str, Any]) -> None:
    """Preserve numeric calibration arrays and nested attributes without pickle."""
    stream_names = {"color_intrinsics": "color", "depth_intrinsics": "depth"}
    for name, value in metadata.items():
        name = _safe_name(name, "Calibration key")
        if isinstance(value, dict):
            _write_tree(group.create_group(stream_names.get(name, name)), value)
        elif isinstance(value, (np.ndarray, list, tuple)):
            array = np.asarray(value)
            if array.dtype.kind in "biuf":
                group.create_dataset(name, data=array)
            else:
                group.attrs[name] = _json(value)
        elif value is None:
            group.attrs[name] = "null"
        elif isinstance(value, (str, int, float, bool, np.generic)):
            group.attrs[name] = value
        else:
            group.attrs[name] = _json(value)


def _intrinsic_matrix(metadata: dict[str, Any]) -> np.ndarray:
    if "intrinsic_cv" in metadata:
        return _finite_array(metadata["intrinsic_cv"], (3, 3), "intrinsic_cv").astype(np.float32)
    if "color_intrinsics" in metadata:
        values = metadata["color_intrinsics"]
        return _finite_array(
            [[values["fx"], 0, values["ppx"]], [0, values["fy"], values["ppy"]], [0, 0, 1]],
            (3, 3),
            "color_intrinsics",
        ).astype(np.float32)
    return np.full((3, 3), np.nan, dtype=np.float32)


class HDF5EpisodeWriter:
    """Append validated samples and publish an episode only after finalization.

    Existing destinations are rejected both before acquisition and atomically
    at publication. Any append/finalize failure removes the temporary output.
    Call ``abort`` in a collector's ``finally`` block, or use this as a context
    manager. ``source_frame_index`` is optional for teleoperation; when supplied
    it must be exactly ``0, 1, ...`` throughout the episode.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float,
        gripper_max_m: float = 0.1,
        expected_frames: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().absolute()
        if os.path.lexists(self.path):
            raise FileExistsError(f"Output already exists: {self.path}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite.")
        if not math.isfinite(gripper_max_m) or gripper_max_m <= 0:
            raise ValueError("gripper_max_m must be positive and finite.")
        if expected_frames is not None:
            expected_frames = _integer(expected_frames, "expected_frames", minimum=1)
        metadata = dict(metadata or {})
        _json(metadata)
        try:
            import h5py
        except ImportError as error:  # pragma: no cover - installation dependent
            raise ImportError("HDF5 collection requires h5py; install the `piper` extra.") from error

        self._count = 0
        self._fps = float(fps)
        self._gripper_max_m = float(gripper_max_m)
        self._expected_frames = expected_frames
        self._last_timestamp_ns: int | None = None
        self._first_timestamp_ns: int | None = None
        self._source_index_available: bool | None = None
        self._camera_specs: dict[str, dict[str, Any]] | None = None
        self._jpeg: dict[str, list[bytes]] = {}
        self._datasets: dict[str, Any] = {}
        self._root = None
        self._file = None
        self._temporary: Path | None = None
        self._state = "open"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        self._temporary = Path(name)
        try:
            # mkstemp exclusively creates the destination before h5py receives
            # the open file; no predictable temporary name can be overwritten.
            self._file = os.fdopen(descriptor, "w+b")
            self._root = h5py.File(self._file, "w")
            for key, value in metadata.items():
                if value is not None:
                    self._root.attrs[key] = _metadata_attribute(value)
            self._root.attrs.update(
                {
                    "sim": metadata.get("sim", False),
                    "fps": self._fps,
                    "capture_target_fps": metadata.get("capture_target_fps", self._fps),
                    "robot": "agilex_piper",
                    "vector_order": VECTOR_ORDER,
                    "joint_action_units": "radians",
                    "gripper_encoding": "normalized_0_closed_1_open",
                    "gripper_max_m": self._gripper_max_m,
                    "joint_action_source": metadata.get("joint_action_source", "measured_follower_feedback"),
                    "action_source": metadata.get("action_source", "provided_command_when_available"),
                    "endpose_quaternion_order": "wxyz",
                    "endpose_coordinate_frame": "per_arm_base_link",
                    "endpose_translation_units": "meters",
                    "image_encoding": "jpeg",
                    "image_color_order": "RGB",
                    "timestamp_clock": "host_monotonic_ns",
                }
            )
            self._create_series("joint_action/vector", (14,))
            self._create_series("joint_action/left_arm", (6,))
            self._create_series("joint_action/left_gripper", ())
            self._create_series("joint_action/right_arm", (6,))
            self._create_series("joint_action/right_gripper", ())
            self._create_series("action", (14,))
            self._create_series("recording_metadata/action_available", (), "bool")
            for side in _SIDES:
                self._create_series(f"endpose/{side}_endpose", (7,))
                self._create_series(f"endpose/{side}_available", (), "bool")
                self._root[f"endpose/{side}_gripper"] = self._root[f"joint_action/{side}_gripper"]
            self._root.require_group("observations")["qpos"] = self._root["joint_action/vector"]
            self._root.create_group("observation")
            self._create_series("recording_metadata/timestamp_ns", (), "int64")
            self._create_series("recording_metadata/source_frame_index", (), "int64")
        except BaseException:
            self.abort()
            raise

    def _create_series(
        self, name: str, shape: tuple[int, ...], dtype: str = "float64", **kwargs: Any
    ) -> None:
        self._datasets[name] = self._root.create_dataset(
            name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype, chunks=(1, *shape), **kwargs
        )

    def _append_value(self, name: str, value: Any) -> None:
        dataset = self._datasets[name]
        dataset.resize(self._count + 1, axis=0)
        dataset[self._count] = value

    def _normalized_vector(self, value: Any, label: str) -> np.ndarray:
        vector = _finite_array(value, (14,), label).copy()
        grippers = vector[[6, 13]]
        if np.any((grippers < 0) | (grippers > self._gripper_max_m)):
            raise ValueError(f"{label} gripper values must be meters in [0, {self._gripper_max_m}].")
        vector[[6, 13]] /= self._gripper_max_m
        return vector

    def _prepare_cameras(self, cameras: dict[str, CameraSample]) -> dict[str, dict[str, Any]]:
        prepared = {}
        for name, camera in cameras.items():
            name = _safe_name(name, "Camera name")
            output_name = _CAMERA_NAMES.get(name, name)
            if output_name in prepared:
                raise ValueError(f"Camera aliases collide: {output_name}.")
            rgb = np.asarray(camera.rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1 or rgb.dtype != np.uint8:
                raise ValueError(f"Camera {name} RGB must be a nonempty HWC uint8 array with 3 channels.")
            depth = None
            if camera.depth_mm is not None:
                depth = _finite_array(camera.depth_mm, rgb.shape[:2], f"Camera {name} depth_mm")
                if np.any(depth < 0):
                    raise ValueError(f"Camera {name} depth_mm must be nonnegative millimeters.")
            timestamp = (
                None
                if camera.timestamp_ns is None
                else _integer(camera.timestamp_ns, f"Camera {name} timestamp_ns")
            )
            frame_number = (
                None
                if camera.frame_number is None
                else _integer(camera.frame_number, f"Camera {name} frame_number")
            )
            device_timestamp = camera.device_timestamp_ms
            if device_timestamp is not None and (not math.isfinite(device_timestamp) or device_timestamp < 0):
                raise ValueError(f"Camera {name} device_timestamp_ms must be finite and nonnegative.")
            spec = {
                "source_name": name,
                "shape": rgb.shape,
                "depth": depth is not None,
                "timestamp": timestamp is not None,
                "frame_number": frame_number is not None,
                "device_timestamp": device_timestamp is not None,
                "calibration_json": _json(camera.calibration),
            }
            if self._camera_specs is not None:
                previous = self._camera_specs.get(output_name)
                if previous is None or any(spec[key] != previous[key] for key in spec):
                    raise ValueError(f"Camera {name} configuration changed within the episode.")
                for key, value in (
                    ("timestamp", timestamp),
                    ("frame_number", frame_number),
                    ("device_timestamp", device_timestamp),
                ):
                    if value is not None and value <= previous[f"last_{key}"]:
                        raise ValueError(
                            f"Camera {name} {key} must be strictly increasing; reused or reordered frame."
                        )
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not ok:
                raise RuntimeError(f"Failed to JPEG-encode camera {name} frame {self._count}.")
            prepared[output_name] = {
                "spec": spec,
                "camera": camera,
                "depth": depth,
                "timestamp": timestamp,
                "frame_number": frame_number,
                "device_timestamp": device_timestamp,
                "jpeg": encoded.tobytes(),
            }
        if self._camera_specs is not None and set(prepared) != set(self._camera_specs):
            raise ValueError("Camera set must remain constant throughout the episode.")
        return prepared

    def _initialize_cameras(self, prepared: dict[str, dict[str, Any]]) -> None:
        self._camera_specs = {}
        for name, values in prepared.items():
            spec = values["spec"]
            calibration = values["camera"].calibration
            self._camera_specs[name] = spec
            self._jpeg[name] = []
            group = self._root["observation"].create_group(name)
            group.attrs.update(
                {"world_extrinsic_available": False, "source_camera_name": spec["source_name"]}
            )
            for key, shape in (("intrinsic_cv", (3, 3)), ("extrinsic_cv", (3, 4)), ("cam2world_gl", (4, 4))):
                self._create_series(f"observation/{name}/{key}", shape, "float32")
            if spec["depth"]:
                self._create_series(
                    f"observation/{name}/depth", spec["shape"][:2], compression="lzf", shuffle=True
                )
                group.attrs.update({"depth_encoding": "float64_millimeters", "depth_alignment": "color"})
                if "depth_scale_m" in calibration:
                    group.attrs["depth_scale_m"] = calibration["depth_scale_m"]
            for key, group_name, dtype in (
                ("timestamp", "camera_capture_timestamp_ns", "int64"),
                ("frame_number", "camera_frame_number", "int64"),
                ("device_timestamp", "camera_device_timestamp_ms", "float64"),
            ):
                if spec[key]:
                    self._create_series(f"recording_metadata/{group_name}/{name}", (), dtype)
            calibration_group = self._root.require_group("camera_calibration").create_group(name)
            calibration_group.attrs["json"] = spec["calibration_json"]
            _write_tree(calibration_group, calibration)
        self._root.attrs["camera_names"] = ",".join(prepared)
        depths = [name for name, values in prepared.items() if values["spec"]["depth"]]
        self._root.attrs["depth_camera_names"] = ",".join(depths)
        self._root.attrs["camera_stream_mode"] = "rgbd" if depths else "rgb" if prepared else "none"
        if depths:
            self._root.attrs.update({"depth_encoding": "float64_millimeters", "depth_alignment": "color"})

    def append(self, sample: CollectionSample) -> None:
        """Validate and append a sample; any failure aborts the entire episode."""
        if self._state != "open":
            raise RuntimeError(f"Cannot append to a {self._state} HDF5 episode.")
        try:
            if self._expected_frames is not None and self._count >= self._expected_frames:
                raise ValueError(f"Episode exceeds expected_frames={self._expected_frames}.")
            timestamp = _integer(sample.timestamp_ns, "timestamp_ns")
            if self._last_timestamp_ns is not None and timestamp <= self._last_timestamp_ns:
                raise ValueError("timestamp_ns must be strictly increasing.")
            position = self._normalized_vector(sample.position, "position")
            action = (
                np.full(14, np.nan)
                if sample.action is None
                else self._normalized_vector(sample.action, "action")
            )
            source_available = sample.source_frame_index is not None
            if self._source_index_available is not None and source_available != self._source_index_available:
                raise ValueError("source_frame_index must be supplied for every sample or none.")
            source_index = -1
            if source_available:
                source_index = _integer(sample.source_frame_index, "source_frame_index")
                if source_index != self._count:
                    raise ValueError("source_frame_index must be exactly 0..N-1.")
            if set(sample.endposes) - set(_SIDES):
                raise ValueError("endposes keys must be left or right.")
            endposes = {
                side: (
                    _finite_array(sample.endposes[side], (7,), f"{side} endpose")
                    if side in sample.endposes
                    else np.full(7, np.nan)
                )
                for side in _SIDES
            }
            cameras = self._prepare_cameras(sample.cameras)
            if self._camera_specs is None:
                self._initialize_cameras(cameras)
            self._append_value("joint_action/vector", position)
            for name, value in (
                ("left_arm", position[:6]),
                ("left_gripper", position[6]),
                ("right_arm", position[7:13]),
                ("right_gripper", position[13]),
            ):
                self._append_value(f"joint_action/{name}", value)
            self._append_value("action", action)
            self._append_value("recording_metadata/action_available", sample.action is not None)
            self._append_value("recording_metadata/timestamp_ns", timestamp)
            self._append_value("recording_metadata/source_frame_index", source_index)
            for side, value in endposes.items():
                self._append_value(f"endpose/{side}_endpose", value)
                self._append_value(f"endpose/{side}_available", side in sample.endposes)
            for name, values in cameras.items():
                self._jpeg[name].append(values["jpeg"])
                calibration = values["camera"].calibration
                self._append_value(f"observation/{name}/intrinsic_cv", _intrinsic_matrix(calibration))
                self._append_value(f"observation/{name}/extrinsic_cv", np.full((3, 4), np.nan))
                self._append_value(f"observation/{name}/cam2world_gl", np.full((4, 4), np.nan))
                if values["depth"] is not None:
                    self._append_value(f"observation/{name}/depth", values["depth"])
                for key, group_name in (
                    ("timestamp", "camera_capture_timestamp_ns"),
                    ("frame_number", "camera_frame_number"),
                    ("device_timestamp", "camera_device_timestamp_ms"),
                ):
                    if values[key] is not None:
                        self._append_value(f"recording_metadata/{group_name}/{name}", values[key])
                for key in ("timestamp", "frame_number", "device_timestamp"):
                    self._camera_specs[name][f"last_{key}"] = values[key]
            if self._first_timestamp_ns is None:
                self._first_timestamp_ns = timestamp
            self._last_timestamp_ns = timestamp
            self._source_index_available = source_available
            self._count += 1
        except BaseException:
            self.abort()
            raise

    def finalize(self) -> Path:
        """Finish fixed-width JPEG datasets and atomically publish the episode."""
        if self._state == "finalized":
            return self.path
        if self._state != "open":
            raise RuntimeError(f"Cannot finalize a {self._state} HDF5 episode.")
        try:
            if not self._count:
                raise ValueError("Cannot finalize an empty episode.")
            if self._expected_frames is not None and self._count != self._expected_frames:
                raise ValueError(
                    f"Frame count {self._count} does not match expected_frames={self._expected_frames}."
                )
            for name, payloads in self._jpeg.items():
                width = max(map(len, payloads))
                dataset = self._root[f"observation/{name}"].create_dataset(
                    "rgb", shape=(self._count,), dtype=f"S{width}"
                )
                dataset.attrs["encoding"] = "jpeg"
                for index, payload in enumerate(payloads):
                    dataset[index] = payload
            self._root.create_dataset("pointcloud", shape=(self._count, 0), dtype="float64")
            self._root["pointcloud"].attrs["available"] = False
            for side in _SIDES:
                available = bool(self._root[f"endpose/{side}_available"][:].all())
                self._root[f"endpose/{side}_endpose"].attrs["available"] = available
            self._root.attrs["num_frames"] = self._count
            self._root.attrs["source_frame_count"] = self._count if self._source_index_available else 0
            self._root.attrs["source_frame_index_available"] = self._source_index_available
            self._root.attrs["frame_alignment"] = (
                (
                    "one_synthetic_frame_per_source_frame"
                    if self._root.attrs["sim"]
                    else "one_real_frame_per_source_frame"
                )
                if self._source_index_available
                else "host_sampled"
            )
            duration_ns = self._last_timestamp_ns - self._first_timestamp_ns
            self._root.attrs["capture_fps"] = (
                (self._count - 1) * 1e9 / duration_ns
                if self._count > 1
                else self._root.attrs["capture_target_fps"]
            )
            self._root.flush()
            self._root.close()
            self._root = None
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
            os.link(self._temporary, self.path)
            self._state = "finalized"
            self._temporary.unlink()
            self._temporary = None
            self._jpeg.clear()
            return self.path
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Close and delete temporary artifacts; a published episode is retained."""
        try:
            if self._root is not None:
                self._root.close()
        finally:
            self._root = None
            try:
                if self._file is not None:
                    self._file.close()
            finally:
                self._file = None
                if self._temporary is not None:
                    self._temporary.unlink(missing_ok=True)
                    self._temporary = None
                self._jpeg.clear()
                if self._state != "finalized":
                    self._state = "aborted"

    def __enter__(self) -> HDF5EpisodeWriter:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()
