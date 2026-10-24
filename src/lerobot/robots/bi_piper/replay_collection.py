"""Validate HDF5 sources and plan the two-pass Piper collection workflow.

The timing follows ``piper_data_collection/replay_data_collection/
robotwin_two_pass_collect.py``: replay has a fixed, uniformly scaled source
duration, and each source frame maps to exactly one collection command. This
module performs no CAN, camera, or other hardware operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .hdf5_replay import _PIPER_JOINT_LIMITS, VECTOR_ORDER, _normalize_vector_order

_GRIPPER_COLUMNS = (6, 13)
_JOINT_COLUMNS = (*range(6), *range(7, 13))
_NORMALIZED_GRIPPER = "normalized_0_closed_1_open"


def _positive(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return number


def _gripper_maximum(value: float) -> float:
    maximum = _positive(value, "gripper_max_m")
    if maximum > 0.1:
        raise ValueError("gripper_max_m cannot exceed the Piper maximum of 0.1 m.")
    return maximum


def _validate_positions(positions: np.ndarray, *, gripper_max_m: float) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14 or len(values) < 1:
        raise ValueError(f"positions must have shape (N, 14) with N >= 1, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("positions contain NaN or infinite values.")
    for side, offset in (("left", 0), ("right", 7)):
        for index, (motor, (lower, upper)) in enumerate(_PIPER_JOINT_LIMITS.items()):
            invalid = np.flatnonzero(
                (values[:, offset + index] < lower) | (values[:, offset + index] > upper)
            )
            if len(invalid):
                frame = int(invalid[0])
                raise ValueError(
                    f"Frame {frame} {side}_{motor}={values[frame, offset + index]:.9g} rad "
                    f"is outside the Piper joint limits [{lower}, {upper}]."
                )
        invalid = np.flatnonzero((values[:, offset + 6] < 0) | (values[:, offset + 6] > gripper_max_m))
        if len(invalid):
            frame = int(invalid[0])
            raise ValueError(
                f"Frame {frame} {side}_gripper={values[frame, offset + 6]:.9g} m "
                f"is outside [0, {gripper_max_m}]."
            )
    return values


@dataclass(frozen=True)
class ReplaySource:
    """Source positions in left/right joint-and-gripper order, radians/meters."""

    path: Path
    positions: np.ndarray
    source_hz: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _validate_positions(self.positions, gripper_max_m=0.1)
        object.__setattr__(self, "positions", values.copy())
        object.__setattr__(self, "source_hz", _positive(self.source_hz, "source_hz"))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def frame_count(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class ReplayPlan:
    """Interpolated commands and the unique command associated with each source frame."""

    positions: np.ndarray
    command_times_s: np.ndarray
    source_frame_command_indices: np.ndarray
    control_hz: float
    speed_scale: float

    @property
    def duration_s(self) -> float:
        return float(self.command_times_s[-1])

    @property
    def frame_count(self) -> int:
        return len(self.positions)

    @property
    def source_frame_count(self) -> int:
        return len(self.source_frame_command_indices)


def _text(value: object) -> str:
    return value.decode("utf-8").lower() if isinstance(value, bytes) else str(value).lower()


def _read_split_positions(handle: Any, paths: tuple[str, str, str, str]) -> np.ndarray:
    arrays = [np.asarray(handle[path][:], dtype=np.float64) for path in paths]
    if any(array.ndim != 2 or array.shape[1] != 6 for array in (arrays[0], arrays[2])):
        raise ValueError("Arm joint datasets must have shape (N, 6).")
    frame_count = len(arrays[0])
    for index in (1, 3):
        if arrays[index].shape not in {(frame_count,), (frame_count, 1)}:
            raise ValueError("Gripper datasets must have shape (N,) or (N, 1), matching arm frames.")
    if len(arrays[2]) != frame_count:
        raise ValueError("Left and right arm datasets must contain the same number of frames.")
    return np.column_stack(arrays)


def _read_positions(handle: Any) -> tuple[np.ndarray, str]:
    if "joint_action/vector" in handle:
        return np.asarray(handle["joint_action/vector"][:], dtype=np.float64), "joint_action/vector"
    layouts = (
        (
            "joint_action/split",
            (
                "joint_action/left_arm",
                "joint_action/left_gripper",
                "joint_action/right_arm",
                "joint_action/right_gripper",
            ),
        ),
        (
            "robotwin/action",
            (
                "action/left_arm_joint_states",
                "action/left_ee_joint_states",
                "action/right_arm_joint_states",
                "action/right_ee_joint_states",
            ),
        ),
    )
    for layout, paths in layouts:
        if all(path in handle for path in paths):
            return _read_split_positions(handle, paths), layout
    raise ValueError("HDF5 source has no supported joint_action or native RoboTwin action datasets.")


def load_replay_source(
    path: str | Path, *, source_hz: float | None = None, gripper_max_m: float = 0.1
) -> ReplaySource:
    """Read a canonical or native RoboTwin HDF5 trajectory without clipping targets.

    An explicit ``source_hz`` overrides file frequency metadata. Undeclared
    vector order uses the RoboTwin convention and records that assumption in
    ``metadata["vector_order_assumed"]``. Undeclared gripper encoding is the
    normalized RoboTwin convention; returned grippers are always in meters.
    """
    maximum = _gripper_maximum(gripper_max_m)
    if source_hz is not None:
        source_hz = _positive(source_hz, "source_hz")
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Replay source does not exist: {source_path}")
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on optional installation extras
        raise ImportError(
            "HDF5 collection requires h5py; install the piper extra with `uv sync --extra piper`."
        ) from error

    with h5py.File(source_path, "r") as handle:
        positions, layout = _read_positions(handle)
        vector_attrs = handle["joint_action/vector"].attrs if "joint_action/vector" in handle else {}
        orders = [attrs["vector_order"] for attrs in (handle.attrs, vector_attrs) if "vector_order" in attrs]
        for order in orders:
            _normalize_vector_order(order)
        units = _text(handle.attrs.get("joint_action_units", handle.attrs.get("joint_units", "radians")))
        if units not in {"rad", "radian", "radians"}:
            raise ValueError(f"Unsupported joint units {units!r}; replay requires radians.")
        encoding = _text(handle.attrs.get("gripper_encoding", _NORMALIZED_GRIPPER))
        if encoding not in {_NORMALIZED_GRIPPER, "normalized", "meters", "meter", "m"}:
            raise ValueError(f"Unsupported gripper encoding {encoding!r}.")
        frequency_origin = "override"
        if source_hz is None:
            frequency_origin = "fallback"
            frequency: Any = 25.0
            for name in ("fps", "control_hz"):
                if name in handle.attrs:
                    frequency = handle.attrs[name]
                    frequency_origin = name
                    break
            else:
                if "additional_info/frequency" in handle:
                    frequency = handle["additional_info/frequency"][()]
                    frequency_origin = "additional_info/frequency"
            if np.asarray(frequency).ndim != 0:
                raise ValueError("Source frequency must be a scalar.")
            source_hz = _positive(frequency, "source_hz")

    if positions.ndim != 2 or positions.shape[1] != 14 or len(positions) < 1:
        raise ValueError(f"positions must have shape (N, 14) with N >= 1, got {positions.shape}.")
    if encoding in {_NORMALIZED_GRIPPER, "normalized"}:
        grippers = positions[:, _GRIPPER_COLUMNS]
        if not np.isfinite(grippers).all():
            raise ValueError("Gripper targets contain NaN or infinite values.")
        if np.any((grippers < 0) | (grippers > 1)):
            raise ValueError("Normalized gripper targets must be in [0, 1].")
        positions[:, _GRIPPER_COLUMNS] *= maximum
    _validate_positions(positions, gripper_max_m=maximum)
    return ReplaySource(
        path=source_path,
        positions=positions,
        source_hz=source_hz,
        metadata={
            "layout": layout,
            "vector_order": VECTOR_ORDER,
            "vector_order_assumed": not orders,
            "joint_units": "radians",
            "gripper_encoding": encoding,
            "gripper_max_m": maximum,
            "source_hz_origin": frequency_origin,
        },
    )


def build_replay_plan(
    source: ReplaySource,
    *,
    control_hz: float = 30,
    speed_scale: float = 0.5,
    max_joint_step_rad: float = 0.15,
    max_gripper_step_m: float = 0.01,
    no_grasp: bool = False,
    gripper_max_m: float = 0.1,
) -> ReplayPlan:
    """Interpolate fixed-duration commands, rejecting unsafe steps without retiming.

    Source targets are retained exactly at their collection indices, including
    both endpoints. The caller should schedule every command against
    ``command_times_s`` and sample only at ``source_frame_command_indices``.
    """
    rate = _positive(control_hz, "control_hz")
    scale = _positive(speed_scale, "speed_scale")
    joint_step_limit = _positive(max_joint_step_rad, "max_joint_step_rad")
    gripper_step_limit = _positive(max_gripper_step_m, "max_gripper_step_m")
    maximum = _gripper_maximum(gripper_max_m)
    original = _validate_positions(source.positions, gripper_max_m=maximum).copy()
    source_rate = _positive(source.source_hz, "source_hz") * scale
    if rate + 1e-9 < source_rate:
        raise ValueError(
            f"control_hz={rate:g} is below the replayed source-frame rate {source_rate:g}; "
            "one-to-one source-frame collection is impossible."
        )
    if no_grasp:
        original[:, _GRIPPER_COLUMNS] = maximum
    if len(original) == 1:
        return ReplayPlan(original, np.zeros(1), np.zeros(1, dtype=np.int64), rate, scale)

    source_times = np.arange(len(original), dtype=np.float64) / source_rate
    duration = float(source_times[-1])
    command_intervals = max(int(round(duration * rate)), len(original) - 1)
    times = np.linspace(0.0, duration, command_intervals + 1)
    indices = np.rint(np.linspace(0, command_intervals, len(original))).astype(np.int64)
    if len(np.unique(indices)) != len(original):
        raise ValueError("Source frames do not map to unique replay commands.")
    positions = np.column_stack([np.interp(times, source_times, original[:, column]) for column in range(14)])
    positions[indices] = original
    differences = np.abs(np.diff(positions, axis=0))
    joint_step = float(np.max(differences[:, _JOINT_COLUMNS]))
    gripper_step = float(np.max(differences[:, _GRIPPER_COLUMNS]))
    if joint_step > joint_step_limit + 1e-12:
        raise ValueError(
            f"Replay joint step {joint_step:.6g} rad exceeds max_joint_step_rad={joint_step_limit:g}; "
            "reduce speed_scale or increase control_hz."
        )
    if gripper_step > gripper_step_limit + 1e-12:
        raise ValueError(
            f"Replay gripper step {gripper_step:.6g} m exceeds max_gripper_step_m={gripper_step_limit:g}; "
            "reduce speed_scale or increase control_hz."
        )
    return ReplayPlan(positions, times, indices, rate, scale)
