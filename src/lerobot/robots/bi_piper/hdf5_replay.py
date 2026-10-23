"""Load and replay bimanual Piper trajectories stored in HDF5 files.

The recorder used by the Piper workflows stores actions under
``joint_action/vector`` in the order ``left j1..j6, left gripper, right
j1..j6, right gripper``.  This module deliberately keeps HDF5 handling out of
the hardware class so trajectories can be validated without connecting CAN.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.robot_utils import precise_sleep

if TYPE_CHECKING:
    from .bi_piper import BiPiper

logger = logging.getLogger(__name__)

_ACTION_DIM = 14
_ARM_DIM = 6
_MOTORS = ("j1", "j2", "j3", "j4", "j5", "j6")
_VECTOR_PATH = "joint_action/vector"
VECTOR_ORDER = "left_j1..j6,left_gripper,right_j1..j6,right_gripper"
_EXPANDED_VECTOR_ORDER = ",".join(
    f"{side}_{motor}" for side in ("left", "right") for motor in (*_MOTORS, "gripper")
)
_PIPER_JOINT_LIMITS = {
    "j1": (-2.6179, 2.6179),
    "j2": (0.0, 3.14),
    "j3": (-2.967, 0.0),
    "j4": (-1.745, 1.745),
    "j5": (-1.22, 1.22),
    "j6": (-2.09439, 2.09439),
}
_MAX_GRIPPER_RANGE_M = 0.1


def _first_out_of_range(values: np.ndarray, lower: float, upper: float) -> tuple[int, float] | None:
    invalid = np.flatnonzero((values < lower) | (values > upper))
    if invalid.size == 0:
        return None
    frame_index = int(invalid[0])
    return frame_index, float(values[frame_index])


def _validate_action_ranges(
    actions: np.ndarray,
    *,
    gripper_encoding: str,
    source: Path,
    joint_limits_by_side: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> None:
    """Reject targets that cannot be represented by a physical Piper."""

    stop = len(actions) if end_frame is None else end_frame
    limits_by_side = joint_limits_by_side or {
        "left": _PIPER_JOINT_LIMITS,
        "right": _PIPER_JOINT_LIMITS,
    }
    for side, offset in (("left", 0), ("right", 7)):
        limits = limits_by_side[side]
        for motor_index, motor in enumerate(_MOTORS):
            lower, upper = limits[motor]
            violation = _first_out_of_range(actions[start_frame:stop, offset + motor_index], lower, upper)
            if violation is not None:
                relative_frame, value = violation
                frame = start_frame + relative_frame
                raise ValueError(
                    f"Trajectory {source} frame {frame} {side}_{motor}={value:.9g} rad is outside "
                    f"the configured range [{lower:.9g}, {upper:.9g}]."
                )

        if gripper_encoding == "normalized_0_closed_1_open":
            lower, upper, unit = 0.0, 1.0, "normalized"
        else:
            lower, upper, unit = 0.0, _MAX_GRIPPER_RANGE_M, "m"
        violation = _first_out_of_range(actions[start_frame:stop, offset + 6], lower, upper)
        if violation is not None:
            relative_frame, value = violation
            frame = start_frame + relative_frame
            raise ValueError(
                f"Trajectory {source} frame {frame} {side}_gripper={value:.9g} {unit} is outside "
                f"[{lower:.9g}, {upper:.9g}]."
            )


@dataclass(frozen=True)
class HDF5Trajectory:
    """Validated action trajectory and the source file's unit metadata."""

    actions: np.ndarray
    gripper_encoding: str
    joint_units: str
    vector_order: str
    source: Path

    def __post_init__(self) -> None:
        if self.actions.ndim != 2 or self.actions.shape[1] != _ACTION_DIM:
            raise ValueError(f"Expected actions with shape (frames, 14), got {self.actions.shape}.")
        if self.actions.shape[0] == 0:
            raise ValueError(f"Trajectory {self.source} contains no action frames.")
        if not np.isfinite(self.actions).all():
            raise ValueError(f"Trajectory {self.source} contains NaN or infinite action values.")
        if self.vector_order != VECTOR_ORDER:
            raise ValueError(f"Unsupported vector order {self.vector_order!r}; expected {VECTOR_ORDER!r}.")
        _validate_action_ranges(
            self.actions,
            gripper_encoding=self.gripper_encoding,
            source=self.source,
        )

    @property
    def num_frames(self) -> int:
        return int(self.actions.shape[0])

    def action(
        self,
        index: int,
        *,
        gripper_unit: str = "normalized",
        gripper_range_m: tuple[float, float] = (0.0, 0.1),
    ) -> RobotAction:
        """Return one action using the public ``BiPiper.send_action`` keys.

        ``gripper_unit`` must match each Piper arm configuration.  HDF5 files
        produced by the bimanual recorder use ``normalized_0_closed_1_open``;
        physical-meter files are also accepted.
        """

        if not 0 <= index < self.num_frames:
            raise IndexError(f"Frame index {index} is outside [0, {self.num_frames}).")
        if gripper_unit not in {"normalized", "meters"}:
            raise ValueError("gripper_unit must be 'normalized' or 'meters'.")
        lower_m, upper_m = gripper_range_m
        if not np.isfinite((lower_m, upper_m)).all() or not 0 <= lower_m < upper_m:
            raise ValueError(f"gripper_range_m must be finite and increasing, got {gripper_range_m}.")

        frame = self.actions[index]
        action: dict[str, float] = {}
        for side, offset in (("left", 0), ("right", 7)):
            for motor_index, motor in enumerate(_MOTORS):
                action[f"{side}_{motor}.pos"] = float(frame[offset + motor_index])
            gripper = float(frame[offset + 6])
            if self.gripper_encoding in {"meters", "meter", "m"} and not lower_m <= gripper <= upper_m:
                raise ValueError(
                    f"Trajectory {self.source} frame {index} {side}_gripper={gripper:.9g} m is outside "
                    f"the configured gripper_range_m [{lower_m:.9g}, {upper_m:.9g}]."
                )
            if self.gripper_encoding == "normalized_0_closed_1_open" and gripper_unit == "meters":
                gripper = lower_m + gripper * (upper_m - lower_m)
            elif self.gripper_encoding in {"meters", "meter", "m"} and gripper_unit == "normalized":
                gripper = (gripper - lower_m) / (upper_m - lower_m)
            action[f"{side}_gripper.pos"] = gripper
        return action

    def iter_actions(
        self,
        *,
        gripper_unit: str = "normalized",
        gripper_range_m: tuple[float, float] = (0.0, 0.1),
    ) -> Iterator[RobotAction]:
        for index in range(self.num_frames):
            yield self.action(index, gripper_unit=gripper_unit, gripper_range_m=gripper_range_m)


def _normalize_vector_order(value: object) -> str:
    if isinstance(value, np.ndarray):
        flattened = value.reshape(-1).tolist()
        text = ",".join(_attribute_text(item) for item in flattened)
    elif isinstance(value, (list, tuple)):
        text = ",".join(_attribute_text(item) for item in value)
    else:
        text = _attribute_text(value)
    compact = "".join(text.split())
    if compact == VECTOR_ORDER:
        return VECTOR_ORDER
    if compact.removesuffix(".pos").replace(".pos,", ",") == _EXPANDED_VECTOR_ORDER:
        return VECTOR_ORDER
    raise ValueError(f"Unsupported vector_order {text!r}; expected {VECTOR_ORDER!r}.")


def _vector_order(handle, *, required: bool) -> str:
    root_value = handle.attrs.get("vector_order")
    dataset_value = handle[_VECTOR_PATH].attrs.get("vector_order") if _VECTOR_PATH in handle else None
    if root_value is None and dataset_value is None:
        if required:
            raise ValueError(
                f"HDF5 trajectory contains {_VECTOR_PATH} but has no vector_order metadata; "
                f"expected {VECTOR_ORDER!r}."
            )
        return VECTOR_ORDER

    normalized = [
        _normalize_vector_order(value) for value in (root_value, dataset_value) if value is not None
    ]
    if len(set(normalized)) != 1:
        raise ValueError("Root and joint_action/vector vector_order metadata disagree.")
    return normalized[0]


def _read_vector(handle) -> tuple[np.ndarray, str]:
    if _VECTOR_PATH in handle:
        vector = np.asarray(handle[_VECTOR_PATH][:], dtype=np.float64)
        if vector.ndim != 2 or vector.shape[1] != _ACTION_DIM:
            raise ValueError(f"{_VECTOR_PATH} must have shape (frames, 14), got {vector.shape}.")
        return vector, _vector_order(handle, required=True)

    required = (
        "joint_action/left_arm",
        "joint_action/left_gripper",
        "joint_action/right_arm",
        "joint_action/right_gripper",
    )
    missing = [path for path in required if path not in handle]
    if missing:
        raise ValueError(f"HDF5 trajectory is missing {_VECTOR_PATH} and datasets: {missing}.")
    arrays = [np.asarray(handle[path][:], dtype=np.float64) for path in required]
    if (
        arrays[0].ndim != 2
        or arrays[0].shape[1] != _ARM_DIM
        or arrays[2].ndim != 2
        or arrays[2].shape[1] != _ARM_DIM
    ):
        raise ValueError("joint_action/*_arm datasets must have shape (frames, 6).")
    frame_count = arrays[0].shape[0]
    if any(array.shape[0] != frame_count for array in arrays):
        raise ValueError("All joint_action datasets must contain the same number of frames.")
    return np.column_stack((arrays[0], arrays[1], arrays[2], arrays[3])), _vector_order(
        handle, required=False
    )


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").lower()
    return str(value).lower()


def load_hdf5_trajectory(path: str | Path, *, clip_joint_roundoff: bool = False) -> HDF5Trajectory:
    """Load and validate a Piper HDF5 action trajectory.

    ``clip_joint_roundoff`` permits recorded joint values up to 1e-5 radians
    outside the rounded SDK limits, snapping them back inside those limits.
    Larger violations still fail validation; gripper limits are unchanged.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {source}")
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise ImportError("HDF5 replay requires h5py. Install it with `uv pip install h5py`.") from error

    with h5py.File(source, "r") as handle:
        actions, vector_order = _read_vector(handle)
        attrs = handle.attrs
        gripper_encoding = _attribute_text(attrs.get("gripper_encoding", "normalized_0_closed_1_open"))
        joint_units = _attribute_text(attrs.get("joint_action_units", attrs.get("joint_units", "radians")))

    if joint_units not in {"radian", "radians", "rad"}:
        raise ValueError(
            f"Unsupported joint action units {joint_units!r} in {source}; Piper replay expects radians."
        )
    if gripper_encoding not in {"normalized_0_closed_1_open", "meters", "meter", "m"}:
        raise ValueError(f"Unsupported gripper encoding {gripper_encoding!r} in {source}.")
    if clip_joint_roundoff:
        clipped_count = 0
        for offset in (0, 7):
            for index, (lower, upper) in enumerate(_PIPER_JOINT_LIMITS.values()):
                values = actions[:, offset + index]
                clipped = np.clip(values, lower, upper)
                delta = np.abs(values - clipped)
                within_roundoff = (delta > 0) & (delta <= 1e-5)
                clipped_count += int(within_roundoff.sum())
                values[within_roundoff] = clipped[within_roundoff]
        if clipped_count:
            logger.warning("Clipped %d joint roundoff values (<=1e-5 rad) in %s", clipped_count, source)
    return HDF5Trajectory(actions, gripper_encoding, joint_units, vector_order, source)


def _robot_replay_config(
    robot: BiPiper,
    *,
    gripper_unit: str,
    gripper_range_m: tuple[float, float] | None,
) -> tuple[tuple[float, float], dict[str, Mapping[str, tuple[float, float]]]]:
    arm_configs: Any = getattr(robot, "_arm_configs", None)
    if not isinstance(arm_configs, Mapping) or set(arm_configs) != {"left", "right"}:
        raise ValueError("HDF5 replay requires left and right Piper arm configurations.")

    configured_units = {str(config.gripper_unit) for config in arm_configs.values()}
    if len(configured_units) != 1:
        raise ValueError("Both Piper arms must use the same gripper_unit for one HDF5 trajectory.")
    configured_unit = next(iter(configured_units))
    if gripper_unit != configured_unit:
        raise ValueError(
            f"Replay gripper_unit={gripper_unit!r} does not match the robot configuration "
            f"{configured_unit!r}."
        )

    configured_ranges = {
        tuple(float(value) for value in config.gripper_range_m) for config in arm_configs.values()
    }
    if len(configured_ranges) != 1:
        raise ValueError("Both Piper arms must use the same gripper_range_m for one HDF5 trajectory.")
    configured_range = next(iter(configured_ranges))
    if gripper_range_m is not None:
        requested_range = tuple(float(value) for value in gripper_range_m)
        if requested_range != configured_range:
            raise ValueError(
                f"Replay gripper_range_m={requested_range} does not match the robot configuration "
                f"{configured_range}."
            )

    relative_limits = {
        side: getattr(config, "max_relative_target", None) for side, config in arm_configs.items()
    }
    enabled_relative_limits = {side: value for side, value in relative_limits.items() if value is not None}
    if enabled_relative_limits:
        raise ValueError(
            "HDF5 replay requires absolute target mode: set max_relative_target=None for both Piper "
            f"arms before sending commands (enabled on {sorted(enabled_relative_limits)})."
        )

    joint_limits_by_side = {side: config.joint_limits for side, config in arm_configs.items()}
    return configured_range, joint_limits_by_side


def _assert_action_not_clipped(
    requested: RobotAction,
    sent: object,
    *,
    frame_index: int,
) -> None:
    if not isinstance(sent, Mapping):
        raise RuntimeError(
            f"Robot returned {type(sent).__name__} for replay frame {frame_index}; "
            "cannot verify that the action was sent without clipping."
        )
    if set(sent) != set(requested):
        missing = sorted(set(requested) - set(sent))
        unexpected = sorted(set(sent) - set(requested))
        raise RuntimeError(
            f"Robot altered replay frame {frame_index} action keys; missing={missing}, "
            f"unexpected={unexpected}. Replay stopped."
        )
    for key, requested_value in requested.items():
        try:
            sent_value = float(sent[key])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Robot returned a non-numeric value for {key} at replay frame {frame_index}."
            ) from error
        if not math.isfinite(sent_value) or not math.isclose(
            sent_value, float(requested_value), rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError(
                f"Robot clipped or altered replay frame {frame_index} {key}: "
                f"requested={float(requested_value):.9g}, sent={sent_value:.9g}. Replay stopped."
            )


def replay_hdf5_trajectory(
    robot: BiPiper,
    trajectory: HDF5Trajectory,
    *,
    control_hz: float,
    start_frame: int = 0,
    end_frame: int | None = None,
    gripper_unit: str = "normalized",
    gripper_range_m: tuple[float, float] | None = None,
) -> int:
    """Send a validated trajectory to a connected robot at ``control_hz``.

    Deadlines are based on the trajectory start time rather than sleeping a
    fixed duration after each command, so command latency does not accumulate.
    Returns the number of frames sent.  The caller owns robot connection and
    teardown, which makes this function suitable for tests and larger flows.
    """

    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError(f"control_hz must be positive and finite, got {control_hz}.")
    if not 0 <= start_frame < trajectory.num_frames:
        raise ValueError(f"start_frame must be in [0, {trajectory.num_frames}), got {start_frame}.")
    stop = trajectory.num_frames if end_frame is None else end_frame
    if not start_frame < stop <= trajectory.num_frames:
        raise ValueError(
            f"end_frame must satisfy start_frame < end_frame <= {trajectory.num_frames}, got {stop}."
        )

    gripper_range_m, joint_limits_by_side = _robot_replay_config(
        robot,
        gripper_unit=gripper_unit,
        gripper_range_m=gripper_range_m,
    )
    _validate_action_ranges(
        trajectory.actions,
        gripper_encoding=trajectory.gripper_encoding,
        source=trajectory.source,
        joint_limits_by_side=joint_limits_by_side,
        start_frame=start_frame,
        end_frame=stop,
    )
    # Materialize conversions before commanding frame zero so an incompatible
    # meter-based gripper range cannot fail halfway through a replay.
    actions = [
        trajectory.action(index, gripper_unit=gripper_unit, gripper_range_m=gripper_range_m)
        for index in range(start_frame, stop)
    ]

    period = 1.0 / control_hz
    started_at = time.perf_counter()
    sent_count = 0
    for frame_index, action in zip(range(start_frame, stop), actions, strict=True):
        requested_action = dict(action)
        sent_action = robot.send_action(action)
        _assert_action_not_clipped(requested_action, sent_action, frame_index=frame_index)
        sent_count += 1
        deadline = started_at + (sent_count * period)
        delay = deadline - time.perf_counter()
        if delay > 0:
            precise_sleep(delay)
        elif frame_index + 1 < stop:
            logger.warning(
                "Replay command for frame %d exceeded its %.4fs period by %.4fs.",
                frame_index,
                period,
                -delay,
            )
    return sent_count
