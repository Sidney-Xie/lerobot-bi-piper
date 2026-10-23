#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass, field
from numbers import Integral, Real

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

PIPER_MOTOR_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
MAX_GRIPPER_RANGE_M = 0.1


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _validate_camera_config(name: str, camera: CameraConfig, *, owner: str) -> None:
    for attribute in ("width", "height", "fps"):
        value = getattr(camera, attribute, None)
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"{owner} camera '{name}' {attribute} must be a positive finite integer, got {value}."
            )

    warmup_s = getattr(camera, "warmup_s", None)
    if warmup_s is not None and (not _is_finite_real(warmup_s) or warmup_s < 0):
        raise ValueError(f"{owner} camera '{name}' warmup_s must be finite and non-negative.")

    exposure = getattr(camera, "exposure", None)
    if exposure is not None and (
        not isinstance(exposure, Integral) or isinstance(exposure, bool) or exposure <= 0
    ):
        raise ValueError(f"{owner} camera '{name}' exposure must be a positive finite integer.")

    for attribute in ("gain", "white_balance"):
        value = getattr(camera, attribute, None)
        if value is not None and (not isinstance(value, Integral) or isinstance(value, bool) or value < 0):
            raise ValueError(f"{owner} camera '{name}' {attribute} must be a non-negative finite integer.")


def _default_joint_limits() -> dict[str, tuple[float, float]]:
    """Piper SDK V2 joint limits, expressed in radians."""
    return {
        "j1": (-2.6179, 2.6179),
        "j2": (0.0, 3.14),
        "j3": (-2.967, 0.0),
        "j4": (-1.745, 1.745),
        "j5": (-1.22, 1.22),
        "j6": (-2.09439, 2.09439),
    }


def _default_max_relative_target() -> dict[str, float]:
    return {
        "j1": 0.05,
        "j2": 0.05,
        "j3": 0.05,
        "j4": 0.05,
        "j5": 0.05,
        "j6": 0.05,
        "gripper": 0.005,
    }


@dataclass(kw_only=True)
class PiperArmConfig:
    """Hardware and safety settings for one AgileX Piper arm.

    Joint features always use radians. ``gripper_unit`` controls only the
    LeRobot-facing gripper feature: ``meters`` exposes physical opening while
    ``normalized`` maps ``gripper_range_m`` to ``[0, 1]``.
    """

    can_name: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    judge_flag: bool = True
    dh_is_offset: int = 1
    start_sdk_joint_limit: bool = True
    start_sdk_gripper_limit: bool = True

    speed_percent: int = 30
    installation_pos: int = 1
    enable_on_connect: bool = True
    disable_on_disconnect: bool = True
    enable_timeout_s: float = 5.0
    disable_timeout_s: float = 1.0
    feedback_timeout_s: float = 0.5
    command_poll_interval_s: float = 0.05
    can_send_timeout_s: float = 0.02
    can_send_retry_interval_s: float = 0.001
    motion_mode_refresh_s: float = 1.0
    fault_recovery_timeout_s: float = 2.0
    preserve_torque_on_return_failure: bool = True

    gripper_range_m: tuple[float, float] = (0.0, 0.07)
    gripper_unit: str = "meters"
    gripper_effort_nm: float = 1.0
    # piper_sdk reports grippers_effort in 0.001 N·m.  Above this physical
    # threshold, expose a closed (zero) gripper position to the policy while
    # retaining the measured position for command-side safety calculations.
    gripper_closed_effort_threshold: float = 1.0

    # Maximum accepted change in one control cycle, in public feature units.
    max_relative_target: float | dict[str, float] | None = field(default_factory=_default_max_relative_target)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=_default_joint_limits)


@RobotConfig.register_subclass("bi_piper")
@dataclass(kw_only=True)
class BiPiperConfig(RobotConfig):
    """Configuration for two Piper motion-output arms."""

    id: str | None = "bi_piper"
    # Feedback-only mode. The SDK connections and cameras are opened, but no
    # motion mode, enable, action, or disconnect-disable command is sent.
    read_only: bool = False
    left_arm_config: PiperArmConfig
    right_arm_config: PiperArmConfig

    # Top-level names remain unchanged. Per-arm camera names receive side prefixes.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    emergency_stop_on_error: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.left_arm_config.can_name == self.right_arm_config.can_name:
            raise ValueError("The left and right Piper arms must use different CAN interfaces.")

        for camera_name, camera_config in self.cameras.items():
            _validate_camera_config(camera_name, camera_config, owner="top-level")

        for side, arm in (("left", self.left_arm_config), ("right", self.right_arm_config)):
            if not arm.can_name:
                raise ValueError(f"{side} can_name must not be empty.")
            if arm.dh_is_offset not in (0, 1):
                raise ValueError(f"{side} dh_is_offset must be 0 or 1.")
            if not 0 <= arm.speed_percent <= 100:
                raise ValueError(f"{side} speed_percent must be in [0, 100].")
            if arm.installation_pos not in (0, 1, 2, 3):
                raise ValueError(f"{side} installation_pos must be 0, 1, 2, or 3.")
            timing_values = {
                "enable_timeout_s": arm.enable_timeout_s,
                "disable_timeout_s": arm.disable_timeout_s,
                "feedback_timeout_s": arm.feedback_timeout_s,
                "command_poll_interval_s": arm.command_poll_interval_s,
                "can_send_timeout_s": arm.can_send_timeout_s,
                "can_send_retry_interval_s": arm.can_send_retry_interval_s,
                "motion_mode_refresh_s": arm.motion_mode_refresh_s,
                "fault_recovery_timeout_s": arm.fault_recovery_timeout_s,
            }
            invalid_timing = {
                name: value
                for name, value in timing_values.items()
                if not _is_finite_real(value) or value <= 0
            }
            if invalid_timing:
                raise ValueError(f"{side} timing values must be positive and finite: {invalid_timing}.")
            if arm.can_send_timeout_s < arm.can_send_retry_interval_s:
                raise ValueError(f"{side} can_send_timeout_s must be at least can_send_retry_interval_s.")
            if not _is_finite_real(arm.gripper_effort_nm) or not 0 <= arm.gripper_effort_nm <= 5:
                raise ValueError(f"{side} gripper_effort_nm must be in [0, 5].")
            if (
                not _is_finite_real(arm.gripper_closed_effort_threshold)
                or arm.gripper_closed_effort_threshold < 0
            ):
                raise ValueError(f"{side} gripper_closed_effort_threshold must be finite and non-negative.")

            grip_min, grip_max = arm.gripper_range_m
            if (
                not _is_finite_real(grip_min)
                or not _is_finite_real(grip_max)
                or not 0 <= grip_min < grip_max <= MAX_GRIPPER_RANGE_M
            ):
                raise ValueError(f"{side} gripper_range_m must be within [0, {MAX_GRIPPER_RANGE_M}] meters.")
            if arm.gripper_unit not in ("meters", "normalized"):
                raise ValueError(f"{side} gripper_unit must be 'meters' or 'normalized'.")

            for camera_name, camera_config in arm.cameras.items():
                _validate_camera_config(camera_name, camera_config, owner=side)

            if set(arm.joint_limits) != set(PIPER_MOTOR_NAMES[:-1]):
                raise ValueError(f"{side} joint_limits must define j1 through j6.")
            for motor, (lower, upper) in arm.joint_limits.items():
                if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                    raise ValueError(f"Invalid {side} {motor} joint limit: {(lower, upper)}")

            if isinstance(arm.max_relative_target, dict):
                if set(arm.max_relative_target) != set(PIPER_MOTOR_NAMES):
                    raise ValueError(f"{side} max_relative_target must define j1 through j6 and gripper.")
                if any(
                    not _is_finite_real(value) or value <= 0 for value in arm.max_relative_target.values()
                ):
                    raise ValueError(f"{side} max_relative_target values must be positive.")
            elif arm.max_relative_target is not None and (
                not _is_finite_real(arm.max_relative_target) or arm.max_relative_target <= 0
            ):
                raise ValueError(f"{side} max_relative_target must be positive or None.")
