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

import logging
import math
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _piper_sdk_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_bi_piper import PIPER_MOTOR_NAMES, BiPiperConfig, PiperArmConfig

if TYPE_CHECKING or _piper_sdk_available:
    from piper_sdk import C_PiperInterface_V2
else:
    C_PiperInterface_V2 = None

logger = logging.getLogger(__name__)

# Piper messages and commands use 0.001 degree for joints and 0.001 mm
# for the gripper. LeRobot-facing values use radians and meters/normalized.
_JOINT_TO_SDK = 180_000.0 / math.pi
_JOINT_FROM_SDK = math.pi / 180_000.0
_GRIPPER_TO_SDK = 1_000_000.0
_GRIPPER_FROM_SDK = 1.0 / _GRIPPER_TO_SDK
_GRIPPER_EFFORT_FROM_SDK = 1.0 / 1_000.0
_SIDES = ("left", "right")


class BiPiper(Robot):
    """Two AgileX Piper arms controlled through ``piper_sdk`` V2.

    Feature order is stable and matches the stage2 checkpoint contract:
    left j1..j6, left gripper, right j1..j6, right gripper. Both arms are
    guarded by absolute limits, per-cycle delta limits, feedback freshness,
    and a dual-arm emergency stop on I/O failures.
    """

    config_class = BiPiperConfig
    name = "bi_piper"

    def __init__(self, config: BiPiperConfig):
        require_package("piper-sdk", extra="piper", import_name="piper_sdk")
        super().__init__(config)
        self.config = config
        self._arm_configs = {
            "left": config.left_arm_config,
            "right": config.right_arm_config,
        }
        self._clients: dict[str, Any | None] = {"left": None, "right": None}
        self._enabled = {"left": False, "right": False}
        self._ever_connected = {"left": False, "right": False}
        self._last_motion_mode_send = {"left": 0.0, "right": 0.0}
        self._last_gripper_effort = {"left": 0.0, "right": 0.0}
        self._gripper_closed_by_effort = {"left": False, "right": False}
        self._hardware_fault = False
        self._preserve_torque_on_disconnect = False

        camera_configs = dict(config.cameras)
        for side, arm_config in self._arm_configs.items():
            for name, camera_config in arm_config.cameras.items():
                prefixed_name = f"{side}_{name}"
                if prefixed_name in camera_configs:
                    raise ValueError(f"Duplicate camera name: {prefixed_name}")
                camera_configs[prefixed_name] = camera_config
        self._camera_configs = camera_configs
        self.cameras = make_cameras_from_configs(camera_configs)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{side}_{motor}.pos": float for side in _SIDES for motor in PIPER_MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for name, camera_config in self._camera_configs.items():
            if getattr(camera_config, "use_rgb", True):
                features[name] = (camera_config.height, camera_config.width, 3)
            if getattr(camera_config, "use_depth", False):
                features[f"{name}_depth"] = (camera_config.height, camera_config.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        arms_connected = all(
            client is not None and bool(client.get_connect_status()) for client in self._clients.values()
        )
        return arms_connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def has_hardware_fault(self) -> bool:
        """Whether this connection observed an I/O or safety failure."""
        return self._hardware_fault

    def calibrate(self) -> None:
        logger.info("Piper calibration is managed by the controller; LeRobot calibration is a no-op.")

    def setup_motors(self) -> None:
        raise NotImplementedError("Configure Piper CAN and motor parameters with the vendor tools.")

    def _new_client(self, arm_config: PiperArmConfig):
        if C_PiperInterface_V2 is None:  # pragma: no cover - protected by require_package
            raise ImportError("piper_sdk is unavailable")
        client = C_PiperInterface_V2(
            can_name=arm_config.can_name,
            judge_flag=arm_config.judge_flag,
            can_auto_init=True,
            dh_is_offset=arm_config.dh_is_offset,
            start_sdk_joint_limit=arm_config.start_sdk_joint_limit,
            start_sdk_gripper_limit=arm_config.start_sdk_gripper_limit,
        )
        if arm_config.start_sdk_gripper_limit:
            # piper_sdk's built-in range defaults to 0.07 m.  Keep its safety
            # clamp enabled, but align it with the configured 70/100 mm jaw so
            # valid large-jaw targets such as 0.099 m are not truncated.
            client.SetSDKGripperRangeParam(*arm_config.gripper_range_m)
        return client

    def _install_checked_can_sender(self, side: str) -> None:
        """Turn piper_sdk's status-only CAN failures into bounded exceptions.

        piper_sdk 0.6.x catches ``python-can`` send exceptions inside
        ``C_STD_CAN.SendCanMessage``.  Public commands therefore only log an
        error and return ``None``, which otherwise lets the control loop keep
        running with a dropped frame.  Wrap that public transport hook once per
        SDK CAN singleton, retrying only the transient SEND_MESSAGE_FAILED
        status for a small bounded interval.
        """
        client = self._clients[side]
        raw_can = client.GetCanBus()
        if raw_can is None:
            raise ConnectionError(f"{side} Piper has no CAN transport.")

        arm_config = self._arm_configs[side]
        raw_can._lerobot_side = side
        raw_can._lerobot_can_name = arm_config.can_name
        raw_can._lerobot_send_timeout_s = arm_config.can_send_timeout_s
        raw_can._lerobot_send_retry_interval_s = arm_config.can_send_retry_interval_s
        if getattr(raw_can, "_lerobot_checked_sender", False):
            return

        original_send = raw_can.SendCanMessage
        success_status = raw_can.CAN_STATUS.SEND_MESSAGE_SUCCESS
        transient_failure_status = raw_can.CAN_STATUS.SEND_MESSAGE_FAILED

        def checked_send(*args, **kwargs):
            deadline = time.monotonic() + raw_can._lerobot_send_timeout_s
            attempts = 0
            while True:
                attempts += 1
                status = original_send(*args, **kwargs)
                if status is success_status:
                    return status
                if status is not transient_failure_status or time.monotonic() >= deadline:
                    bus_state = raw_can.is_can_bus_ok()
                    raise ConnectionError(
                        f"{raw_can._lerobot_side} Piper CAN send failed on "
                        f"{raw_can._lerobot_can_name}: status={status}, "
                        f"bus_state={bus_state}, attempts={attempts}"
                    )
                time.sleep(raw_can._lerobot_send_retry_interval_s)

        raw_can.SendCanMessage = checked_send
        raw_can._lerobot_checked_sender = True

    def _wait_for_feedback(self, side: str) -> None:
        client = self._clients[side]
        arm_config = self._arm_configs[side]
        deadline = time.monotonic() + arm_config.enable_timeout_s
        while time.monotonic() < deadline:
            joint = client.GetArmJointMsgs()
            gripper = client.GetArmGripperMsgs()
            status = client.GetArmStatus()
            messages = (joint, gripper, status)
            if bool(client.isOk()) and all(
                math.isfinite(float(message.time_stamp))
                and float(message.time_stamp) > 0
                and math.isfinite(float(message.Hz))
                and float(message.Hz) > 0
                for message in messages
            ):
                if status.arm_status.arm_status != 0:
                    raise RuntimeError(
                        f"{side} Piper reports arm status 0x{status.arm_status.arm_status:02X}."
                    )
                return
            time.sleep(0.01)
        raise TimeoutError(
            f"No fresh feedback from {side} Piper on {arm_config.can_name}. "
            "Verify SocketCAN bitrate, cabling, power, and motion-output/slave mode."
        )

    def _connect_arm(self, side: str) -> None:
        if self._clients[side] is None:
            self._clients[side] = self._new_client(self._arm_configs[side])
        client = self._clients[side]
        self._install_checked_can_sender(side)
        # piper_sdk documents can_init=True after DisconnectPort().
        client.ConnectPort(can_init=self._ever_connected[side])
        self._ever_connected[side] = True
        self._wait_for_feedback(side)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._hardware_fault = False
        self._preserve_torque_on_disconnect = False
        connected_sides: list[str] = []
        connected_cameras: list[str] = []
        try:
            for side in _SIDES:
                connected_sides.append(side)
                self._connect_arm(side)

            for name, camera in self.cameras.items():
                connected_cameras.append(name)
                camera.connect()

            if self.config.read_only:
                logger.info("%s connected in read-only mode; motion configuration and enable skipped.", self)
            else:
                self.configure()
                sides_to_enable = [
                    side for side, arm_config in self._arm_configs.items() if arm_config.enable_on_connect
                ]
                if sides_to_enable:
                    self.enable_torque(sides=sides_to_enable)
        except Exception:
            for name in reversed(connected_cameras):
                try:
                    if self.cameras[name].is_connected:
                        self.cameras[name].disconnect()
                except Exception:
                    logger.exception("Failed to disconnect camera %s during rollback", name)
            for side in reversed(connected_sides):
                client = self._clients[side]
                if client is None:
                    continue
                try:
                    if self._enabled[side]:
                        self._disable_arm(side)
                except Exception:
                    logger.exception("Failed to disable %s Piper during rollback", side)
                try:
                    client.DisconnectPort()
                except Exception:
                    logger.exception("Failed to disconnect %s Piper during rollback", side)
            raise

        logger.info(
            "%s connected on %s and %s",
            self,
            self._arm_configs["left"].can_name,
            self._arm_configs["right"].can_name,
        )

    def _set_motion_mode(self, side: str) -> None:
        client = self._clients[side]
        arm_config = self._arm_configs[side]
        # MotionCtrl_2 carries installation_pos and remains available in 0.6.x.
        if hasattr(client, "MotionCtrl_2"):
            client.MotionCtrl_2(
                0x01,
                0x01,
                arm_config.speed_percent,
                0x00,
                0,
                arm_config.installation_pos,
            )
        else:  # pragma: no cover - compatibility with future SDK aliases
            client.ModeCtrl(0x01, 0x01, arm_config.speed_percent, 0x00)
        self._last_motion_mode_send[side] = time.monotonic()

    def _refresh_motion_mode(self, side: str) -> None:
        if (
            time.monotonic() - self._last_motion_mode_send[side]
            >= self._arm_configs[side].motion_mode_refresh_s
        ):
            self._set_motion_mode(side)

    def configure(self) -> None:
        if self.config.read_only:
            logger.info("Skipping motion configuration for read-only %s.", self)
            return
        for side, client in self._clients.items():
            if client is None or not client.get_connect_status():
                raise ConnectionError(f"Cannot configure disconnected {side} Piper.")
            self._set_motion_mode(side)

    def _enable_arm(self, side: str) -> None:
        if self.config.read_only:
            raise RuntimeError("Cannot enable Piper torque while read_only=True.")
        client = self._clients[side]
        arm_config = self._arm_configs[side]
        if client is None or not client.get_connect_status():
            raise ConnectionError(f"Cannot enable disconnected {side} Piper.")
        deadline = time.monotonic() + arm_config.enable_timeout_s
        while time.monotonic() < deadline:
            # EnablePiper() reports the state sampled before sending its command.
            # Poll explicit feedback so success means all six arm joints ACKed.
            client.EnableArm(7)
            if all(client.GetArmEnableStatus()):
                self._enabled[side] = True
                # Enabling must hold the measured opening.  The effort-derived
                # closed state is a policy observation semantic, not a command.
                current = self._read_arm_positions(side)
                gripper_target = self._gripper_to_sdk(side, current["gripper"])
                while time.monotonic() < deadline:
                    client.GripperCtrl(
                        gripper_target,
                        round(arm_config.gripper_effort_nm * 1000),
                        0x01,
                        0,
                    )
                    gripper_state = client.GetArmGripperMsgs().gripper_state
                    if gripper_state.foc_status.driver_enable_status:
                        return
                    time.sleep(arm_config.command_poll_interval_s)
                raise TimeoutError(f"Timed out enabling {side} Piper gripper.")
            time.sleep(arm_config.command_poll_interval_s)
        raise TimeoutError(f"Timed out enabling {side} Piper on {arm_config.can_name}.")

    def enable_torque(self, sides: list[str] | None = None) -> None:
        if self.config.read_only:
            raise RuntimeError("Cannot enable Piper torque while read_only=True.")
        selected_sides = list(_SIDES) if sides is None else sides
        if not selected_sides or any(side not in _SIDES for side in selected_sides):
            raise ValueError(f"sides must contain one or more of {_SIDES}.")
        was_enabled = dict(self._enabled)
        attempted_sides: list[str] = []
        try:
            for side in selected_sides:
                attempted_sides.append(side)
                self._enable_arm(side)
        except Exception:
            # EnableArm can leave only some joints energized without ever
            # setting _enabled. Roll back every newly attempted side, not just
            # sides whose complete-enable feedback was observed.
            for side in reversed(attempted_sides):
                if was_enabled[side]:
                    continue
                try:
                    self._disable_arm(side)
                except Exception:
                    logger.exception("Failed to roll back enable for %s Piper", side)
            raise

    def _disable_arm(self, side: str) -> None:
        client = self._clients[side]
        if client is None or not client.get_connect_status():
            self._enabled[side] = False
            return
        arm_config = self._arm_configs[side]
        deadline = time.monotonic() + arm_config.disable_timeout_s
        try:
            gripper_angle = client.GetArmGripperMsgs().gripper_state.grippers_angle
        except Exception:
            # The angle is ignored by the gripper-disable command; zero lets us
            # still attempt a fail-safe disable when feedback is unavailable.
            gripper_angle = 0
        gripper_effort = round(arm_config.gripper_effort_nm * 1000)

        if not bool(client.isOk()):
            logger.warning(
                "%s Piper feedback is unhealthy; attempting one best-effort disable before closing %s.",
                side,
                arm_config.can_name,
            )
            commands = (
                ("arm", lambda: client.DisableArm(7)),
                ("gripper", lambda: client.GripperCtrl(gripper_angle, gripper_effort, 0x02, 0)),
            )
            for target, command in commands:
                try:
                    command()
                except Exception as error:
                    logger.warning("Best-effort disable of %s Piper %s failed: %s", side, target, error)
            self._enabled[side] = False
            return

        while time.monotonic() < deadline:
            client.DisableArm(7)
            client.GripperCtrl(gripper_angle, gripper_effort, 0x02, 0)
            gripper_enabled = client.GetArmGripperMsgs().gripper_state.foc_status.driver_enable_status
            if not any(client.GetArmEnableStatus()) and not gripper_enabled:
                self._enabled[side] = False
                return
            time.sleep(arm_config.command_poll_interval_s)
        gripper_enabled = client.GetArmGripperMsgs().gripper_state.foc_status.driver_enable_status
        if any(client.GetArmEnableStatus()) or gripper_enabled:
            raise TimeoutError(f"Timed out disabling {side} Piper.")
        # Feedback crossed to disabled at the deadline boundary.
        self._enabled[side] = False

    def disable_torque(self) -> None:
        errors = []
        for side in _SIDES:
            try:
                self._disable_arm(side)
            except Exception as error:
                errors.append((side, error))
        if errors:
            raise RuntimeError(f"Failed to disable Piper arms: {errors}")

    def emergency_stop(self) -> None:
        """Emergency-stop both arms. Resuming requires explicit re-enable."""
        errors = []
        for side, client in self._clients.items():
            if client is None:
                continue
            try:
                client.EmergencyStop(0x01)
            except Exception as error:
                errors.append((side, error))
            finally:
                self._enabled[side] = False
        if errors:
            raise RuntimeError(f"Failed to emergency-stop Piper arms: {errors}")

    def resume_from_emergency_stop(self) -> None:
        """Clear emergency-stop state without enabling motors."""
        errors = []
        for side, client in self._clients.items():
            if client is None:
                continue
            try:
                client.EmergencyStop(0x02)
            except Exception as error:
                errors.append((side, error))
        self._enabled = {"left": False, "right": False}
        if errors:
            raise RuntimeError(f"Failed to resume Piper arms: {errors}")

    def _feedback_ready_for_recovery(self, side: str, allowed_statuses: set[int]) -> bool:
        client = self._clients[side]
        if client is None or not client.get_connect_status() or not bool(client.isOk()):
            return False
        joint = client.GetArmJointMsgs()
        gripper = client.GetArmGripperMsgs()
        status = client.GetArmStatus()
        now = time.time()
        timeout_s = self._arm_configs[side].feedback_timeout_s
        messages = (joint, gripper, status)
        return (
            all(
                math.isfinite(float(message.time_stamp))
                and float(message.time_stamp) > 0
                and now - float(message.time_stamp) <= timeout_s
                and math.isfinite(float(message.Hz))
                and float(message.Hz) > 0
                for message in messages
            )
            and status.arm_status.arm_status in allowed_statuses
        )

    def _wait_for_recovery_feedback(self, allowed_statuses: set[int], deadline: float) -> bool:
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if all(self._feedback_ready_for_recovery(side, allowed_statuses) for side in _SIDES):
                    return True
            except Exception as error:
                # Feedback reads may fail while the CAN bus is still recovering;
                # keep polling until the deadline instead of aborting teardown.
                last_error = error
            time.sleep(0.02)
        if last_error is not None:
            logger.debug("Piper recovery feedback read failed during recovery: %s", last_error)
        return False

    def recover_for_safe_return(self) -> bool:
        """Recover both arms after an I/O timeout so teardown can return home.

        Recovery is allowed only after both CAN streams are fresh.  The method
        clears emergency stop, waits for normal status, and explicitly enables
        both arms.  It never rolls one arm back to disabled if the other arm
        fails recovery; preserving support is safer than an asymmetric drop.
        """
        timeout_s = min(config.fault_recovery_timeout_s for config in self._arm_configs.values())
        if not self._wait_for_recovery_feedback({0x00, 0x01}, time.monotonic() + timeout_s):
            logger.error("Piper feedback did not recover within %.2fs; cannot return safely.", timeout_s)
            self.preserve_torque_for_manual_recovery()
            return False

        try:
            self.resume_from_emergency_stop()
            if not self._wait_for_recovery_feedback({0x00}, time.monotonic() + timeout_s):
                raise TimeoutError("Piper status did not clear emergency stop before return.")
            for side in _SIDES:
                self._enable_arm(side)
        except Exception as error:
            logger.error("Could not prepare both Piper arms for safe return: %s", error)
            self.preserve_torque_for_manual_recovery()
            return False

        self._preserve_torque_on_disconnect = False
        logger.info("Both Piper arms recovered and are ready to return to the initial position.")
        return True

    def preserve_torque_for_manual_recovery(self) -> None:
        """Keep motor support on disconnect when an automatic return is unsafe."""
        if any(config.preserve_torque_on_return_failure for config in self._arm_configs.values()):
            self._preserve_torque_on_disconnect = True

    def _assert_healthy(self, side: str, joint_message, gripper_message) -> None:
        client = self._clients[side]
        if not client.isOk():
            raise ConnectionError(f"{side} Piper CAN feedback thread is unhealthy.")

        now = time.time()
        timeout_s = self._arm_configs[side].feedback_timeout_s
        for kind, message in (("joint", joint_message), ("gripper", gripper_message)):
            timestamp = float(message.time_stamp)
            if not math.isfinite(timestamp) or timestamp <= 0 or now - timestamp > timeout_s:
                raise TimeoutError(f"{side} Piper {kind} feedback is stale.")

        status = client.GetArmStatus()
        status_timestamp = float(status.time_stamp)
        if not math.isfinite(status_timestamp) or status_timestamp <= 0 or now - status_timestamp > timeout_s:
            raise TimeoutError(f"{side} Piper status feedback is stale.")
        if status.arm_status.arm_status != 0:
            raise RuntimeError(f"{side} Piper reports arm status 0x{status.arm_status.arm_status:02X}.")

    def _read_arm_positions(self, side: str, *, for_policy: bool = False) -> dict[str, float]:
        client = self._clients[side]
        joint_message = client.GetArmJointMsgs()
        gripper_message = client.GetArmGripperMsgs()
        self._assert_healthy(side, joint_message, gripper_message)

        joint_state = joint_message.joint_state
        positions = {
            f"j{index}": float(getattr(joint_state, f"joint_{index}")) * _JOINT_FROM_SDK
            for index in range(1, 7)
        }
        gripper_state = gripper_message.gripper_state
        gripper_m = float(gripper_state.grippers_angle) * _GRIPPER_FROM_SDK
        gripper_effort = float(gripper_state.grippers_effort) * _GRIPPER_EFFORT_FROM_SDK
        feedback_values = {**positions, "gripper": gripper_m, "gripper_effort": gripper_effort}
        non_finite = {name: value for name, value in feedback_values.items() if not math.isfinite(value)}
        if non_finite:
            raise ValueError(f"{side} Piper returned non-finite position/effort feedback: {non_finite}.")
        closed_by_effort = gripper_effort > self._arm_configs[side].gripper_closed_effort_threshold
        self._last_gripper_effort[side] = gripper_effort
        if closed_by_effort != self._gripper_closed_by_effort[side]:
            logger.info(
                "%s Piper gripper effort state changed: effort=%.3f, closed=%s",
                side,
                gripper_effort,
                closed_by_effort,
            )
            self._gripper_closed_by_effort[side] = closed_by_effort

        positions["gripper"] = (
            0.0 if for_policy and closed_by_effort else self._gripper_from_meters(side, gripper_m)
        )
        return positions

    def _gripper_from_meters(self, side: str, value_m: float) -> float:
        arm_config = self._arm_configs[side]
        if arm_config.gripper_unit == "meters":
            return value_m
        lower_m, upper_m = arm_config.gripper_range_m
        normalized = (value_m - lower_m) / (upper_m - lower_m)
        return min(max(normalized, 0.0), 1.0)

    def _gripper_to_sdk(self, side: str, value: float) -> int:
        arm_config = self._arm_configs[side]
        if arm_config.gripper_unit == "normalized":
            lower_m, upper_m = arm_config.gripper_range_m
            value = lower_m + value * (upper_m - lower_m)
        return round(value * _GRIPPER_TO_SDK)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        try:
            observation: RobotObservation = {}
            for side in _SIDES:
                for motor, value in self._read_arm_positions(side, for_policy=True).items():
                    observation[f"{side}_{motor}.pos"] = value

            for name, camera in self.cameras.items():
                if getattr(camera, "use_rgb", True):
                    observation[name] = camera.read_latest()
                if getattr(camera, "use_depth", False):
                    observation[f"{name}_depth"] = camera.read_latest_depth()
        except Exception:
            self._stop_after_error("observation failure")
            raise

        logger.debug("%s get_observation took %.1fms", self, (time.perf_counter() - start) * 1000.0)
        return observation

    def _safe_side_action(
        self,
        side: str,
        requested: dict[str, float],
        present: dict[str, float],
    ) -> dict[str, float]:
        arm_config = self._arm_configs[side]
        clipped = {}
        for motor, value in requested.items():
            if motor == "gripper":
                lower, upper = (
                    (0.0, 1.0) if arm_config.gripper_unit == "normalized" else arm_config.gripper_range_m
                )
            else:
                lower, upper = arm_config.joint_limits[motor]
            clipped[motor] = min(max(value, lower), upper)

        if arm_config.max_relative_target is None:
            return clipped

        goal_present = {motor: (value, present[motor]) for motor, value in clipped.items()}
        if isinstance(arm_config.max_relative_target, dict):
            relative_limit = {motor: arm_config.max_relative_target[motor] for motor in goal_present}
        else:
            relative_limit = float(arm_config.max_relative_target)
        return ensure_safe_goal_position(goal_present, relative_limit)

    def _send_side_action(
        self,
        side: str,
        sent: dict[str, float],
        present: dict[str, float],
    ) -> None:
        client = self._clients[side]
        arm_config = self._arm_configs[side]
        self._refresh_motion_mode(side)

        if any(motor != "gripper" for motor in sent):
            joint_targets = [sent.get(f"j{index}", present[f"j{index}"]) for index in range(1, 7)]
            client.JointCtrl(*(round(value * _JOINT_TO_SDK) for value in joint_targets))

        if "gripper" in sent:
            client.GripperCtrl(
                self._gripper_to_sdk(side, sent["gripper"]),
                round(arm_config.gripper_effort_nm * 1000),
                0x01,
                0,
            )

    def _stop_after_error(self, message: str) -> None:
        self._hardware_fault = True
        if self.config.read_only or not self.config.emergency_stop_on_error:
            return
        try:
            self.emergency_stop()
        except Exception:
            logger.exception("Emergency stop after %s also failed", message)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.read_only:
            raise RuntimeError("Cannot send Piper actions while read_only=True.")
        unknown = set(action) - set(self.action_features)
        if unknown:
            raise ValueError(f"Unknown BiPiper action keys: {sorted(unknown)}")
        if not action:
            return {}

        numeric_action: dict[str, float] = {}
        try:
            for key, value in action.items():
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise ValueError(f"Action {key} must be finite, got {value}.")
                numeric_action[key] = numeric_value
        except (TypeError, ValueError):
            self._stop_after_error("invalid action")
            raise

        selected_sides = [
            side for side in _SIDES if any(key.startswith(f"{side}_") for key in numeric_action)
        ]
        disabled = [side for side in selected_sides if not self._enabled[side]]
        if disabled:
            raise RuntimeError(f"Cannot command disabled Piper arms: {disabled}")

        try:
            # Read both arms before writing either to prevent a partial bimanual command.
            present = {side: self._read_arm_positions(side) for side in _SIDES}
            safe_by_side = {}
            for side in selected_sides:
                requested = {
                    key.removeprefix(f"{side}_").removesuffix(".pos"): value
                    for key, value in numeric_action.items()
                    if key.startswith(f"{side}_")
                }
                safe_by_side[side] = self._safe_side_action(side, requested, present[side])
        except Exception:
            self._stop_after_error("feedback failure")
            raise

        try:
            for side in selected_sides:
                self._send_side_action(side, safe_by_side[side], present[side])
        except Exception:
            self._stop_after_error("command failure")
            raise

        return {
            f"{side}_{motor}.pos": value
            for side in selected_sides
            for motor, value in safe_by_side[side].items()
        }

    def disconnect(self) -> None:
        errors = []
        for name, camera in self.cameras.items():
            if not camera.is_connected:
                continue
            try:
                camera.disconnect()
            except Exception as error:
                errors.append((name, error))

        for side in _SIDES:
            client = self._clients[side]
            if client is None or not client.get_connect_status():
                continue
            # EmergencyStop halts motion but does not remove motor torque. Keep or
            # remove that torque according to the explicit teardown policy, even
            # when an earlier emergency stop already cleared motion authorization.
            if self.config.read_only:
                # This connection never changed motion mode or enabled torque,
                # so teardown must not alter the controller's existing state.
                self._enabled[side] = False
            elif self._preserve_torque_on_disconnect:
                logger.critical(
                    "Leaving %s Piper motor torque engaged while closing %s because automatic return "
                    "was not safe. Support the arm and disable it manually after CAN recovery.",
                    side,
                    self._arm_configs[side].can_name,
                )
                self._enabled[side] = False
            elif self._arm_configs[side].disable_on_disconnect:
                try:
                    self._disable_arm(side)
                except Exception as error:
                    errors.append((side, error))
            elif self._enabled[side]:
                logger.warning("Leaving %s Piper enabled while disconnecting CAN.", side)
                self._enabled[side] = False
            try:
                client.DisconnectPort()
            except Exception as error:
                errors.append((side, error))

        if errors:
            raise RuntimeError(f"BiPiper disconnect completed with errors: {errors}")
        logger.info("%s disconnected.", self)
