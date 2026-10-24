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

"""Passive recording of two Piper hardware leader/follower pairs.

One leader and one follower share each CAN bus. The controller performs the
1:1 following: the leader publishes control frames and the follower publishes
feedback frames. This module only listens to those streams. Configure roles
on isolated physical arms before joining each pair to its shared CAN bus.
CAN-ID offsets are not supported by the SDK's standard message decoder.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.import_utils import _piper_sdk_available, require_package

if TYPE_CHECKING or _piper_sdk_available:
    from piper_sdk import C_PiperInterface_V2
else:
    C_PiperInterface_V2 = None

_JOINT_FROM_SDK = math.pi / 180_000.0
_POSITION_FROM_SDK = 1e-6
_SIDES = ("left", "right")
logger = logging.getLogger(__name__)


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")


def _assert_fresh(message: Any, *, name: str, timeout_s: float) -> None:
    """Validate the SDK wrapper, whose timestamps use wall-clock seconds."""
    timestamp = float(message.time_stamp)
    hz = float(message.Hz)
    age_s = time.time() - timestamp
    if (
        not math.isfinite(timestamp)
        or timestamp <= 0
        or not math.isfinite(hz)
        or hz <= 0
        or not -timeout_s <= age_s <= timeout_s
    ):
        raise TimeoutError(f"Missing or stale {name}; check hardware roles and default CAN IDs.")


def read_endpose(client: Any, feedback_timeout_s: float = 0.5) -> NDArray[np.float64]:
    """Read follower XYZ in meters and orientation as quaternion ``w, x, y, z``.

    Piper end-pose feedback uses micrometers for translation and millidegrees
    for roll, pitch, yaw. Rotation follows ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    The function performs no CAN transmission and preserves the arm base frame.
    """
    _positive_finite("feedback_timeout_s", feedback_timeout_s)
    message = client.GetArmEndPoseMsgs()
    _assert_fresh(message, name="follower end-pose feedback", timeout_s=feedback_timeout_s)
    pose = message.end_pose
    values = np.array(
        [pose.X_axis, pose.Y_axis, pose.Z_axis, pose.RX_axis, pose.RY_axis, pose.RZ_axis],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Piper end-pose feedback contains non-finite values.")
    roll, pitch, yaw = values[3:] * (_JOINT_FROM_SDK / 2.0)
    cr, cp, cy = math.cos(roll), math.cos(pitch), math.cos(yaw)
    sr, sp, sy = math.sin(roll), math.sin(pitch), math.sin(yaw)
    quaternion = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return np.concatenate((values[:3] * _POSITION_FROM_SDK, quaternion))


@dataclass(frozen=True)
class SharedCANSample:
    """One observation/command pair, ordered left joints + jaw, right joints + jaw."""

    follower_position: NDArray[np.float64]
    leader_action: NDArray[np.float64]
    endposes: dict[str, NDArray[np.float64]]


class SharedCANBiPiper:
    """Observe hardware 1:1 following on ``can_left`` and ``can_right``.

    Opening, reading, error handling, and closing never send motion, motor
    enable/disable, role, offset, gripper, or emergency-stop commands. Stopping
    the recorder therefore leaves hardware following active. This class cannot
    establish or change the physical arms' leader/follower configuration.
    """

    def __init__(
        self,
        left_can: str = "can_left",
        right_can: str = "can_right",
        feedback_timeout_s: float = 0.5,
        gripper_max_m: float = 0.1,
        connect_timeout_s: float = 5.0,
        poll_interval_s: float = 0.01,
        judge_flag: bool = True,
    ) -> None:
        if not isinstance(left_can, str) or not left_can.strip():
            raise ValueError("left_can must be a non-empty CAN interface name.")
        if not isinstance(right_can, str) or not right_can.strip():
            raise ValueError("right_can must be a non-empty CAN interface name.")
        if left_can == right_can:
            raise ValueError("Left and right pairs must use different CAN interfaces.")
        for name, value in (
            ("feedback_timeout_s", feedback_timeout_s),
            ("gripper_max_m", gripper_max_m),
            ("connect_timeout_s", connect_timeout_s),
            ("poll_interval_s", poll_interval_s),
        ):
            _positive_finite(name, value)
        if gripper_max_m > 0.1:
            raise ValueError("gripper_max_m must not exceed 0.1 meters.")
        self.can_names = {"left": left_can, "right": right_can}
        self.feedback_timeout_s = feedback_timeout_s
        self.gripper_max_m = gripper_max_m
        self.connect_timeout_s = connect_timeout_s
        self.poll_interval_s = poll_interval_s
        self.judge_flag = judge_flag
        self._clients: dict[str, Any] = {}
        self._ever_connected: set[str] = set()

    @property
    def is_connected(self) -> bool:
        return len(self._clients) == 2 and all(
            client.get_connect_status() for client in self._clients.values()
        )

    def connect(self) -> None:
        if self._clients:
            raise RuntimeError("SharedCANBiPiper is already connected; disconnect before reconnecting.")
        require_package("piper-sdk", extra="piper", import_name="piper_sdk")
        if C_PiperInterface_V2 is None:  # pragma: no cover - protected by require_package
            raise ImportError("piper_sdk is unavailable")
        try:
            for side, can_name in self.can_names.items():
                client = C_PiperInterface_V2(
                    can_name=can_name,
                    judge_flag=self.judge_flag,
                    can_auto_init=True,
                    start_sdk_joint_limit=False,
                    start_sdk_gripper_limit=False,
                )
                # piper_sdk uses one singleton per CAN name. Attaching to a
                # connection owned by another robot would invalidate passivity
                # and would close its bus on teardown, so reject that case.
                if client.get_connect_status():
                    raise RuntimeError(
                        f"{can_name} already has a connected Piper SDK client in this process."
                    )
                self._clients[side] = client
                client.ConnectPort(can_init=side in self._ever_connected, piper_init=False)
                self._ever_connected.add(side)
            deadline = time.monotonic() + self.connect_timeout_s
            while True:
                try:
                    self.read_sample()
                    break
                except (ConnectionError, TimeoutError) as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Timed out waiting for both leader commands and follower feedback."
                        ) from error
                    time.sleep(self.poll_interval_s)
        except BaseException:
            try:
                self.disconnect()
            except Exception:
                logger.exception("Failed to close shared CAN clients after connection failure.")
            raise

    def _read_side(self, side: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        client = self._clients[side]
        if not client.isOk():
            raise ConnectionError(f"{side} shared Piper CAN receive thread is unhealthy.")
        status = client.GetArmStatus()
        _assert_fresh(status, name=f"{side} follower status", timeout_s=self.feedback_timeout_s)
        if status.arm_status.arm_status != 0:
            raise RuntimeError(f"{side} follower reports arm status 0x{status.arm_status.arm_status:02X}.")
        joint_feedback = client.GetArmJointMsgs()
        gripper_feedback = client.GetArmGripperMsgs()
        joint_control = client.GetArmJointCtrl()
        gripper_control = client.GetArmGripperCtrl()
        for kind, message in (
            ("follower joints", joint_feedback),
            ("follower gripper", gripper_feedback),
            ("leader joint commands", joint_control),
            ("leader gripper commands", gripper_control),
        ):
            _assert_fresh(message, name=f"{side} {kind}", timeout_s=self.feedback_timeout_s)
        positions = []
        for source, joints, gripper in (
            ("follower", joint_feedback.joint_state, gripper_feedback.gripper_state),
            ("leader", joint_control.joint_ctrl, gripper_control.gripper_ctrl),
        ):
            values = np.array(
                [getattr(joints, f"joint_{index}") * _JOINT_FROM_SDK for index in range(1, 7)]
                + [gripper.grippers_angle * _POSITION_FROM_SDK],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise ValueError(f"{side} {source} position contains non-finite values.")
            if not 0.0 <= values[6] <= self.gripper_max_m:
                raise ValueError(f"{side} {source} gripper opening {values[6]} exceeds the configured range.")
            positions.append(values)
        return positions[0], positions[1]

    def read_sample(self) -> SharedCANSample:
        if not self.is_connected:
            raise ConnectionError("Both shared CAN interfaces must be connected before reading.")
        follower_positions = []
        leader_actions = []
        endposes = {}
        for side in _SIDES:
            follower, leader = self._read_side(side)
            follower_positions.append(follower)
            leader_actions.append(leader)
            endposes[side] = read_endpose(self._clients[side], self.feedback_timeout_s)
        return SharedCANSample(
            follower_position=np.concatenate(follower_positions),
            leader_action=np.concatenate(leader_actions),
            endposes=endposes,
        )

    def disconnect(self) -> None:
        errors = []
        clients, self._clients = self._clients, {}
        for side, client in reversed(list(clients.items())):
            try:
                client.DisconnectPort()
            except Exception as error:
                errors.append((side, error))
        if errors:
            raise RuntimeError(f"Failed to close shared Piper CAN interfaces: {errors}")

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self.disconnect()
        except Exception:
            if not exc_info or exc_info[0] is None:
                raise
            logger.exception("Failed to close shared CAN clients while handling an earlier error.")
