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

"""Hardware-free tests for Piper gripper effort-derived policy state."""

# The fake intentionally mirrors piper_sdk's vendor-defined method names.
# ruff: noqa: N802

import time
from collections import Counter
from types import SimpleNamespace

import pytest

from lerobot.robots.bi_piper.bi_piper import BiPiper
from lerobot.robots.bi_piper.config_bi_piper import PiperArmConfig


class GripperFeedbackClient:
    def __init__(self, *, angle: int, effort: int) -> None:
        self.angle = angle
        self.effort = effort
        self.calls: Counter[str] = Counter()

    def get_connect_status(self) -> bool:
        return True

    def isOk(self) -> bool:
        return True

    def GetArmJointMsgs(self):
        self.calls["GetArmJointMsgs"] += 1
        joints = SimpleNamespace(**{f"joint_{index}": 0 for index in range(1, 7)})
        return SimpleNamespace(time_stamp=time.time(), joint_state=joints)

    def GetArmGripperMsgs(self):
        self.calls["GetArmGripperMsgs"] += 1
        gripper = SimpleNamespace(
            grippers_angle=self.angle,
            grippers_effort=self.effort,
        )
        return SimpleNamespace(time_stamp=time.time(), gripper_state=gripper)

    def GetArmStatus(self):
        self.calls["GetArmStatus"] += 1
        return SimpleNamespace(
            time_stamp=time.time(),
            arm_status=SimpleNamespace(arm_status=0),
        )


def _robot(*, left_effort: int, right_effort: int, threshold: float = 1.0) -> BiPiper:
    """Build only the state needed by the read-only observation path."""
    robot = object.__new__(BiPiper)
    robot.id = "test-bi-piper"
    robot._arm_configs = {
        side: PiperArmConfig(
            can_name=f"can_{side}",
            gripper_unit="normalized",
            gripper_range_m=(0.0, 0.1),
            gripper_closed_effort_threshold=threshold,
        )
        for side in ("left", "right")
    }
    robot._clients = {
        "left": GripperFeedbackClient(angle=40_000, effort=left_effort),
        "right": GripperFeedbackClient(angle=60_000, effort=right_effort),
    }
    robot._last_gripper_effort = {"left": 0.0, "right": 0.0}
    robot._gripper_closed_by_effort = {"left": False, "right": False}
    robot._hardware_fault = False
    robot.cameras = {}
    return robot


def test_observation_uses_effort_and_angle_from_each_gripper_snapshot() -> None:
    robot = _robot(left_effort=1_001, right_effort=999)

    observation = robot.get_observation()

    assert observation["left_gripper.pos"] == 0.0
    assert observation["right_gripper.pos"] == pytest.approx(0.6)
    assert robot._last_gripper_effort == pytest.approx({"left": 1.001, "right": 0.999})
    assert robot._gripper_closed_by_effort == {"left": True, "right": False}
    assert robot._clients["left"].calls["GetArmGripperMsgs"] == 1
    assert robot._clients["right"].calls["GetArmGripperMsgs"] == 1


def test_effort_equal_to_threshold_does_not_override_policy_angle() -> None:
    robot = _robot(left_effort=1_000, right_effort=1_000)

    observation = robot.get_observation()

    assert observation["left_gripper.pos"] == pytest.approx(0.4)
    assert observation["right_gripper.pos"] == pytest.approx(0.6)


def test_command_side_position_keeps_measured_angle_above_effort_threshold() -> None:
    robot = _robot(left_effort=2_000, right_effort=0)

    positions = robot._read_arm_positions("left")

    assert positions["gripper"] == pytest.approx(0.4)
    assert robot._gripper_closed_by_effort["left"] is True
