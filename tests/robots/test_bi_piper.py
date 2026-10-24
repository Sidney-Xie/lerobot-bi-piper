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

"""Hardware-free contract tests for the dual AgileX Piper robot plugin."""

# The fake intentionally mirrors piper_sdk's vendor-defined method names.
# ruff: noqa: N802

import math
import time
from collections import Counter, deque
from types import SimpleNamespace

import pytest

from lerobot.robots.bi_piper import bi_piper as bi_piper_module
from lerobot.robots.bi_piper.bi_piper import BiPiper
from lerobot.robots.bi_piper.config_bi_piper import BiPiperConfig, PiperArmConfig

SDK_PER_RADIAN = 180_000.0 / math.pi
MOTOR_KEYS = tuple(
    f"{side}_{motor}.pos"
    for side in ("left", "right")
    for motor in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
)


class FakePiperClient:
    """Small behavioral fake for the subset of piper_sdk used by BiPiper."""

    def __init__(self, side: str, events: list[tuple]):
        self.side = side
        self.events = events
        self.calls: Counter[str] = Counter()
        self.connected = True
        self.ok = True
        self.arm_status = 0
        self.joints = [0] * 6
        self.gripper = 35_000
        self.gripper_effort = 0
        self.gripper_enabled = True
        self.enabled = [True] * 6
        self.enable_feedback: deque[list[bool]] = deque()
        self.feedback_timestamp: float | None = None
        self.feedback_hz = 100.0
        self.fail_joint_write = False
        self.fail_gripper_write = False
        self.raw_can = FakeRawCan()

    def GetCanBus(self):
        return self.raw_can

    def get_connect_status(self) -> bool:
        return self.connected

    def ConnectPort(self, can_init: bool = False) -> None:
        self.calls["ConnectPort"] += 1
        self.events.append((self.side, "ConnectPort", can_init))
        self.connected = True

    def DisconnectPort(self) -> None:
        self.calls["DisconnectPort"] += 1
        self.events.append((self.side, "DisconnectPort"))
        self.connected = False

    def isOk(self) -> bool:
        self.calls["isOk"] += 1
        return self.ok

    def GetArmJointMsgs(self):
        self.calls["GetArmJointMsgs"] += 1
        state = SimpleNamespace(
            **{f"joint_{index}": value for index, value in enumerate(self.joints, start=1)}
        )
        timestamp = time.time() if self.feedback_timestamp is None else self.feedback_timestamp
        return SimpleNamespace(time_stamp=timestamp, Hz=self.feedback_hz, joint_state=state)

    def GetArmGripperMsgs(self):
        self.calls["GetArmGripperMsgs"] += 1
        foc_status = SimpleNamespace(driver_enable_status=self.gripper_enabled)
        state = SimpleNamespace(
            grippers_angle=self.gripper,
            grippers_effort=self.gripper_effort,
            foc_status=foc_status,
        )
        timestamp = time.time() if self.feedback_timestamp is None else self.feedback_timestamp
        return SimpleNamespace(time_stamp=timestamp, Hz=self.feedback_hz, gripper_state=state)

    def GetArmStatus(self):
        self.calls["GetArmStatus"] += 1
        state = SimpleNamespace(arm_status=self.arm_status)
        timestamp = time.time() if self.feedback_timestamp is None else self.feedback_timestamp
        return SimpleNamespace(time_stamp=timestamp, Hz=self.feedback_hz, arm_status=state)

    def MotionCtrl_2(self, *args: int) -> None:
        self.calls["MotionCtrl_2"] += 1
        self.events.append((self.side, "MotionCtrl_2", *args))

    def JointCtrl(self, *joints: int) -> None:
        self.calls["JointCtrl"] += 1
        self.events.append((self.side, "JointCtrl", *joints))
        if self.fail_joint_write:
            raise OSError(f"{self.side} joint write failed")

    def GripperCtrl(self, *args: int) -> None:
        self.calls["GripperCtrl"] += 1
        self.events.append((self.side, "GripperCtrl", *args))
        if self.fail_gripper_write:
            raise OSError(f"{self.side} gripper write failed")
        if args[2] == 0x01:
            self.gripper_enabled = True
        elif args[2] == 0x02:
            self.gripper_enabled = False

    def EmergencyStop(self, command: int) -> None:
        self.calls["EmergencyStop"] += 1
        self.events.append((self.side, "EmergencyStop", command))

    def EnableArm(self, motor_num: int) -> None:
        self.calls["EnableArm"] += 1
        self.events.append((self.side, "EnableArm", motor_num))

    def DisableArm(self, motor_num: int) -> None:
        self.calls["DisableArm"] += 1
        self.events.append((self.side, "DisableArm", motor_num))

    def GetArmEnableStatus(self) -> list[bool]:
        self.calls["GetArmEnableStatus"] += 1
        if self.enable_feedback:
            self.enabled = list(self.enable_feedback.popleft())
        return list(self.enabled)


def _arm_config(can_name: str, **overrides) -> PiperArmConfig:
    values = {
        "can_name": can_name,
        "gripper_unit": "normalized",
        "max_relative_target": None,
        "enable_timeout_s": 0.2,
        "feedback_timeout_s": 0.2,
        "disable_timeout_s": 0.2,
        "can_send_timeout_s": 0.003,
        "can_send_retry_interval_s": 0.001,
    }
    values.update(overrides)
    return PiperArmConfig(**values)


@pytest.fixture
def robot_factory(monkeypatch):
    monkeypatch.setattr(bi_piper_module, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(bi_piper_module, "make_cameras_from_configs", lambda configs: {})

    def make_robot(*, read_only: bool = False, **arm_overrides):
        events: list[tuple] = []
        config = BiPiperConfig(
            id="test_bi_piper",
            read_only=read_only,
            left_arm_config=_arm_config("can_left", **arm_overrides),
            right_arm_config=_arm_config("can_right", **arm_overrides),
        )
        robot = BiPiper(config)
        clients = {side: FakePiperClient(side, events) for side in ("left", "right")}
        robot._clients = clients
        robot._enabled = {"left": not read_only, "right": not read_only}
        return robot, clients, events

    return make_robot


def _full_action() -> dict[str, float]:
    left = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.25)
    right = (-0.1, 0.3, -0.4, 0.2, -0.5, 0.6, 0.75)
    return dict(zip(MOTOR_KEYS, left + right, strict=True))


def _emergency_events(events: list[tuple]) -> list[tuple]:
    return [event for event in events if event[1] == "EmergencyStop"]


class FakeCanStatus:
    SEND_MESSAGE_SUCCESS = object()
    SEND_MESSAGE_FAILED = object()
    SEND_CAN_BUS_NOT_OK = object()
    BUS_STATE_ACTIVE = object()


class FakeRawCan:
    CAN_STATUS = FakeCanStatus

    def __init__(self) -> None:
        self.statuses = deque()
        self.calls = 0

    def SendCanMessage(self, *_args, **_kwargs):
        self.calls += 1
        if self.statuses:
            return self.statuses.popleft()
        return self.CAN_STATUS.SEND_MESSAGE_SUCCESS

    def is_can_bus_ok(self):
        return self.CAN_STATUS.BUS_STATE_ACTIVE


def test_feature_order_observation_units_and_normalized_gripper(robot_factory):
    robot, clients, _ = robot_factory()
    left_radians = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)
    right_radians = (-0.1, 0.3, -0.4, 0.2, -0.5, 0.6)
    clients["left"].joints = [round(value * SDK_PER_RADIAN) for value in left_radians]
    clients["right"].joints = [round(value * SDK_PER_RADIAN) for value in right_radians]
    clients["left"].gripper = 17_500
    clients["right"].gripper = 52_500

    assert tuple(robot.action_features) == MOTOR_KEYS
    assert tuple(robot.observation_features) == MOTOR_KEYS

    observation = robot.get_observation()
    assert tuple(observation) == MOTOR_KEYS
    assert tuple(observation[key] for key in MOTOR_KEYS[:6]) == pytest.approx(left_radians, abs=1e-5)
    assert observation["left_gripper.pos"] == pytest.approx(0.25)
    assert tuple(observation[key] for key in MOTOR_KEYS[7:13]) == pytest.approx(right_radians, abs=1e-5)
    assert observation["right_gripper.pos"] == pytest.approx(0.75)


def test_bimanual_send_order_and_sdk_unit_conversion(robot_factory):
    robot, _, events = robot_factory()
    action = _full_action()

    sent = robot.send_action(action)

    assert sent == pytest.approx(action)
    writes = [event for event in events if event[1] in {"MotionCtrl_2", "JointCtrl", "GripperCtrl"}]
    assert [event[:2] for event in writes] == [
        ("left", "MotionCtrl_2"),
        ("left", "JointCtrl"),
        ("left", "GripperCtrl"),
        ("right", "MotionCtrl_2"),
        ("right", "JointCtrl"),
        ("right", "GripperCtrl"),
    ]
    assert writes[0][2:] == (0x01, 0x01, 30, 0x00, 0, 1)
    assert writes[1][2:] == tuple(round(action[key] * SDK_PER_RADIAN) for key in MOTOR_KEYS[:6])
    assert writes[2][2:] == (17_500, 1_000, 0x01, 0)
    assert writes[4][2:] == tuple(round(action[key] * SDK_PER_RADIAN) for key in MOTOR_KEYS[7:13])
    assert writes[5][2:] == (52_500, 1_000, 0x01, 0)


def test_meter_gripper_accepts_99_mm_return_position(robot_factory):
    robot, _, events = robot_factory(gripper_unit="meters", gripper_range_m=(0.0, 0.1))

    sent = robot.send_action({"left_gripper.pos": 0.099, "right_gripper.pos": 0.099})

    assert sent == pytest.approx({"left_gripper.pos": 0.099, "right_gripper.pos": 0.099})
    gripper_writes = [event for event in events if event[1] == "GripperCtrl"]
    assert [event[2] for event in gripper_writes] == [99_000, 99_000]


def test_new_client_updates_enabled_sdk_gripper_limit(monkeypatch):
    configured_ranges = []

    class FakeInterface:
        def __init__(self, **_kwargs):
            pass

        def SetSDKGripperRangeParam(self, lower: float, upper: float) -> None:
            configured_ranges.append((lower, upper))

    monkeypatch.setattr(bi_piper_module, "C_PiperInterface_V2", FakeInterface)
    arm_config = _arm_config(
        "can_left",
        gripper_unit="meters",
        gripper_range_m=(0.0, 0.1),
        start_sdk_gripper_limit=True,
    )
    robot = BiPiper.__new__(BiPiper)

    robot._new_client(arm_config)

    assert configured_ranges == [(0.0, 0.1)]


def test_motion_mode_is_refreshed_instead_of_sent_every_control_tick(robot_factory):
    robot, clients, _ = robot_factory()

    robot.send_action(_full_action())
    robot.send_action(_full_action())

    assert clients["left"].calls["MotionCtrl_2"] == 1
    assert clients["right"].calls["MotionCtrl_2"] == 1


def test_checked_sdk_sender_retries_transient_failure_and_raises_persistent_failure(robot_factory):
    robot, clients, _ = robot_factory()
    raw_can = clients["left"].raw_can
    robot._install_checked_can_sender("left")

    raw_can.statuses.extend([FakeCanStatus.SEND_MESSAGE_FAILED, FakeCanStatus.SEND_MESSAGE_SUCCESS])
    raw_can.SendCanMessage(0x159, [0] * 8)
    assert raw_can.calls == 2

    raw_can.statuses.extend([FakeCanStatus.SEND_CAN_BUS_NOT_OK])
    with pytest.raises(ConnectionError, match="left Piper CAN send failed on can_left"):
        raw_can.SendCanMessage(0x159, [0] * 8)


def test_absolute_and_per_cycle_limits_are_both_applied(robot_factory):
    per_motor_delta = {f"j{index}": 0.05 for index in range(1, 7)}
    per_motor_delta["gripper"] = 0.1
    robot, _, _ = robot_factory(max_relative_target=per_motor_delta)
    present = {f"j{index}": 0.0 for index in range(1, 7)}
    present["gripper"] = 0.5

    safe = robot._safe_side_action(
        "left",
        {"j1": 100.0, "j2": -100.0, "j3": 100.0, "gripper": 100.0},
        present,
    )

    # Absolute limits run first (j2/j3 clamp to zero, gripper to one), then delta limits.
    assert safe == pytest.approx({"j1": 0.05, "j2": 0.0, "j3": 0.0, "gripper": 0.6})


def test_integer_scalar_per_cycle_limit_is_supported(robot_factory):
    robot, _, _ = robot_factory(max_relative_target=1)
    present = {f"j{index}": 0.0 for index in range(1, 7)}
    present["gripper"] = 0.0

    safe = robot._safe_side_action("left", {"j1": 2.0}, present)

    assert safe == pytest.approx({"j1": 1.0})


def test_right_feedback_failure_prevents_any_arm_write_and_stops_both(robot_factory):
    robot, clients, events = robot_factory()
    clients["right"].ok = False

    with pytest.raises(ConnectionError, match="right Piper"):
        robot.send_action(_full_action())

    assert not any(event[1] in {"MotionCtrl_2", "JointCtrl", "GripperCtrl"} for event in events)
    assert _emergency_events(events) == [
        ("left", "EmergencyStop", 0x01),
        ("right", "EmergencyStop", 0x01),
    ]


def test_observation_feedback_failure_stops_both_arms(robot_factory):
    robot, clients, events = robot_factory()
    clients["right"].ok = False

    with pytest.raises(ConnectionError, match="right Piper"):
        robot.get_observation()

    assert _emergency_events(events) == [
        ("left", "EmergencyStop", 0x01),
        ("right", "EmergencyStop", 0x01),
    ]
    assert robot.has_hardware_fault


@pytest.mark.parametrize("field", ["joints", "gripper", "gripper_effort"])
def test_non_finite_feedback_stops_both_arms(robot_factory, field):
    robot, clients, events = robot_factory()
    if field == "joints":
        clients["left"].joints[0] = math.nan
    else:
        setattr(clients["left"], field, math.inf)

    with pytest.raises(ValueError, match="non-finite position/effort feedback"):
        robot.get_observation()

    assert _emergency_events(events) == [
        ("left", "EmergencyStop", 0x01),
        ("right", "EmergencyStop", 0x01),
    ]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["feedback_timestamp", "feedback_hz"])
def test_connect_feedback_gate_rejects_non_finite_timestamp_and_rate(robot_factory, field, value):
    robot, clients, _ = robot_factory(enable_timeout_s=0.01)
    setattr(clients["left"], field, value)

    with pytest.raises(TimeoutError, match="No fresh feedback from left Piper"):
        robot._wait_for_feedback("left")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["feedback_timestamp", "feedback_hz"])
def test_recovery_feedback_gate_rejects_non_finite_timestamp_and_rate(robot_factory, field, value):
    robot, clients, _ = robot_factory()
    setattr(clients["left"], field, value)

    assert not robot._feedback_ready_for_recovery("left", {0x00, 0x01})


def test_write_failure_emergency_stops_both_arms(robot_factory):
    robot, clients, events = robot_factory()
    clients["left"].fail_joint_write = True

    with pytest.raises(OSError, match="left joint write failed"):
        robot.send_action(_full_action())

    assert clients["right"].calls["JointCtrl"] == 0
    assert _emergency_events(events) == [
        ("left", "EmergencyStop", 0x01),
        ("right", "EmergencyStop", 0x01),
    ]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_action_emergency_stops_both_arms(robot_factory, value):
    robot, _, events = robot_factory()

    with pytest.raises(ValueError, match="must be finite"):
        robot.send_action({"left_j1.pos": value})

    assert _emergency_events(events) == [
        ("left", "EmergencyStop", 0x01),
        ("right", "EmergencyStop", 0x01),
    ]


def test_enable_and_disable_poll_explicit_joint_status(robot_factory):
    robot, clients, events = robot_factory()
    left = clients["left"]
    robot._enabled["left"] = False
    left.enabled = [False] * 6
    left.enable_feedback.extend(([False] * 6, [True] * 6))

    robot.enable_torque(sides=["left"])

    assert left.calls["EnableArm"] == 2
    assert left.calls["GetArmEnableStatus"] == 2
    assert robot._enabled["left"]

    left.enable_feedback.extend(([True] * 6, [False] * 6))
    robot._disable_arm("left")

    assert left.calls["DisableArm"] == 2
    assert left.calls["GetArmEnableStatus"] == 4
    assert not robot._enabled["left"]
    assert not any(event[1] in {"EnablePiper", "DisablePiper"} for event in events)


def test_partial_enable_timeout_disables_attempted_arm(robot_factory):
    robot, clients, events = robot_factory(enable_timeout_s=0.01, command_poll_interval_s=0.02)
    left = clients["left"]
    robot._enabled["left"] = False
    left.enabled = [False] * 6
    left.enable_feedback.extend(([True, False, False, False, False, False], [False] * 6))

    with pytest.raises(TimeoutError, match="Timed out enabling left Piper"):
        robot.enable_torque(sides=["left"])

    assert left.calls["EnableArm"] == 1
    assert left.calls["DisableArm"] == 1
    assert left.calls["GripperCtrl"] == 1
    assert not robot._enabled["left"]
    assert events.index(("left", "EnableArm", 7)) < events.index(("left", "DisableArm", 7))


def test_disconnect_disables_each_enabled_arm_before_closing_can(robot_factory):
    robot, clients, events = robot_factory()
    for client in clients.values():
        client.enable_feedback.append([False] * 6)

    robot.disconnect()

    for side in ("left", "right"):
        disable_index = events.index((side, "DisableArm", 7))
        disconnect_index = events.index((side, "DisconnectPort"))
        assert disable_index < disconnect_index
        assert not clients[side].connected


def test_disconnect_preserves_torque_when_disable_on_disconnect_is_false(robot_factory):
    robot, clients, events = robot_factory(disable_on_disconnect=False)

    robot.disconnect()

    assert not any(event[1] in {"DisableArm", "GripperCtrl"} for event in events)
    assert all(not client.connected for client in clients.values())


def test_read_only_connection_never_sends_motion_or_enable_commands(robot_factory):
    robot, clients, events = robot_factory(read_only=True)
    for client in clients.values():
        client.connected = False

    robot.connect()

    assert robot.is_connected
    assert tuple(robot.get_observation()) == MOTOR_KEYS
    assert not any(
        event[1] in {"MotionCtrl_2", "EnableArm", "DisableArm", "JointCtrl", "GripperCtrl"}
        for event in events
    )
    with pytest.raises(RuntimeError, match="read_only=True"):
        robot.enable_torque()
    with pytest.raises(RuntimeError, match="read_only=True"):
        robot.send_action(_full_action())

    robot.disconnect()

    assert all(not client.connected for client in clients.values())
    assert not any(event[1] in {"DisableArm", "GripperCtrl"} for event in events)


def test_read_only_feedback_failure_never_sends_can_commands(robot_factory):
    robot, clients, events = robot_factory(read_only=True)
    clients["right"].ok = False

    with pytest.raises(ConnectionError, match="right Piper"):
        robot.get_observation()

    assert robot.has_hardware_fault
    assert not any(
        event[1]
        in {
            "EmergencyStop",
            "MotionCtrl_2",
            "EnableArm",
            "DisableArm",
            "JointCtrl",
            "GripperCtrl",
        }
        for event in events
    )


def test_disconnect_disables_torque_after_emergency_stop(robot_factory):
    robot, clients, events = robot_factory()
    robot.emergency_stop()
    for client in clients.values():
        client.enable_feedback.append([False] * 6)

    robot.disconnect()

    assert [(side, "DisableArm", 7) for side in ("left", "right")] == [
        event for event in events if event[1] == "DisableArm"
    ]


def test_disconnect_does_not_retry_disable_when_feedback_is_unhealthy(robot_factory):
    robot, clients, _ = robot_factory()
    clients["left"].ok = False
    clients["right"].ok = False

    robot.disconnect()

    for client in clients.values():
        assert client.calls["DisableArm"] == 1
        assert client.calls["GripperCtrl"] == 1
        assert not client.connected


def test_fault_recovery_reenables_both_arms_for_safe_return(robot_factory):
    robot, clients, events = robot_factory()
    robot._hardware_fault = True
    robot._enabled = {"left": False, "right": False}

    assert robot.recover_for_safe_return()

    assert [(side, "EmergencyStop", 0x02) for side in ("left", "right")] == [
        event for event in events if event[1] == "EmergencyStop"
    ]
    for client in clients.values():
        assert client.calls["EnableArm"] == 1
    assert robot._enabled == {"left": True, "right": True}


def test_failed_fault_recovery_preserves_torque_on_disconnect(robot_factory):
    robot, clients, events = robot_factory(fault_recovery_timeout_s=0.03)
    robot._hardware_fault = True
    clients["left"].ok = False

    assert not robot.recover_for_safe_return()
    robot.disconnect()

    assert not any(event[1] in {"DisableArm", "GripperCtrl"} for event in events)
    assert all(not client.connected for client in clients.values())


def test_config_rejects_same_can_interface_for_both_arms():
    with pytest.raises(ValueError, match="different CAN interfaces"):
        BiPiperConfig(
            left_arm_config=_arm_config("can0"),
            right_arm_config=_arm_config("can0"),
        )


@pytest.mark.parametrize(
    "field",
    [
        "enable_timeout_s",
        "disable_timeout_s",
        "feedback_timeout_s",
        "command_poll_interval_s",
        "can_send_timeout_s",
        "can_send_retry_interval_s",
        "motion_mode_refresh_s",
        "fault_recovery_timeout_s",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_config_rejects_non_finite_timing_values(field, value):
    with pytest.raises(ValueError, match="positive and finite"):
        BiPiperConfig(
            left_arm_config=_arm_config("can_left", **{field: value}),
            right_arm_config=_arm_config("can_right"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", math.nan),
        ("height", math.inf),
        ("fps", -math.inf),
        ("warmup_s", math.nan),
        ("exposure", 0),
        ("exposure", math.inf),
        ("gain", -math.inf),
    ],
)
def test_config_rejects_non_finite_top_level_camera_values(field, value):
    camera_values = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "warmup_s": 2,
        "exposure": 120,
        "gain": 16,
    }
    camera_values[field] = value

    with pytest.raises(ValueError, match=field):
        BiPiperConfig(
            left_arm_config=_arm_config("can_left"),
            right_arm_config=_arm_config("can_right"),
            cameras={"cam_high": SimpleNamespace(**camera_values)},
        )
