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

"""Hardware-free tests for passive hardware leader/follower collection."""

# Vendor methods use camel case.
# ruff: noqa: N802

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.bi_piper import shared_can
from lerobot.robots.bi_piper.shared_can import SharedCANBiPiper, read_endpose


class FakePassiveClient:
    def __init__(self, can_name: str) -> None:
        self.can_name = can_name
        self.connected = False
        self.ok = True
        self.arm_status = 0
        self.calls: list[tuple] = []
        self.timestamps: dict[str, float] = {}
        self.hz: dict[str, float] = {}
        self.joints = [90_000, -45_000, 0, 30_000, 0, -15_000]
        self.controls = [value + 100 for value in self.joints]
        self.gripper = 80_000
        self.target_gripper = 90_000
        self.pose = [100_000, -200_000, 300_000, 0, 0, 90_000]
        self.fail_connect = False

    def __getattr__(self, name: str):
        if name.startswith(
            ("Enable", "Disable", "Motion", "Mode", "Joint", "Gripper", "Master", "Emergency", "Set")
        ):
            raise AssertionError(f"Passive recorder attempted {name}")
        raise AttributeError(name)

    def get_connect_status(self) -> bool:
        return self.connected

    def ConnectPort(self, *, can_init: bool, piper_init: bool) -> None:
        self.calls.append(("ConnectPort", can_init, piper_init))
        if self.fail_connect:
            raise OSError("connect failed")
        self.connected = True

    def DisconnectPort(self) -> None:
        self.calls.append(("DisconnectPort",))
        self.connected = False

    def isOk(self) -> bool:
        return self.ok

    def _message(self, kind: str, **payload):
        return SimpleNamespace(
            time_stamp=self.timestamps.get(kind, time.time()), Hz=self.hz.get(kind, 100.0), **payload
        )

    def GetArmStatus(self):
        return self._message("status", arm_status=SimpleNamespace(arm_status=self.arm_status))

    def GetArmJointMsgs(self):
        return self._message(
            "joint", joint_state=SimpleNamespace(**{f"joint_{i}": v for i, v in enumerate(self.joints, 1)})
        )

    def GetArmGripperMsgs(self):
        return self._message(
            "gripper", gripper_state=SimpleNamespace(grippers_angle=self.gripper, grippers_effort=5000)
        )

    def GetArmJointCtrl(self):
        return self._message(
            "control", joint_ctrl=SimpleNamespace(**{f"joint_{i}": v for i, v in enumerate(self.controls, 1)})
        )

    def GetArmGripperCtrl(self):
        return self._message(
            "gripper_control", gripper_ctrl=SimpleNamespace(grippers_angle=self.target_gripper)
        )

    def GetArmEndPoseMsgs(self):
        return self._message(
            "pose",
            end_pose=SimpleNamespace(
                **dict(
                    zip(
                        ("X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis"),
                        self.pose,
                        strict=True,
                    )
                )
            ),
        )


@pytest.fixture
def passive_factory(monkeypatch):
    clients = {name: FakePassiveClient(name) for name in ("can_left", "can_right")}
    constructor_calls = []

    def constructor(**kwargs):
        constructor_calls.append(kwargs)
        return clients[kwargs["can_name"]]

    monkeypatch.setattr(shared_can, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(shared_can, "C_PiperInterface_V2", constructor)

    def make(**kwargs):
        return SharedCANBiPiper(connect_timeout_s=0.002, poll_interval_s=0.001, **kwargs)

    return make, clients, constructor_calls


def test_shared_pairs_are_passive_and_preserve_physical_feedback(passive_factory):
    make, clients, constructors = passive_factory
    clients["can_right"].joints[0] = -90_000
    recorder = make()
    with recorder:
        assert recorder.is_connected
        sample = recorder.read_sample()
        assert sample.follower_position.shape == (14,)
        assert sample.leader_action.shape == (14,)
        np.testing.assert_allclose(sample.follower_position[[0, 7]], [math.pi / 2, -math.pi / 2])
        # Even high gripper effort must preserve measured 80 mm, never force 0.
        np.testing.assert_allclose(sample.follower_position[[6, 13]], [0.08, 0.08])
        np.testing.assert_allclose(sample.leader_action[[6, 13]], [0.09, 0.09])
        assert sample.leader_action[0] != sample.follower_position[0]
        assert set(sample.endposes) == {"left", "right"}
    assert not recorder.is_connected
    assert [kwargs["can_name"] for kwargs in constructors] == ["can_left", "can_right"]
    assert all(not kwargs["start_sdk_joint_limit"] for kwargs in constructors)
    assert all(not kwargs["start_sdk_gripper_limit"] for kwargs in constructors)
    for client in clients.values():
        assert client.calls == [("ConnectPort", False, False), ("DisconnectPort",)]


@pytest.mark.parametrize(
    ("euler", "quaternion"),
    [
        ([0, 0, 0], [1, 0, 0, 0]),
        ([90_000, 0, 0], [math.sqrt(0.5), math.sqrt(0.5), 0, 0]),
        ([0, 90_000, 0], [math.sqrt(0.5), 0, math.sqrt(0.5), 0]),
        ([0, 0, 90_000], [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]),
        ([90_000, 90_000, 90_000], [math.sqrt(0.5), 0, math.sqrt(0.5), 0]),
    ],
)
def test_endpose_units_and_quaternion_order(euler, quaternion):
    client = FakePassiveClient("can_test")
    client.pose[3:] = euler
    pose = read_endpose(client)
    np.testing.assert_allclose(pose[:3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(pose[3:], quaternion, atol=1e-15)
    assert math.isclose(float(np.linalg.norm(pose[3:])), 1.0)
    assert client.calls == []


@pytest.mark.parametrize("kind", ["status", "joint", "gripper", "control", "gripper_control", "pose"])
@pytest.mark.parametrize("bad_timestamp", [0.0, float("nan"), float("inf"), -1.0, "stale", "future"])
def test_rejects_missing_or_stale_streams(passive_factory, kind, bad_timestamp):
    make, clients, _ = passive_factory
    recorder = make()
    recorder.connect()
    if bad_timestamp == "stale":
        bad_timestamp = time.time() - 2
    elif bad_timestamp == "future":
        bad_timestamp = time.time() + 2
    clients["can_right"].timestamps[kind] = bad_timestamp
    with pytest.raises(TimeoutError, match="Missing or stale"):
        recorder.read_sample()
    recorder.disconnect()


@pytest.mark.parametrize("kind", ["status", "joint", "gripper", "control", "gripper_control", "pose"])
@pytest.mark.parametrize("bad_hz", [0.0, float("nan"), -1.0])
def test_rejects_streams_without_positive_rate(passive_factory, kind, bad_hz):
    make, clients, _ = passive_factory
    recorder = make()
    recorder.connect()
    clients["can_left"].hz[kind] = bad_hz
    with pytest.raises(TimeoutError):
        recorder.read_sample()
    recorder.disconnect()


@pytest.mark.parametrize("field", ["joints", "controls", "pose"])
def test_rejects_nonfinite_values(passive_factory, field):
    make, clients, _ = passive_factory
    recorder = make()
    recorder.connect()
    getattr(clients["can_left"], field)[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        recorder.read_sample()
    recorder.disconnect()


@pytest.mark.parametrize("field", ["gripper", "target_gripper"])
@pytest.mark.parametrize("value", [-1, 100_001, float("nan")])
def test_rejects_invalid_gripper_without_clipping(passive_factory, field, value):
    make, clients, _ = passive_factory
    recorder = make()
    recorder.connect()
    setattr(clients["can_left"], field, value)
    with pytest.raises(ValueError):
        recorder.read_sample()
    recorder.disconnect()


def test_rejects_unhealthy_receiver_and_follower_fault(passive_factory):
    make, clients, _ = passive_factory
    with make() as recorder:
        clients["can_left"].ok = False
        with pytest.raises(ConnectionError, match="unhealthy"):
            recorder.read_sample()
        clients["can_left"].ok = True
        clients["can_left"].arm_status = 0x07
        with pytest.raises(RuntimeError, match="0x07"):
            recorder.read_sample()


def test_partial_connection_failure_closes_both_attempted_clients(passive_factory):
    make, clients, _ = passive_factory
    clients["can_right"].fail_connect = True
    recorder = make()
    with pytest.raises(OSError, match="connect failed"):
        recorder.connect()
    assert not recorder.is_connected
    assert all(client.calls[-1] == ("DisconnectPort",) for client in clients.values())


def test_missing_leader_stream_times_out_and_closes_clients(passive_factory):
    make, clients, _ = passive_factory
    clients["can_left"].timestamps["control"] = 0
    recorder = make()
    with pytest.raises(TimeoutError, match="leader commands and follower feedback"):
        recorder.connect()
    assert not recorder.is_connected
    assert all(client.calls[-1] == ("DisconnectPort",) for client in clients.values())


def test_existing_sdk_singleton_is_not_closed(passive_factory):
    make, clients, _ = passive_factory
    clients["can_right"].connected = True
    recorder = make()
    with pytest.raises(RuntimeError, match="already has a connected"):
        recorder.connect()
    assert clients["can_right"].calls == []
    assert clients["can_right"].connected
    assert not clients["can_left"].connected


def test_reconnect_and_idempotent_disconnect(passive_factory):
    make, clients, _ = passive_factory
    recorder = make()
    with pytest.raises(ConnectionError):
        recorder.read_sample()
    recorder.connect()
    with pytest.raises(RuntimeError, match="already connected"):
        recorder.connect()
    recorder.disconnect()
    recorder.disconnect()
    recorder.connect()
    recorder.disconnect()
    for client in clients.values():
        assert client.calls == [
            ("ConnectPort", False, False),
            ("DisconnectPort",),
            ("ConnectPort", True, False),
            ("DisconnectPort",),
        ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_can": ""},
        {"right_can": " "},
        {"left_can": "can_right"},
        {"feedback_timeout_s": 0},
        {"feedback_timeout_s": float("nan")},
        {"feedback_timeout_s": float("inf")},
        {"gripper_max_m": 0.101},
        {"gripper_max_m": -1},
        {"connect_timeout_s": -1},
        {"poll_interval_s": 0},
    ],
)
def test_invalid_configuration_is_rejected_without_sdk_access(kwargs):
    with pytest.raises(ValueError):
        SharedCANBiPiper(**kwargs)
