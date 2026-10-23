from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.bi_piper.hdf5_replay import (
    VECTOR_ORDER,
    load_hdf5_trajectory,
    replay_hdf5_trajectory,
)

h5py = pytest.importorskip("h5py")


def _write_trajectory(path, *, vector=None, split=False, **attrs):
    attrs.setdefault("vector_order", VECTOR_ORDER)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("joint_action")
        if split:
            values = np.asarray(vector)
            group.create_dataset("left_arm", data=values[:, :6])
            group.create_dataset("left_gripper", data=values[:, 6])
            group.create_dataset("right_arm", data=values[:, 7:13])
            group.create_dataset("right_gripper", data=values[:, 13])
        else:
            group.create_dataset("vector", data=vector)
        for key, value in attrs.items():
            if value is not None:
                handle.attrs[key] = value


def _joint_limits():
    return {
        "j1": (-2.6179, 2.6179),
        "j2": (0.0, 3.14),
        "j3": (-2.967, 0.0),
        "j4": (-1.745, 1.745),
        "j5": (-1.22, 1.22),
        "j6": (-2.09439, 2.09439),
    }


def _arm_config(*, gripper_unit="normalized", gripper_range_m=(0.0, 0.1), joint_limits=None):
    return SimpleNamespace(
        gripper_unit=gripper_unit,
        gripper_range_m=gripper_range_m,
        joint_limits=_joint_limits() if joint_limits is None else joint_limits,
        max_relative_target=None,
    )


class FakeRobot:
    def __init__(self, *, left_config=None, right_config=None, send_transform=None):
        self.actions = []
        self._arm_configs = {
            "left": left_config or _arm_config(),
            "right": right_config or _arm_config(),
        }
        self._send_transform = send_transform or (lambda action: action)

    def send_action(self, action):
        self.actions.append(action)
        return self._send_transform(dict(action))


def test_load_vector_and_convert_normalized_gripper_to_meters(tmp_path):
    vector = np.zeros((2, 14), dtype=np.float64)
    vector[:, 6] = 0.5
    vector[:, 13] = 1.0
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector, gripper_encoding="normalized_0_closed_1_open")

    trajectory = load_hdf5_trajectory(path)
    action = trajectory.action(0, gripper_unit="meters", gripper_range_m=(0.0, 0.07))

    assert trajectory.num_frames == 2
    assert action["left_j1.pos"] == 0.0
    assert action["left_gripper.pos"] == pytest.approx(0.035)
    assert action["right_gripper.pos"] == pytest.approx(0.07)


def test_load_split_action_datasets(tmp_path):
    vector = np.zeros((2, 14), dtype=np.float64)
    vector[:, 1] = (0.1, 0.2)
    vector[:, 2] = (-0.1, -0.2)
    vector[:, 6] = (0.25, 0.5)
    vector[:, 13] = (0.75, 1.0)
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector, split=True, vector_order=None)

    trajectory = load_hdf5_trajectory(path)

    np.testing.assert_array_equal(trajectory.actions, vector)


def test_replay_uses_absolute_control_deadlines(tmp_path, monkeypatch):
    vector = np.zeros((3, 14), dtype=np.float64)
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector)
    trajectory = load_hdf5_trajectory(path)

    monkeypatch.setattr("lerobot.robots.bi_piper.hdf5_replay.precise_sleep", lambda _: None)
    robot = FakeRobot()

    assert replay_hdf5_trajectory(robot, trajectory, control_hz=25.0) == 3
    assert len(robot.actions) == 3
    assert set(robot.actions[0]) == {
        f"{side}_{motor}.pos"
        for side in ("left", "right")
        for motor in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
    }


def test_rejects_nonfinite_trajectory(tmp_path):
    vector = np.zeros((1, 14), dtype=np.float64)
    vector[0, 0] = np.nan
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector)

    with pytest.raises(ValueError, match="NaN"):
        load_hdf5_trajectory(path)


def test_rejects_non_radian_trajectory(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=np.zeros((1, 14)), joint_action_units="degrees")

    with pytest.raises(ValueError, match="radians"):
        load_hdf5_trajectory(path)


def test_vector_dataset_requires_order_metadata(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=np.zeros((1, 14)), vector_order=None)

    with pytest.raises(ValueError, match="no vector_order metadata"):
        load_hdf5_trajectory(path)


def test_rejects_unexpected_vector_order(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_trajectory(
        path,
        vector=np.zeros((1, 14)),
        vector_order="right_j1..j6,right_gripper,left_j1..j6,left_gripper",
    )

    with pytest.raises(ValueError, match="Unsupported vector_order"):
        load_hdf5_trajectory(path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (0, 2.7, "left_j1"),
        (9, 0.01, "right_j3"),
        (6, -0.01, "left_gripper"),
        (13, 1.01, "right_gripper"),
    ],
)
def test_rejects_targets_outside_physical_ranges(tmp_path, column, value, message):
    vector = np.zeros((1, 14), dtype=np.float64)
    vector[0, column] = value
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector)

    with pytest.raises(ValueError, match=message):
        load_hdf5_trajectory(path)


def test_rejects_meter_gripper_outside_physical_range(tmp_path):
    vector = np.zeros((1, 14), dtype=np.float64)
    vector[0, 6] = 0.101
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector, gripper_encoding="meters")

    with pytest.raises(ValueError, match="left_gripper"):
        load_hdf5_trajectory(path)


def test_replay_rejects_gripper_unit_mismatch_before_sending(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=np.zeros((1, 14)))
    trajectory = load_hdf5_trajectory(path)
    robot = FakeRobot(
        left_config=_arm_config(gripper_unit="meters"),
        right_config=_arm_config(gripper_unit="meters"),
    )

    with pytest.raises(ValueError, match="does not match"):
        replay_hdf5_trajectory(robot, trajectory, control_hz=25.0, gripper_unit="normalized")
    assert robot.actions == []


def test_replay_rejects_inconsistent_robot_gripper_ranges(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=np.zeros((1, 14)))
    trajectory = load_hdf5_trajectory(path)
    robot = FakeRobot(right_config=_arm_config(gripper_range_m=(0.0, 0.07)))

    with pytest.raises(ValueError, match="same gripper_range_m"):
        replay_hdf5_trajectory(robot, trajectory, control_hz=25.0)
    assert robot.actions == []


def test_replay_rejects_explicit_gripper_range_mismatch(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=np.zeros((1, 14)))
    trajectory = load_hdf5_trajectory(path)
    robot = FakeRobot()

    with pytest.raises(ValueError, match="does not match"):
        replay_hdf5_trajectory(
            robot,
            trajectory,
            control_hz=25.0,
            gripper_range_m=(0.0, 0.07),
        )
    assert robot.actions == []


def test_replay_rejects_meter_gripper_outside_configured_range_before_sending(tmp_path):
    vector = np.zeros((1, 14), dtype=np.float64)
    vector[0, (6, 13)] = 0.08
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector, gripper_encoding="meters")
    trajectory = load_hdf5_trajectory(path)
    robot = FakeRobot(
        left_config=_arm_config(gripper_unit="meters", gripper_range_m=(0.0, 0.07)),
        right_config=_arm_config(gripper_unit="meters", gripper_range_m=(0.0, 0.07)),
    )

    with pytest.raises(ValueError, match="configured gripper_range_m"):
        replay_hdf5_trajectory(robot, trajectory, control_hz=25.0, gripper_unit="meters")
    assert robot.actions == []


def test_replay_rejects_target_outside_configured_joint_limits_before_sending(tmp_path):
    vector = np.zeros((1, 14), dtype=np.float64)
    vector[0, 0] = 0.2
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector)
    trajectory = load_hdf5_trajectory(path)
    narrow_limits = _joint_limits()
    narrow_limits["j1"] = (-0.1, 0.1)
    robot = FakeRobot(left_config=_arm_config(joint_limits=narrow_limits))

    with pytest.raises(ValueError, match="left_j1"):
        replay_hdf5_trajectory(robot, trajectory, control_hz=25.0)
    assert robot.actions == []


def test_replay_rejects_relative_target_clamp_before_sending(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=np.zeros((1, 14), dtype=np.float64))
    trajectory = load_hdf5_trajectory(path)
    robot = FakeRobot()
    robot._arm_configs["right"].max_relative_target = dict.fromkeys(
        ("j1", "j2", "j3", "j4", "j5", "j6", "gripper"), 0.1
    )

    with pytest.raises(ValueError, match="absolute target mode"):
        replay_hdf5_trajectory(robot, trajectory, control_hz=25.0)
    assert robot.actions == []


def test_replay_stops_when_robot_clips_action(tmp_path, monkeypatch):
    vector = np.zeros((2, 14), dtype=np.float64)
    vector[:, 1] = 0.1
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=vector)
    trajectory = load_hdf5_trajectory(path)

    def clip_left_j2(action):
        action["left_j2.pos"] = 0.05
        return action

    robot = FakeRobot(send_transform=clip_left_j2)
    monkeypatch.setattr("lerobot.robots.bi_piper.hdf5_replay.precise_sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="clipped or altered replay frame 0 left_j2.pos"):
        replay_hdf5_trajectory(robot, trajectory, control_hz=25.0)
    assert len(robot.actions) == 1


def test_replay_detects_robot_mutating_requested_action(tmp_path, monkeypatch):
    path = tmp_path / "episode.hdf5"
    _write_trajectory(path, vector=np.zeros((1, 14), dtype=np.float64))
    trajectory = load_hdf5_trajectory(path)
    robot = FakeRobot()

    def mutate_in_place(action):
        robot.actions.append(action)
        action["left_j2.pos"] = 0.05
        return action

    robot.send_action = mutate_in_place
    monkeypatch.setattr("lerobot.robots.bi_piper.hdf5_replay.precise_sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="clipped or altered replay frame 0 left_j2.pos"):
        replay_hdf5_trajectory(robot, trajectory, control_hz=25.0)


def test_real_execution_requires_fixed_confirmation_phrase():
    from examples.bi_piper.replay_hdf5 import EXECUTION_CONFIRMATION, parse_args

    with pytest.raises(SystemExit):
        parse_args(["--execute"])
    with pytest.raises(SystemExit):
        parse_args(["--execute", "--confirm", "yes"])

    args = parse_args(["--execute", "--confirm", EXECUTION_CONFIRMATION])
    assert args.execute is True
    assert args.confirm == EXECUTION_CONFIRMATION


@pytest.mark.parametrize("option", ["--max-joint-step-rad=nan", "--max-gripper-step=inf"])
def test_replay_cli_rejects_non_finite_step_limits(option):
    from examples.bi_piper.replay_hdf5 import parse_args

    with pytest.raises(SystemExit):
        parse_args([option])


def test_replay_cli_keeps_torque_by_default():
    from examples.bi_piper.replay_hdf5 import parse_args

    assert parse_args([]).disable_on_disconnect is False
    assert parse_args(["--disable-on-disconnect"]).disable_on_disconnect is True


@pytest.mark.parametrize(
    "argument",
    ["--max-joint-step-rad=nan", "--max-joint-step-rad=inf", "--max-gripper-step=nan"],
)
def test_replay_cli_rejects_nonfinite_action_step_limits(argument):
    from examples.bi_piper.replay_hdf5 import parse_args

    with pytest.raises(SystemExit):
        parse_args([argument])
