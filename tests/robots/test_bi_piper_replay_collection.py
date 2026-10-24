from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.bi_piper.replay_collection import (
    VECTOR_ORDER,
    ReplaySource,
    build_replay_plan,
    load_replay_source,
)

h5py = pytest.importorskip("h5py")


def _positions(frames=4):
    values = np.zeros((frames, 14), dtype=np.float64)
    values[:, 0] = np.linspace(0, 0.09, frames)
    values[:, 1] = 0.05
    values[:, 2] = -0.05
    values[:, 6] = np.linspace(0.02, 0.05, frames)
    values[:, 7] = np.linspace(0, -0.09, frames)
    values[:, 8] = 0.05
    values[:, 9] = -0.05
    values[:, 13] = np.linspace(0.05, 0.02, frames)
    return values


def _write(path, *, values=None, layout="vector", **attrs):
    values = _positions() if values is None else np.asarray(values)
    with h5py.File(path, "w") as handle:
        if layout == "vector":
            handle.create_dataset("joint_action/vector", data=values)
        elif layout == "split":
            for name, data in (
                ("left_arm", values[:, :6]),
                ("left_gripper", values[:, 6]),
                ("right_arm", values[:, 7:13]),
                ("right_gripper", values[:, 13:14]),
            ):
                handle.create_dataset(f"joint_action/{name}", data=data)
        else:
            for side, offset in (("left", 0), ("right", 7)):
                handle.create_dataset(f"action/{side}_arm_joint_states", data=values[:, offset : offset + 6])
                handle.create_dataset(
                    f"action/{side}_ee_joint_states", data=values[:, offset + 6 : offset + 7]
                )
        for name, value in attrs.items():
            handle.attrs[name] = value


@pytest.mark.parametrize("layout", ["vector", "split", "native"])
def test_load_supported_layouts_and_normalized_grippers(tmp_path, layout):
    path = tmp_path / "episode.hdf5"
    values = _positions()
    normalized = values.copy()
    normalized[:, (6, 13)] /= 0.1
    _write(path, values=normalized, layout=layout, fps=25)

    source = load_replay_source(path)

    np.testing.assert_allclose(source.positions, values)
    assert source.source_hz == 25
    assert source.metadata["vector_order_assumed"] is True
    assert source.metadata["vector_order"] == VECTOR_ORDER
    assert source.metadata["gripper_encoding"] == "normalized_0_closed_1_open"
    assert source.metadata["joint_units"] == "radians"
    assert source.metadata["source_hz_origin"] == "fps"


def test_load_meter_grippers_preserves_values_and_order_metadata(tmp_path):
    path = tmp_path / "meters.hdf5"
    values = _positions()
    _write(path, values=values, gripper_encoding="meters", vector_order=VECTOR_ORDER)

    source = load_replay_source(path, gripper_max_m=0.07)

    np.testing.assert_array_equal(source.positions, values)
    assert source.metadata["vector_order_assumed"] is False
    assert source.metadata["gripper_max_m"] == 0.07
    assert source.source_hz == 25
    assert source.metadata["source_hz_origin"] == "fallback"


@pytest.mark.parametrize("frequency_field", ["fps", "control_hz", "additional_info/frequency"])
def test_source_frequency_and_explicit_override(tmp_path, frequency_field):
    path = tmp_path / "frequency.hdf5"
    _write(path)
    with h5py.File(path, "a") as handle:
        if "/" in frequency_field:
            handle.create_dataset(frequency_field, data=20)
        else:
            handle.attrs[frequency_field] = 20
    source = load_replay_source(path)
    assert source.source_hz == 20
    assert source.metadata["source_hz_origin"] == frequency_field
    override = load_replay_source(path, source_hz=15)
    assert override.source_hz == 15
    assert override.metadata["source_hz_origin"] == "override"


def test_frequency_metadata_precedence(tmp_path):
    path = tmp_path / "frequency.hdf5"
    _write(path, fps=30, control_hz=20)
    with h5py.File(path, "a") as handle:
        handle.create_dataset("additional_info/frequency", data=10)
    assert load_replay_source(path).source_hz == 30


@pytest.mark.parametrize("scope", ["root", "dataset"])
def test_incompatible_declared_vector_order_is_rejected(tmp_path, scope):
    path = tmp_path / "order.hdf5"
    _write(path)
    with h5py.File(path, "a") as handle:
        target = handle if scope == "root" else handle["joint_action/vector"]
        target.attrs["vector_order"] = "right_j1..j6,right_gripper,left_j1..j6,left_gripper"
    with pytest.raises(ValueError, match="vector_order"):
        load_replay_source(path)


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({"joint_units": "degrees"}, "radians"),
        ({"gripper_encoding": "millimeters"}, "gripper encoding"),
        ({"fps": 0}, "source_hz"),
        ({"fps": float("nan")}, "source_hz"),
        ({"fps": [25, 30]}, "scalar"),
    ],
)
def test_invalid_units_and_frequency_are_rejected(tmp_path, attrs, message):
    path = tmp_path / "invalid.hdf5"
    _write(path, **attrs)
    with pytest.raises(ValueError, match=message):
        load_replay_source(path)


@pytest.mark.parametrize("shape", [(0, 14), (4, 13), (14,), (2, 7, 2)])
def test_invalid_vector_shape_is_rejected(tmp_path, shape):
    path = tmp_path / "shape.hdf5"
    _write(path, values=np.zeros(shape))
    with pytest.raises(ValueError, match="shape"):
        load_replay_source(path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [(0, np.nan, "NaN"), (1, -0.01, "left_j2"), (9, 0.01, "right_j3"), (6, 1.01, "Normalized")],
)
def test_invalid_targets_are_rejected_without_clipping(tmp_path, column, value, message):
    path = tmp_path / "targets.hdf5"
    values = _positions()
    values[0, column] = value
    _write(path, values=values)
    with pytest.raises(ValueError, match=message):
        load_replay_source(path)


def test_meter_gripper_must_fit_selected_hardware_range(tmp_path):
    path = tmp_path / "gripper.hdf5"
    values = _positions()
    values[0, 6] = 0.08
    _write(path, values=values, gripper_encoding="meters")
    with pytest.raises(ValueError, match="left_gripper"):
        load_replay_source(path, gripper_max_m=0.07)


def test_split_datasets_require_matching_shapes(tmp_path):
    path = tmp_path / "split.hdf5"
    _write(path, layout="split")
    with h5py.File(path, "a") as handle:
        del handle["joint_action/right_arm"]
        handle.create_dataset("joint_action/right_arm", data=np.zeros((3, 6)))
    with pytest.raises(ValueError, match="same number of frames"):
        load_replay_source(path)


def test_half_speed_plan_preserves_source_frames_and_endpoints():
    source = ReplaySource(Path("source.hdf5"), _positions(10), 25)
    plan = build_replay_plan(source)

    assert plan.duration_s == pytest.approx(9 / (25 * 0.5))
    assert plan.command_times_s[0] == 0
    assert plan.command_times_s[-1] == plan.duration_s
    assert np.all(np.diff(plan.command_times_s) > 0)
    assert np.all(np.diff(plan.source_frame_command_indices) > 0)
    assert plan.source_frame_count == 10
    assert plan.frame_count == len(plan.command_times_s)
    assert plan.source_frame_command_indices[0] == 0
    assert plan.source_frame_command_indices[-1] == plan.frame_count - 1
    np.testing.assert_array_equal(plan.positions[plan.source_frame_command_indices], source.positions)


def test_no_grasp_changes_only_grippers_and_keeps_source_unchanged():
    source = ReplaySource(Path("source.hdf5"), _positions(), 25)
    original = source.positions.copy()
    normal = build_replay_plan(source)
    opened = build_replay_plan(source, no_grasp=True, gripper_max_m=0.07)
    joint_columns = [*range(6), *range(7, 13)]

    np.testing.assert_array_equal(opened.positions[:, joint_columns], normal.positions[:, joint_columns])
    np.testing.assert_array_equal(opened.positions[:, (6, 13)], 0.07)
    np.testing.assert_array_equal(source.positions, original)


def test_plan_rejects_control_rate_below_source_capture_rate():
    source = ReplaySource(Path("source.hdf5"), _positions(), 25)
    with pytest.raises(ValueError, match="one-to-one"):
        build_replay_plan(source, control_hz=10, speed_scale=0.5)


@pytest.mark.parametrize(("column", "value", "message"), [(0, 0.5, "joint step"), (6, 0.1, "gripper step")])
def test_plan_rejects_large_steps_without_stretching_time(column, value, message):
    positions = np.zeros((2, 14))
    positions[1, column] = value
    source = ReplaySource(Path("source.hdf5"), positions, 25)
    with pytest.raises(ValueError, match=message):
        build_replay_plan(source, control_hz=25, speed_scale=1)


def test_single_frame_source_and_plan(tmp_path):
    path = tmp_path / "single.hdf5"
    _write(path, values=_positions(1), gripper_encoding="meters")
    source = load_replay_source(path)
    plan = build_replay_plan(source)
    assert plan.duration_s == 0
    assert plan.source_frame_count == plan.frame_count == 1
    np.testing.assert_array_equal(plan.source_frame_command_indices, [0])
    np.testing.assert_array_equal(plan.positions, source.positions)


@pytest.mark.parametrize(
    "name", ["control_hz", "speed_scale", "max_joint_step_rad", "max_gripper_step_m", "gripper_max_m"]
)
def test_plan_rejects_nonpositive_and_nonfinite_parameters(name):
    source = ReplaySource(Path("source.hdf5"), _positions(), 25)
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=name):
            build_replay_plan(source, **{name: value})


def test_missing_or_unsupported_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_replay_source(tmp_path / "missing.hdf5")
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError, match="no supported"):
        load_replay_source(path)
