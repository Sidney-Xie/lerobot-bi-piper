from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from lerobot.robots.bi_piper.hdf5_recording import CameraSample, CollectionSample, HDF5EpisodeWriter

h5py = pytest.importorskip("h5py")


def _sample(index=0, *, cameras=True, source=True, action=True):
    position = np.arange(14, dtype=np.float64) / 100
    position[[6, 13]] = [0.025, 0.075]
    command = position.copy()
    command[0] += 0.1
    command[[6, 13]] = [0.05, 0.1]
    camera_samples = {}
    if cameras:
        calibration = {
            "serial": "12345",
            "intrinsic_cv": np.array([[500, 0, 12], [0, 510, 8], [0, 0, 1]], dtype=np.float64),
            "color_intrinsics": {
                "width": 24,
                "height": 16,
                "fx": 500.0,
                "fy": 510.0,
                "ppx": 12.0,
                "ppy": 8.0,
                "distortion_coefficients": np.zeros(5),
            },
            "depth_scale_m": 0.001,
            "depth_to_color_translation_m": np.zeros(3),
        }
        camera_samples["cam_high"] = CameraSample(
            rgb=np.full((16, 24, 3), [230, 50, 20], dtype=np.uint8),
            depth_mm=np.full((16, 24), 1200 + index, dtype=np.float64),
            timestamp_ns=1_000_000_000 + index * 40_000_000,
            frame_number=index + 10,
            device_timestamp_ms=100.0 + index * 40,
            calibration=calibration,
        )
    return CollectionSample(
        position=position,
        timestamp_ns=2_000_000_000 + index * 40_000_000,
        action=command if action else None,
        source_frame_index=index if source else None,
        endposes={"left": np.array([0.1, 0.2, 0.3, 1, 0, 0, 0], dtype=np.float64)},
        cameras=camera_samples,
    )


def _assert_no_artifacts(directory):
    assert list(directory.iterdir()) == []


def test_writes_robotwin_jpeg_depth_feedback_commands_and_calibration(tmp_path):
    path = tmp_path / "episode.hdf5"
    first = _sample(0)
    with HDF5EpisodeWriter(
        path, fps=50, expected_frames=2, metadata={"workflow": "two_pass", "capture_target_fps": 25}
    ) as writer:
        writer.append(first)
        writer.append(_sample(1))
        assert not path.exists()

    assert writer.finalize() == path
    assert list(tmp_path.iterdir()) == [path]
    with h5py.File(path, "r") as root:
        assert root.attrs["sim"] == np.False_
        assert root.attrs["fps"] == 50
        assert root.attrs["capture_target_fps"] == 25
        assert root.attrs["capture_fps"] == pytest.approx(25)
        assert root.attrs["workflow"] == "two_pass"
        assert root.attrs["joint_action_source"] == "measured_follower_feedback"
        assert root.attrs["gripper_encoding"] == "normalized_0_closed_1_open"
        assert root.attrs["endpose_quaternion_order"] == "wxyz"
        expected = first.position.copy()
        expected[[6, 13]] /= 0.1
        np.testing.assert_allclose(root["joint_action/vector"][0], expected)
        np.testing.assert_allclose(root["joint_action/left_arm"][0], first.position[:6])
        np.testing.assert_allclose(root["joint_action/right_arm"][0], first.position[7:13])
        assert root["joint_action/left_gripper"][0] == pytest.approx(0.25)
        assert root["joint_action/right_gripper"][0] == pytest.approx(0.75)
        assert root["observations/qpos"].id == root["joint_action/vector"].id
        np.testing.assert_allclose(root["action"][0, [0, 6, 13]], [first.action[0], 0.5, 1])
        np.testing.assert_array_equal(root["recording_metadata/action_available"], [True, True])
        np.testing.assert_array_equal(root["recording_metadata/timestamp_ns"], [2_000_000_000, 2_040_000_000])
        np.testing.assert_array_equal(root["recording_metadata/source_frame_index"], [0, 1])
        np.testing.assert_allclose(root["endpose/left_endpose"][0], first.endposes["left"])
        assert root["endpose/left_endpose"].attrs["available"]
        assert not root["endpose/right_endpose"].attrs["available"]
        assert np.isnan(root["endpose/right_endpose"][:]).all()
        np.testing.assert_array_equal(root["endpose/right_available"], [False, False])
        assert root["pointcloud"].shape == (2, 0)
        camera = root["observation/head_camera"]
        assert camera["rgb"].shape == (2,)
        assert camera["rgb"].dtype.kind == "S"
        payload = bytes(camera["rgb"][0])
        assert payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
        rgb = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        np.testing.assert_allclose(rgb, first.cameras["cam_high"].rgb, atol=3)
        assert camera["depth"].dtype == np.float64
        assert camera["depth"].compression == "lzf"
        assert camera["depth"].chunks == (1, 16, 24)
        assert camera["depth"][0, 0, 0] == 1200.0
        assert camera["depth"][1, 0, 0] == 1201.0
        assert camera.attrs["depth_scale_m"] == 0.001
        intrinsic = first.cameras["cam_high"].calibration["intrinsic_cv"]
        np.testing.assert_allclose(camera["intrinsic_cv"][0], intrinsic)
        assert not camera.attrs["world_extrinsic_available"]
        assert np.isnan(camera["extrinsic_cv"][:]).all()
        assert np.isnan(camera["cam2world_gl"][:]).all()
        calibration = root["camera_calibration/head_camera"]
        assert calibration.attrs["serial"] == "12345"
        assert calibration["color"].attrs["fx"] == 500.0
        np.testing.assert_array_equal(calibration["color/distortion_coefficients"], np.zeros(5))
        np.testing.assert_allclose(calibration["intrinsic_cv"], intrinsic)
        np.testing.assert_array_equal(root["recording_metadata/camera_frame_number/head_camera"], [10, 11])
        np.testing.assert_array_equal(
            root["recording_metadata/camera_capture_timestamp_ns/head_camera"], [1_000_000_000, 1_040_000_000]
        )
        np.testing.assert_array_equal(
            root["recording_metadata/camera_device_timestamp_ms/head_camera"], [100, 140]
        )


def test_teleoperation_missing_commands_and_source_are_explicit(tmp_path):
    path = tmp_path / "episode.hdf5"
    with HDF5EpisodeWriter(path, fps=30) as writer:
        writer.append(_sample(cameras=False, source=False, action=False))
    with h5py.File(path, "r") as root:
        assert np.isnan(root["action"][:]).all()
        np.testing.assert_array_equal(root["recording_metadata/action_available"], [False])
        np.testing.assert_array_equal(root["recording_metadata/source_frame_index"], [-1])
        assert not root.attrs["source_frame_index_available"]
        assert root.attrs["frame_alignment"] == "host_sampled"
        assert root.attrs["camera_stream_mode"] == "none"


@pytest.mark.parametrize("action_source", ["leader_can_control", "sent_replay_target"])
def test_preserves_collection_provenance_and_numeric_metadata(tmp_path, action_source):
    path = tmp_path / "episode.hdf5"
    initial_state = np.arange(14, dtype=np.float64) / 100
    with HDF5EpisodeWriter(
        path,
        fps=50,
        metadata={
            "sim": True,
            "backend": "offline",
            "action_source": action_source,
            "capture_target_fps": 25,
            "post_replay_initial_state": initial_state,
        },
    ) as writer:
        writer.append(_sample(cameras=False))
    with h5py.File(path, "r") as root:
        assert root.attrs["sim"]
        assert root.attrs["backend"] == "offline"
        assert root.attrs["action_source"] == action_source
        assert root.attrs["fps"] == 50
        assert root.attrs["capture_target_fps"] == 25
        assert root.attrs["capture_fps"] == 25
        np.testing.assert_array_equal(root.attrs["post_replay_initial_state"], initial_state)


def test_maps_wrist_camera_names_and_supports_rgb_only(tmp_path):
    sample = _sample()
    image = sample.cameras["cam_high"].rgb
    sample.cameras = {name: CameraSample(image) for name in ("cam_left_wrist", "cam_right_wrist")}
    path = tmp_path / "episode.hdf5"
    with HDF5EpisodeWriter(path, fps=25) as writer:
        writer.append(sample)
    with h5py.File(path, "r") as root:
        assert set(root["observation"]) == {"left_camera", "right_camera"}
        assert root.attrs["camera_stream_mode"] == "rgb"
        assert "depth" not in root["observation/left_camera"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"position": np.zeros(13)}, "position must have shape"),
        ({"position": np.full(14, np.nan)}, "NaN or infinity"),
        ({"action": np.full(14, np.inf)}, "NaN or infinity"),
        ({"position": np.full(14, 0.11)}, "gripper values must be meters"),
        ({"timestamp_ns": 1.5}, "must be an integer"),
        ({"source_frame_index": 1}, "exactly 0..N-1"),
        ({"endposes": {"left": np.ones(6)}}, "left endpose must have shape"),
        ({"endposes": {"left": np.full(7, np.nan)}}, "NaN or infinity"),
    ],
)
def test_rejects_invalid_sample_and_removes_temporary_file(tmp_path, change, message):
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25)
    with pytest.raises(ValueError, match=message):
        writer.append(replace(_sample(cameras=False), **change))
    _assert_no_artifacts(tmp_path)
    with pytest.raises(RuntimeError, match="aborted"):
        writer.finalize()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rgb": np.zeros((16, 24, 3), dtype=np.float32)}, "HWC uint8"),
        ({"depth_mm": np.zeros((15, 24))}, "depth_mm must have shape"),
        ({"depth_mm": np.full((16, 24), np.inf)}, "NaN or infinity"),
        ({"depth_mm": np.full((16, 24), -1)}, "nonnegative millimeters"),
        ({"frame_number": -1}, "must be in"),
        ({"device_timestamp_ms": np.nan}, "finite and nonnegative"),
        ({"calibration": {"scale": np.nan}}, "Out of range float values"),
    ],
)
def test_rejects_invalid_camera_and_removes_temporary_file(tmp_path, change, message):
    sample = _sample()
    sample.cameras["cam_high"] = replace(sample.cameras["cam_high"], **change)
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25)
    with pytest.raises(ValueError, match=message):
        writer.append(sample)
    _assert_no_artifacts(tmp_path)


@pytest.mark.parametrize("field", ["timestamp_ns", "frame_number", "device_timestamp_ms"])
def test_rejects_reused_camera_timestamps_or_frame_numbers(tmp_path, field):
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25)
    writer.append(_sample(0))
    second = _sample(1)
    setattr(second.cameras["cam_high"], field, getattr(_sample(0).cameras["cam_high"], field))
    with pytest.raises(ValueError, match="strictly increasing"):
        writer.append(second)
    _assert_no_artifacts(tmp_path)


def test_variable_jpeg_lengths_are_preserved_without_truncation(tmp_path):
    path = tmp_path / "episode.hdf5"
    second = _sample(1)
    second.cameras["cam_high"].rgb[:] = np.random.default_rng(12).integers(
        0, 256, second.cameras["cam_high"].rgb.shape, dtype=np.uint8
    )
    with HDF5EpisodeWriter(path, fps=25) as writer:
        writer.append(_sample(0))
        writer.append(second)
    with h5py.File(path, "r") as root:
        dataset = root["observation/head_camera/rgb"]
        assert len(dataset[0]) < len(dataset[1]) == dataset.dtype.itemsize
        for value in dataset:
            assert bytes(value).endswith(b"\xff\xd9")
            decoded = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (16, 24, 3)


def test_rejects_nonincreasing_robot_timestamp(tmp_path):
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25)
    writer.append(_sample(0, cameras=False))
    with pytest.raises(ValueError, match="strictly increasing"):
        writer.append(replace(_sample(1, cameras=False), timestamp_ns=_sample(0).timestamp_ns))
    _assert_no_artifacts(tmp_path)


@pytest.mark.parametrize(
    "change", ["camera_set", "shape", "depth", "frame_number", "calibration", "source_index"]
)
def test_rejects_changing_episode_configuration(tmp_path, change):
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25)
    writer.append(_sample(0))
    second = _sample(1)
    camera = second.cameras["cam_high"]
    if change == "camera_set":
        second.cameras = {}
    elif change == "shape":
        camera.rgb = np.zeros((32, 24, 3), dtype=np.uint8)
        camera.depth_mm = np.zeros((32, 24))
    elif change == "depth":
        camera.depth_mm = None
    elif change == "frame_number":
        camera.frame_number = None
    elif change == "calibration":
        camera.calibration["serial"] = "changed"
    else:
        second.source_frame_index = None
    with pytest.raises(ValueError):
        writer.append(second)
    _assert_no_artifacts(tmp_path)


def test_rejects_colliding_camera_aliases(tmp_path):
    sample = _sample()
    sample.cameras["head_camera"] = sample.cameras["cam_high"]
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25)
    with pytest.raises(ValueError, match="aliases collide"):
        writer.append(sample)
    _assert_no_artifacts(tmp_path)


@pytest.mark.parametrize("actual", [0, 1, 3])
def test_expected_frame_count_is_exact(tmp_path, actual):
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25, expected_frames=2)
    with pytest.raises(ValueError, match="empty|expected_frames"):
        for index in range(actual):
            writer.append(_sample(index, cameras=False))
        writer.finalize()
    _assert_no_artifacts(tmp_path)


def test_existing_output_is_rejected_without_modification(tmp_path):
    path = tmp_path / "episode.hdf5"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        HDF5EpisodeWriter(path, fps=25)
    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_publish_does_not_replace_output_created_during_recording(tmp_path):
    path = tmp_path / "episode.hdf5"
    writer = HDF5EpisodeWriter(path, fps=25)
    writer.append(_sample(cameras=False))
    path.write_bytes(b"winner")
    with pytest.raises(FileExistsError):
        writer.finalize()
    assert path.read_bytes() == b"winner"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_publish_failure_removes_temporary_file(tmp_path, monkeypatch):
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25)
    writer.append(_sample())

    def fail_link(*args):
        raise PermissionError("simulated link failure")

    monkeypatch.setattr("lerobot.robots.bi_piper.hdf5_recording.os.link", fail_link)
    with pytest.raises(PermissionError, match="simulated link failure"):
        writer.finalize()
    _assert_no_artifacts(tmp_path)


def test_jpeg_failure_aborts_episode(tmp_path, monkeypatch):
    writer = HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25)
    monkeypatch.setattr("lerobot.robots.bi_piper.hdf5_recording.cv2.imencode", lambda *_: (False, None))
    with pytest.raises(RuntimeError, match="JPEG-encode"):
        writer.append(_sample())
    _assert_no_artifacts(tmp_path)


def test_keyboard_interrupt_aborts_context_manager(tmp_path):
    with pytest.raises(KeyboardInterrupt), HDF5EpisodeWriter(tmp_path / "episode.hdf5", fps=25) as writer:
        writer.append(_sample())
        raise KeyboardInterrupt
    _assert_no_artifacts(tmp_path)


def test_abort_is_idempotent_and_retains_finalized_output(tmp_path):
    path = tmp_path / "episode.hdf5"
    writer = HDF5EpisodeWriter(path, fps=25)
    writer.append(_sample(cameras=False))
    writer.finalize()
    writer.abort()
    writer.abort()
    assert path.is_file()
    with pytest.raises(RuntimeError, match="finalized"):
        writer.append(_sample())
