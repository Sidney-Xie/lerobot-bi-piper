"""Exercise complete collection state machines with fake clocks and hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.bi_piper import collection
from lerobot.robots.bi_piper.collection import (
    BiPiperCollectionSession,
    collect_shared_can_episode,
    run_two_pass_episode,
)
from lerobot.robots.bi_piper.hdf5_recording import CameraSample
from lerobot.robots.bi_piper.replay_collection import ReplaySource, build_replay_plan

h5py = pytest.importorskip("h5py")


class FakeClock:
    def __init__(self):
        self.now = 1.0

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return round(self.now * 1e9)

    def sleep(self, seconds):
        self.now += seconds


class FakeSession:
    def __init__(self):
        self.position = np.array([0, 0.05, -0.05, 0, 0, 0, 0.0995] * 2, dtype=float)
        self.events = []
        self.stop = Mock()

    def move_to(self, target):
        self.events.append(("move", target.copy()))
        self.position = target.copy()

    def send_position(self, target):
        self.events.append(("send", target.copy()))
        self.position = target.copy()
        return target.copy()

    def read_position(self):
        result = self.position.copy()
        result[0] += 0.001  # A real observation is different from the target.
        return result

    def read_endposes(self):
        return {side: np.array([0.1, 0.2, 0.3, 1, 0, 0, 0]) for side in ("left", "right")}


class FakeCameras:
    def __init__(self, clock):
        self.clock = clock
        self.count = 0

    def read(self):
        self.count += 1
        return {
            "cam_high": CameraSample(
                rgb=np.full((4, 6, 3), self.count, dtype=np.uint8),
                timestamp_ns=self.clock.monotonic_ns(),
                frame_number=self.count,
            )
        }


@pytest.fixture
def replay_inputs(tmp_path):
    position = np.array([0, 0.05, -0.05, 0, 0, 0, 0.05] * 2, dtype=float)
    positions = np.stack([position.copy() for _ in range(4)])
    positions[:, 0] = np.arange(4) * 0.01
    source = ReplaySource(tmp_path / "source.hdf5", positions, 25)
    first = build_replay_plan(source, no_grasp=True)
    second = build_replay_plan(source)
    return source, first, second


def test_two_pass_records_only_second_pass_and_resets_both_arms(tmp_path, replay_inputs):
    source, first, second = replay_inputs
    session, clock = FakeSession(), FakeClock()
    cameras = FakeCameras(clock)
    initial = session.position.copy()
    output = tmp_path / "episode.hdf5"

    def prompt(_):
        assert not output.exists()
        assert cameras.count == 0
        assert not list(tmp_path.glob(".*.tmp"))
        np.testing.assert_array_equal(session.position, initial)

    assert (
        run_two_pass_episode(
            session,
            source,
            first,
            second,
            output,
            initial_position=initial,
            prompt=prompt,
            cameras=cameras,
            clock=clock,
        )
        == output
    )
    moves = [target for kind, target in session.events if kind == "move"]
    assert len(moves) == 4
    np.testing.assert_array_equal(moves[1], initial)
    np.testing.assert_array_equal(moves[3], initial)
    assert len([event for event in session.events if event[0] == "send"]) == len(first.positions) + len(
        second.positions
    )
    assert cameras.count == len(source.positions)
    session.stop.assert_not_called()
    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle["recording_metadata/source_frame_index"][:], np.arange(4))
        np.testing.assert_allclose(handle["action"][:, 0], source.positions[:, 0])
        np.testing.assert_allclose(handle["joint_action/vector"][:, 0], source.positions[:, 0] + 0.001)
        np.testing.assert_allclose(handle["action"][:, 6], 0.5)
        assert handle["observation/head_camera/rgb"].shape == (4,)
        assert not handle.attrs["first_pass_recorded"]
        assert handle.attrs["fps"] == 25
        assert handle.attrs["capture_target_fps"] == 12.5
        assert handle.attrs["action_source"] == "sent_replay_target"


def test_camera_failure_aborts_episode_and_stops_without_post_failure_motion(tmp_path, replay_inputs):
    session = FakeSession()
    cameras = SimpleNamespace(read=Mock(side_effect=TimeoutError("stale camera")))
    output = tmp_path / "bad.hdf5"
    with pytest.raises(TimeoutError, match="stale camera"):
        run_two_pass_episode(
            session,
            *replay_inputs,
            output,
            initial_position=session.position.copy(),
            cameras=cameras,
            prompt=lambda _: None,
            clock=FakeClock(),
        )
    session.stop.assert_called_once()
    assert len([event for event in session.events if event[0] == "move"]) == 3
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_return_does_not_publish_the_recording(tmp_path, replay_inputs):
    session = FakeSession()
    original_move = session.move_to
    calls = 0

    def fail_final_return(target):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise TimeoutError("return did not settle")
        original_move(target)

    session.move_to = fail_final_return
    with pytest.raises(TimeoutError, match="return did not settle"):
        run_two_pass_episode(
            session,
            *replay_inputs,
            tmp_path / "bad.hdf5",
            initial_position=session.position.copy(),
            prompt=lambda _: None,
            clock=FakeClock(),
        )
    session.stop.assert_called_once()
    assert not list(tmp_path.glob("*.hdf5"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_cancel_at_scene_reset_has_no_recording(tmp_path, replay_inputs):
    session = FakeSession()
    with pytest.raises(KeyboardInterrupt):
        run_two_pass_episode(
            session,
            *replay_inputs,
            tmp_path / "bad.hdf5",
            initial_position=session.position.copy(),
            prompt=Mock(side_effect=KeyboardInterrupt),
            clock=FakeClock(),
        )
    session.stop.assert_called_once()
    assert not list(tmp_path.glob("*.hdf5"))


def test_existing_output_is_rejected_before_motion(tmp_path, replay_inputs):
    session = FakeSession()
    output = tmp_path / "existing.hdf5"
    output.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        run_two_pass_episode(
            session,
            *replay_inputs,
            output,
            initial_position=session.position.copy(),
            prompt=lambda _: None,
            clock=FakeClock(),
        )
    assert session.events == []
    assert output.read_bytes() == b"original"


def test_late_control_aborts_instead_of_bursting_or_skipping_source_frames(tmp_path, replay_inputs):
    session, clock = FakeSession(), FakeClock()
    original_send = session.send_position

    def slow_send(target):
        sent = original_send(target)
        clock.sleep(1.0)
        return sent

    session.send_position = slow_send
    with pytest.raises(TimeoutError, match="deadline"):
        run_two_pass_episode(
            session,
            *replay_inputs,
            tmp_path / "late.hdf5",
            initial_position=session.position.copy(),
            prompt=lambda _: None,
            max_lateness_s=0.1,
            clock=clock,
        )
    assert len([event for event in session.events if event[0] == "send"]) == 1
    session.stop.assert_called_once()
    assert not list(tmp_path.glob("*.hdf5"))


def test_episode_failure_still_stops_when_writer_abort_raises(tmp_path, replay_inputs, monkeypatch):
    session = FakeSession()
    fake_writer = SimpleNamespace(
        append=Mock(side_effect=ValueError("sample failed")),
        abort=Mock(side_effect=OSError("disk cleanup failed")),
    )
    monkeypatch.setattr(collection, "HDF5EpisodeWriter", lambda *args, **kwargs: fake_writer)
    with pytest.raises(ValueError, match="sample failed"):
        run_two_pass_episode(
            session,
            *replay_inputs,
            tmp_path / "bad.hdf5",
            initial_position=session.position.copy(),
            prompt=lambda _: None,
            clock=FakeClock(),
        )
    session.stop.assert_called_once()


def test_passive_teleop_records_actual_leader_command_and_follower_feedback(tmp_path):
    follower = np.array([0, 0.05, -0.05, 0, 0, 0, 0.03] * 2)
    action = follower.copy()
    action[[0, 6, 13]] = (0.1, 0.04, 0.04)
    sample = SimpleNamespace(follower_position=follower, leader_action=action, endposes={})
    session = SimpleNamespace(read_sample=Mock(return_value=sample))
    result = collect_shared_can_episode(
        session, tmp_path / "teleop.hdf5", duration_s=0.12, fps=25, clock=FakeClock()
    )
    assert session.read_sample.call_count == 3
    with h5py.File(result) as handle:
        assert handle["action"].shape == (3, 14)
        np.testing.assert_allclose(handle["action"][:, 6], 0.4)
        np.testing.assert_allclose(handle["joint_action/vector"][:, 6], 0.3)
        np.testing.assert_array_equal(handle["recording_metadata/source_frame_index"][:], [-1] * 3)
        assert handle.attrs["following"] == "firmware_direct_1_to_1"
        assert handle.attrs["action_source"] == "leader_can_control"


def test_passive_teleop_failure_aborts_without_sending_stop(tmp_path):
    session = SimpleNamespace(read_sample=Mock(side_effect=TimeoutError("stale leader")), stop=Mock())
    with pytest.raises(TimeoutError, match="stale leader"):
        collect_shared_can_episode(session, tmp_path / "bad.hdf5", duration_s=0.1, clock=FakeClock())
    session.stop.assert_not_called()
    assert not list(tmp_path.glob("*.hdf5"))
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("option", [{"fps": float("nan")}, {"duration_s": 0}, {"max_lateness_s": -1}])
def test_passive_collection_validates_timing_before_opening_writer(tmp_path, option):
    with pytest.raises(ValueError):
        collect_shared_can_episode(None, tmp_path / "never.hdf5", **option)
    assert not list(tmp_path.iterdir())


def test_physical_collection_feedback_preserves_loaded_gripper():
    session = object.__new__(BiPiperCollectionSession)
    reader = Mock(
        return_value=dict(
            zip(("j1", "j2", "j3", "j4", "j5", "j6", "gripper"), [0, 0.05, -0.05, 0, 0, 0, 0.04], strict=True)
        )
    )
    session.robot = SimpleNamespace(_read_arm_positions=reader)
    position = session.read_position()
    np.testing.assert_allclose(position[[6, 13]], [0.04, 0.04])
    assert [call.kwargs for call in reader.call_args_list] == [{"for_policy": False}] * 2
