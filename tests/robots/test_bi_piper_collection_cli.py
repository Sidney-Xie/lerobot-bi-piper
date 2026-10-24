from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.scripts import lerobot_collect_bi_piper as cli

h5py = pytest.importorskip("h5py")


def _args(mode, output_dir, *extra):
    source = ["source.hdf5"] if mode == "replay" else []
    return [mode, *source, "--output-dir", str(output_dir), *extra]


def _write_source(path, *, fps=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((3, 14), dtype=np.float64)
    values[:, 0] = [0, 0.01, 0.02]
    values[:, 1] = values[:, 8] = 0.05
    values[:, 2] = values[:, 9] = -0.05
    values[:, 6] = [0.2, 0.3, 0.4]
    values[:, 13] = [0.4, 0.3, 0.2]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("joint_action/vector", data=values)
        handle.attrs["fps"] = fps
    return values


def _forbid_hardware(monkeypatch):
    forbidden = Mock(
        side_effect=AssertionError("The preview/offline path accessed hardware or requested input")
    )
    for name in ("_preflight", "_connect_cameras", "BiPiperCollectionSession", "collect_shared_can_episode"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr("lerobot.robots.bi_piper.shared_can.SharedCANBiPiper", forbidden)
    return forbidden


@pytest.mark.parametrize("mode", ["replay", "teleop"])
def test_execute_requires_the_exact_mode_confirmation(tmp_path, mode):
    confirmation = cli.REPLAY_CONFIRMATION if mode == "replay" else cli.TELEOP_CONFIRMATION
    with pytest.raises(SystemExit) as missing:
        cli.parse_args(_args(mode, tmp_path, "--execute", "--no-cameras"))
    assert missing.value.code == 2
    with pytest.raises(SystemExit):
        cli.parse_args(_args(mode, tmp_path, "--execute", "--no-cameras", "--confirm", "yes"))
    args = cli.parse_args(_args(mode, tmp_path, "--execute", "--no-cameras", "--confirm", confirmation))
    assert args.execute


@pytest.mark.parametrize("mode", ["replay", "teleop"])
def test_confirmation_cannot_enable_hardware_on_its_own(tmp_path, mode):
    confirmation = cli.REPLAY_CONFIRMATION if mode == "replay" else cli.TELEOP_CONFIRMATION
    with pytest.raises(SystemExit):
        cli.parse_args(_args(mode, tmp_path, "--confirm", confirmation))


@pytest.mark.parametrize(
    "switches", [("--execute", "--dry-run"), ("--execute", "--offline"), ("--dry-run", "--offline")]
)
def test_replay_execution_modes_are_mutually_exclusive(tmp_path, switches):
    with pytest.raises(SystemExit):
        cli.parse_args(_args("replay", tmp_path, *switches))


def test_teleoperation_has_no_synthetic_offline_mode(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(_args("teleop", tmp_path, "--offline"))


@pytest.mark.parametrize(
    ("mode", "option"),
    [
        ("replay", "--source-hz"),
        ("replay", "--control-hz"),
        ("replay", "--first-speed-scale"),
        ("replay", "--second-speed-scale"),
        ("replay", "--max-joint-step-rad"),
        ("replay", "--max-gripper-step-m"),
        ("replay", "--move-joint-velocity"),
        ("replay", "--move-gripper-velocity"),
        ("replay", "--settle-timeout-s"),
        ("replay", "--joint-tolerance"),
        ("replay", "--gripper-tolerance-m"),
        ("teleop", "--fps"),
        ("teleop", "--duration"),
        ("teleop", "--feedback-timeout-s"),
        ("teleop", "--max-lateness-s"),
        ("teleop", "--camera-frame-timeout-s"),
        ("teleop", "--gripper-max-m"),
    ],
)
@pytest.mark.parametrize("value", ["0", "nan", "inf"])
def test_nonpositive_or_nonfinite_rates_are_rejected(tmp_path, mode, option, value):
    with pytest.raises(SystemExit):
        cli.parse_args(_args(mode, tmp_path, option, value))


@pytest.mark.parametrize("mode", ["replay", "teleop"])
@pytest.mark.parametrize(
    ("left", "right"), [("can_left", "can_left"), (" can_left ", "can_left"), (" ", "can_right")]
)
def test_can_interfaces_must_be_distinct_and_nonempty(tmp_path, mode, left, right):
    with pytest.raises(SystemExit):
        cli.parse_args(_args(mode, tmp_path, "--left-can", left, "--right-can", right))


def test_can_defaults_and_whitespace_normalization(tmp_path):
    args = cli.parse_args(_args("teleop", tmp_path, "--left-can", " can_left "))
    assert (args.left_can, args.right_can) == ("can_left", "can_right")
    assert args.output_dir == tmp_path.resolve()


@pytest.mark.parametrize(
    "serial_options",
    [
        ["--head-camera", "123"],
        ["--head-camera", "123", "--left-wrist-camera", "456"],
        ["--head-camera", "123", "--left-wrist-camera", "123", "--right-wrist-camera", "789"],
        ["--head-camera", "123", "--left-wrist-camera", " 123 ", "--right-wrist-camera", "789"],
        ["--head-camera", "123", "--left-wrist-camera", " ", "--right-wrist-camera", "789"],
        ["--head-camera", ""],
        ["--no-cameras", "--head-camera", "123"],
    ],
)
def test_invalid_camera_serial_sets_are_rejected(tmp_path, serial_options):
    with pytest.raises(SystemExit):
        cli.parse_args(_args("teleop", tmp_path, *serial_options))


def test_execute_requires_all_cameras_or_explicit_no_cameras(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(_args("teleop", tmp_path, "--execute", "--confirm", cli.TELEOP_CONFIRMATION))
    args = cli.parse_args(
        _args(
            "teleop",
            tmp_path,
            "--execute",
            "--confirm",
            cli.TELEOP_CONFIRMATION,
            "--head-camera",
            "123",
            "--left-wrist-camera",
            "456",
            "--right-wrist-camera",
            "789",
        )
    )
    assert args.execute
    assert not args.no_cameras


def test_camera_serial_whitespace_is_normalized(tmp_path):
    args = cli.parse_args(
        _args(
            "teleop",
            tmp_path,
            "--head-camera",
            " 123 ",
            "--left-wrist-camera",
            "456 ",
            "--right-wrist-camera",
            " 789",
        )
    )
    assert (args.head_camera, args.left_wrist_camera, args.right_wrist_camera) == ("123", "456", "789")


@pytest.mark.parametrize(
    ("mode", "options"),
    [
        ("replay", ["--control-hz", "201"]),
        ("replay", ["--speed-percent", "0"]),
        ("replay", ["--speed-percent", "101"]),
        ("replay", ["--episode-start", "5", "--episode-end", "4"]),
        ("teleop", ["--fps", "201"]),
        ("teleop", ["--num-episodes", "0"]),
        ("teleop", ["--start-index", "-1"]),
        ("teleop", ["--camera-fps", "0"]),
        ("teleop", ["--gripper-max-m", "0.101"]),
    ],
)
def test_sampling_ranges_and_episode_parameters_are_validated(tmp_path, mode, options):
    with pytest.raises(SystemExit):
        cli.parse_args(_args(mode, tmp_path, *options))


@pytest.mark.parametrize(("column", "value"), [(0, float("nan")), (1, -0.1), (2, 0.1), (6, 0.101)])
def test_initial_state_must_be_finite_and_within_joint_and_gripper_limits(tmp_path, column, value):
    initial = np.zeros(14)
    initial[column] = value
    with pytest.raises(SystemExit):
        cli.parse_args(_args("replay", tmp_path, "--initial-state", *map(str, initial)))


def test_initial_state_tracks_selected_gripper_range(tmp_path):
    args = cli.parse_args(_args("replay", tmp_path, "--gripper-max-m", "0.07"))
    np.testing.assert_allclose(args.initial_state, [0, 0.05, -0.05, 0, 0, 0, 0.0695] * 2)
    initial = np.zeros(14)
    initial[13] = 0.08
    with pytest.raises(SystemExit):
        cli.parse_args(
            _args("replay", tmp_path, "--gripper-max-m", "0.07", "--initial-state", *map(str, initial))
        )


def test_directory_discovery_naturally_sorts_and_deduplicates_sources(tmp_path):
    sources = tmp_path / "source"
    for name in ("episode10.hdf5", "episode2.h5", "episode0.hdf5", "episode11.hdf5"):
        _write_source(sources / name)
    (sources / "notes.txt").write_text("Not an episode")
    args = cli.parse_args(
        ["replay", str(sources), str(sources / "episode2.h5"), "--output-dir", str(tmp_path / "out")]
    )

    assert [path.name for path in cli.discover_episodes(args)] == [
        "episode0.hdf5",
        "episode2.h5",
        "episode10.hdf5",
        "episode11.hdf5",
    ]


def test_inclusive_episode_range_and_output_renumbering(tmp_path):
    sources = tmp_path / "source"
    for name in ("episode0.hdf5", "episode_000002.h5", "episode10.hdf5", "episode11.hdf5"):
        _write_source(sources / name)
    args = cli.parse_args(
        [
            "replay",
            str(sources),
            "--output-dir",
            str(tmp_path / "out"),
            "--episode-start",
            "2",
            "--episode-end",
            "10",
            "--output-start-index",
            "50",
        ]
    )

    prepared = cli.prepare_replay(args)

    assert [source.path.name for source, _, _, _ in prepared] == ["episode_000002.h5", "episode10.hdf5"]
    assert [output.name for _, _, _, output in prepared] == ["episode_000050.hdf5", "episode_000051.hdf5"]
    assert not args.output_dir.exists()


def test_empty_range_and_unsupported_input_are_rejected(tmp_path):
    source = tmp_path / "episode0.hdf5"
    _write_source(source)
    args = cli.parse_args(
        ["replay", str(source), "--output-dir", str(tmp_path / "out"), "--episode-start", "1"]
    )
    with pytest.raises(ValueError, match="No source episodes"):
        cli.discover_episodes(args)
    invalid = tmp_path / "episode.pkl"
    invalid.touch()
    args = cli.parse_args(["replay", str(invalid), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="HDF5"):
        cli.discover_episodes(args)


def test_duplicate_output_names_are_rejected_before_writing(tmp_path):
    sources = [tmp_path / "first" / "episode0.hdf5", tmp_path / "second" / "episode0.h5"]
    for source in sources:
        _write_source(source)
    output_dir = tmp_path / "out"
    args = cli.parse_args(["replay", *map(str, sources), "--output-dir", str(output_dir)])
    with pytest.raises(FileExistsError, match="collide"):
        cli.prepare_replay(args)
    assert not output_dir.exists()


@pytest.mark.parametrize("mode", ["replay", "teleop"])
def test_existing_output_is_never_overwritten(tmp_path, mode, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "episode_000000.hdf5"
    existing.write_bytes(b"existing user episode")
    forbidden = _forbid_hardware(monkeypatch)
    if mode == "replay":
        source = tmp_path / "episode_000000.hdf5"
        _write_source(source)
        argv = ["replay", str(source), "--output-dir", str(output_dir), "--offline"]
    else:
        argv = ["teleop", "--output-dir", str(output_dir)]
    with pytest.raises(FileExistsError):
        cli.main(argv)
    forbidden.assert_not_called()
    assert existing.read_bytes() == b"existing user episode"


@pytest.mark.parametrize("explicit", [False, True])
def test_replay_preview_never_connects_or_writes(tmp_path, monkeypatch, explicit):
    source = tmp_path / "episode0.hdf5"
    _write_source(source)
    output_dir = tmp_path / "out"
    forbidden = _forbid_hardware(monkeypatch)
    writer = Mock(side_effect=AssertionError("Preview opened an output writer"))
    monkeypatch.setattr(cli, "HDF5EpisodeWriter", writer)
    argv = ["replay", str(source), "--output-dir", str(output_dir)]
    if explicit:
        argv.append("--dry-run")

    assert cli.main(argv) == 0

    forbidden.assert_not_called()
    writer.assert_not_called()
    assert not output_dir.exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_teleop_preview_never_connects_or_writes(tmp_path, monkeypatch, explicit):
    output_dir = tmp_path / "out"
    forbidden = _forbid_hardware(monkeypatch)
    writer = Mock(side_effect=AssertionError("Preview opened an output writer"))
    monkeypatch.setattr(cli, "HDF5EpisodeWriter", writer)
    argv = ["teleop", "--output-dir", str(output_dir), "--num-episodes", "2", "--duration", "1"]
    if explicit:
        argv.append("--dry-run")

    assert cli.main(argv) == 0

    forbidden.assert_not_called()
    writer.assert_not_called()
    assert not output_dir.exists()


def test_offline_replay_writes_only_aligned_synthetic_second_pass_data(tmp_path, monkeypatch):
    source = tmp_path / "episode3.hdf5"
    source_values = _write_source(source)
    output_dir = tmp_path / "out"
    forbidden = _forbid_hardware(monkeypatch)

    assert (
        cli.main(
            [
                "replay",
                str(source),
                "--output-dir",
                str(output_dir),
                "--offline",
                "--first-speed-scale",
                "0.25",
                "--second-speed-scale",
                "0.5",
                "--output-start-index",
                "12",
                "--first-pass-no-grasp",
            ]
        )
        == 0
    )

    forbidden.assert_not_called()
    outputs = list(output_dir.iterdir())
    assert [path.name for path in outputs] == ["episode_000012.hdf5"]
    with h5py.File(outputs[0], "r") as handle:
        assert bool(handle.attrs["sim"]) is True
        assert handle.attrs["backend"] == "offline"
        assert bool(handle.attrs["first_pass_recorded"]) is False
        assert handle.attrs["workflow"] == "bi_piper_two_pass_replay_collect"
        assert handle.attrs["source_episode"] == str(source.resolve())
        assert handle.attrs["fps"] == 20
        assert handle.attrs["capture_target_fps"] == 10
        assert handle.attrs["capture_fps"] == pytest.approx(10)
        assert handle.attrs["action_source"] == "synthetic_source_target"
        assert handle.attrs["joint_action_source"] == "synthetic_source_target"
        assert handle.attrs["frame_alignment"] == "one_synthetic_frame_per_source_frame"
        np.testing.assert_array_equal(handle["recording_metadata/source_frame_index"][:], [0, 1, 2])
        np.testing.assert_array_equal(
            handle["recording_metadata/timestamp_ns"][:], [1, 100_000_001, 200_000_001]
        )
        np.testing.assert_allclose(handle["action"][:], source_values)
        np.testing.assert_allclose(handle["joint_action/vector"][:], source_values)
        np.testing.assert_array_equal(handle["recording_metadata/action_available"][:], True)
        assert not bool(handle["endpose/left_available"][:].any())
        assert len(handle["observation"]) == 0


def test_replay_camera_rate_must_cover_every_source_frame(tmp_path):
    source = tmp_path / "episode0.hdf5"
    _write_source(source, fps=100)
    args = cli.parse_args(
        [
            "replay",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--execute",
            "--confirm",
            cli.REPLAY_CONFIRMATION,
            "--control-hz",
            "100",
            "--head-camera",
            "123",
            "--left-wrist-camera",
            "456",
            "--right-wrist-camera",
            "789",
        ]
    )
    with pytest.raises(ValueError, match="Camera FPS"):
        cli.prepare_replay(args)


def test_teleop_camera_rate_is_checked_before_preflight(tmp_path, monkeypatch):
    forbidden = _forbid_hardware(monkeypatch)
    with pytest.raises(ValueError, match="Camera FPS"):
        cli.main(
            [
                "teleop",
                "--output-dir",
                str(tmp_path / "out"),
                "--fps",
                "40",
                "--execute",
                "--confirm",
                cli.TELEOP_CONFIRMATION,
                "--head-camera",
                "123",
                "--left-wrist-camera",
                "456",
                "--right-wrist-camera",
                "789",
            ]
        )
    forbidden.assert_not_called()
    assert not (tmp_path / "out").exists()
