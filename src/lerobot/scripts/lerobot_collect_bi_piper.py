"""Collect dual-Piper HDF5 episodes through two-pass replay or shared-CAN teleoperation."""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import socket
import subprocess
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots.bi_piper.collection import (
    BiPiperCollectionSession,
    collect_shared_can_episode,
    run_two_pass_episode,
    validate_position,
)
from lerobot.robots.bi_piper.hdf5_recording import CollectionSample, HDF5EpisodeWriter
from lerobot.robots.bi_piper.replay_collection import build_replay_plan, load_replay_source

REPLAY_CONFIRMATION = "EXECUTE BI_PIPER REPLAY COLLECT"
TELEOP_CONFIRMATION = "RECORD BI_PIPER SHARED CAN TELEOP"
logger = logging.getLogger(__name__)


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    replay = commands.add_parser(
        "replay", help="Replay once, reset, replay again and collect one row per source frame"
    )
    replay.add_argument("episodes", nargs="+", type=Path, help="Source HDF5 files or directories")
    replay.add_argument("--source-hz", type=_positive, help="Override source metadata FPS; fallback is 25")
    replay.add_argument("--control-hz", type=_positive, default=30)
    replay.add_argument("--first-speed-scale", type=_positive, default=0.5)
    replay.add_argument("--second-speed-scale", type=_positive, default=0.5)
    replay.add_argument("--first-pass-no-grasp", action="store_true")
    replay.add_argument("--max-joint-step-rad", type=_positive, default=0.15)
    replay.add_argument("--max-gripper-step-m", type=_positive, default=0.01)
    replay.add_argument("--move-joint-velocity", type=_positive, default=0.5)
    replay.add_argument("--move-gripper-velocity", type=_positive, default=0.05)
    replay.add_argument("--settle-timeout-s", type=_positive, default=5)
    replay.add_argument("--joint-tolerance", type=_positive, default=0.08)
    replay.add_argument("--gripper-tolerance-m", type=_positive, default=0.005)
    replay.add_argument("--initial-state", nargs=14, type=float, metavar="VALUE")
    replay.add_argument("--speed-percent", type=int, default=30)
    replay.add_argument("--left-installation-pos", type=int, choices=(0, 1, 2, 3), default=1)
    replay.add_argument("--right-installation-pos", type=int, choices=(0, 1, 2, 3), default=1)
    replay.add_argument("--episode-start", type=_nonnegative_int)
    replay.add_argument("--episode-end", type=_nonnegative_int)
    replay.add_argument("--output-start-index", type=_nonnegative_int)
    teleop = commands.add_parser(
        "teleop", help="Passively record firmware 1:1 leader/follower pairs on shared CAN"
    )
    teleop.add_argument("--fps", type=_positive, default=25)
    teleop.add_argument("--duration", type=_positive, default=60)
    teleop.add_argument("--num-episodes", type=_positive_int, default=1)
    teleop.add_argument("--start-index", type=_nonnegative_int, default=0)
    for command in (replay, teleop):
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--task", default="")
        command.add_argument("--left-can", default="can_left")
        command.add_argument("--right-can", default="can_right")
        command.add_argument("--gripper-max-m", type=_positive, default=0.1)
        command.add_argument("--feedback-timeout-s", type=_positive, default=0.5)
        command.add_argument("--max-lateness-s", type=_positive, default=0.5)
        command.add_argument("--head-camera", help="RealSense serial for cam_high")
        command.add_argument("--left-wrist-camera", help="RealSense serial for cam_left_wrist")
        command.add_argument("--right-wrist-camera", help="RealSense serial for cam_right_wrist")
        command.add_argument("--camera-width", type=_positive_int, default=640)
        command.add_argument("--camera-height", type=_positive_int, default=480)
        command.add_argument("--camera-fps", type=_positive_int, default=30)
        command.add_argument("--camera-stream-mode", choices=("rgb", "rgbd"), default="rgbd")
        command.add_argument("--camera-frame-timeout-s", type=_positive, default=0.5)
        command.add_argument("--exposure", type=_positive_int)
        command.add_argument("--gain", type=_nonnegative_int)
        command.add_argument("--no-cameras", action="store_true", help="Explicitly record robot state only")
        gate = command.add_mutually_exclusive_group()
        gate.add_argument("--dry-run", action="store_true", help="Preview only; this is also the default")
        gate.add_argument("--execute", action="store_true", help="Open hardware; requires --confirm")
        if command is replay:
            gate.add_argument(
                "--offline",
                action="store_true",
                help="Write explicitly marked synthetic data without hardware",
            )
        command.add_argument(
            "--confirm", help="Exact replay/teleop acknowledgement printed in the documentation"
        )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.left_can, args.right_can = args.left_can.strip(), args.right_can.strip()
    if not args.left_can or not args.right_can or args.left_can == args.right_can:
        parser.error("left and right CAN interfaces must be non-empty and distinct")
    if args.gripper_max_m > 0.1:
        parser.error("--gripper-max-m cannot exceed 0.1 meters")
    expected_confirmation = REPLAY_CONFIRMATION if args.mode == "replay" else TELEOP_CONFIRMATION
    if args.execute and args.confirm != expected_confirmation:
        parser.error(f"--execute requires --confirm '{expected_confirmation}'")
    if not args.execute and args.confirm is not None:
        parser.error("--confirm requires --execute")
    for field in ("head_camera", "left_wrist_camera", "right_wrist_camera"):
        value = getattr(args, field)
        if value is not None:
            value = value.strip()
            if not value:
                parser.error("Camera serials cannot be empty")
            setattr(args, field, value)
    serials = (args.head_camera, args.left_wrist_camera, args.right_wrist_camera)
    if args.no_cameras and any(serials):
        parser.error("--no-cameras cannot be combined with camera serial arguments")
    if any(serials) and not all(serials):
        parser.error("Specify all three camera serials together")
    if all(serials) and (any(not serial.strip() for serial in serials) or len(set(serials)) != 3):
        parser.error("Three camera serials must be non-empty and distinct")
    if args.execute and not args.no_cameras and not all(serials):
        parser.error("Hardware recording requires three camera serials or explicit --no-cameras")
    if args.mode == "replay":
        if not 1 <= args.speed_percent <= 100:
            parser.error("--speed-percent must be in [1,100]")
        if args.control_hz > 200:
            parser.error("--control-hz must be <= 200")
        if (
            args.episode_start is not None
            and args.episode_end is not None
            and args.episode_start > args.episode_end
        ):
            parser.error("--episode-start must be <= --episode-end")
        initial = args.initial_state or [0, 0.05, -0.05, 0, 0, 0, max(0, args.gripper_max_m - 0.0005)] * 2
        try:
            args.initial_state = validate_position(np.asarray(initial), args.gripper_max_m)
        except ValueError as error:
            parser.error(str(error))
    elif args.fps > 200:
        parser.error("--fps must be <= 200")
    return args


def _episode_index(path: Path) -> int | None:
    match = re.fullmatch(r"episode_?(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match[1]) if match else None


def discover_episodes(args: argparse.Namespace) -> list[Path]:
    files: set[Path] = set()
    for candidate in args.episodes:
        candidate = candidate.expanduser().resolve()
        if candidate.is_dir():
            files.update(
                path.resolve()
                for path in candidate.iterdir()
                if path.suffix.lower() in {".h5", ".hdf5"} and path.is_file()
            )
        elif candidate.is_file() and candidate.suffix.lower() in {".h5", ".hdf5"}:
            files.add(candidate)
        else:
            raise ValueError(f"Expected an HDF5 episode or directory: {candidate}")
    if args.episode_start is not None or args.episode_end is not None:
        files = {
            path
            for path in files
            if (index := _episode_index(path)) is not None
            and (args.episode_start is None or index >= args.episode_start)
            and (args.episode_end is None or index <= args.episode_end)
        }

    # Tag components so differently named files remain comparable under Python 3.
    def key(path: Path) -> tuple[Any, ...]:
        index = _episode_index(path)
        if index is not None:
            return (0, index, str(path))
        components = [
            (0, int(part)) if part.isdigit() else (1, part.lower()) for part in re.split(r"(\d+)", str(path))
        ]
        return (1, components, str(path))

    selected = sorted(files, key=key)
    if not selected:
        raise ValueError("No source episodes matched the requested range")
    return selected


def prepare_replay(args: argparse.Namespace) -> list[tuple[Any, Any, Any, Path]]:
    prepared = []
    outputs: set[Path] = set()
    for offset, path in enumerate(discover_episodes(args)):
        source = load_replay_source(path, source_hz=args.source_hz, gripper_max_m=args.gripper_max_m)
        options = {
            "control_hz": args.control_hz,
            "max_joint_step_rad": args.max_joint_step_rad,
            "max_gripper_step_m": args.max_gripper_step_m,
            "gripper_max_m": args.gripper_max_m,
        }
        first = build_replay_plan(
            source, speed_scale=args.first_speed_scale, no_grasp=args.first_pass_no_grasp, **options
        )
        second = build_replay_plan(source, speed_scale=args.second_speed_scale, **options)
        name = (
            f"episode_{args.output_start_index + offset:06d}.hdf5"
            if args.output_start_index is not None
            else f"{path.stem}.hdf5"
        )
        output = args.output_dir / name
        if os.path.lexists(output) or output in outputs:
            raise FileExistsError(f"Output exists or source names collide: {output}")
        outputs.add(output)
        if args.execute and not args.no_cameras and args.camera_fps < source.source_hz * second.speed_scale:
            raise ValueError("Camera FPS must be at least the second-pass source frame rate")
        prepared.append((source, first, second, output))
    return prepared


def _close(resource: Any) -> None:
    try:
        resource.disconnect()
    except Exception:
        logger.exception("Collection resource cleanup failed")


def _connect_cameras(args: argparse.Namespace, stack: ExitStack) -> Any | None:
    if args.no_cameras:
        return None
    from lerobot.robots.bi_piper.collection_cameras import RealSenseCollectionCameras

    cameras = RealSenseCollectionCameras(
        dict(
            zip(
                ("cam_high", "cam_left_wrist", "cam_right_wrist"),
                (args.head_camera, args.left_wrist_camera, args.right_wrist_camera),
                strict=True,
            )
        ),
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        use_depth=args.camera_stream_mode == "rgbd",
        frame_timeout_s=args.camera_frame_timeout_s,
        exposure=args.exposure,
        gain=args.gain,
    )
    stack.callback(_close, cameras)
    cameras.connect()
    return cameras


def _preflight(args: argparse.Namespace) -> None:
    # An installed CLI must not depend on examples/ being importable.
    for name in (args.left_can, args.right_can):
        socket.if_nametoindex(name)
        if int((Path("/sys/class/net") / name / "type").read_text().strip()) != 280:
            raise RuntimeError(f"{name} is not a SocketCAN interface")
        result = subprocess.run(  # nosec B607: fixed diagnostic; iface name validated above
            ["ip", "-details", "link", "show", "dev", name], check=True, capture_output=True, text=True
        )
        if "can state ERROR-ACTIVE" not in result.stdout or not re.search(
            r"\bbitrate\s+1000000\b", result.stdout
        ):
            raise RuntimeError(f"{name} must be ERROR-ACTIVE at 1 Mbit/s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "replay":
        prepared = prepare_replay(args)
        for source, first, second, output in prepared:
            print(
                f"{source.path.name}: frames={len(source.positions)}, source_hz={source.source_hz:g}, "
                f"pass1={first.duration_s:g}s, pass2={second.duration_s:g}s, output={output}"
            )
        if args.offline:
            for source, _first, second, output in prepared:
                writer = HDF5EpisodeWriter(
                    output,
                    fps=source.source_hz,
                    gripper_max_m=args.gripper_max_m,
                    expected_frames=len(source.positions),
                    metadata={
                        "sim": True,
                        "backend": "offline",
                        "workflow": "bi_piper_two_pass_replay_collect",
                        "source_episode": str(source.path),
                        "first_pass_recorded": False,
                        "capture_target_fps": source.source_hz * second.speed_scale,
                        "action_source": "synthetic_source_target",
                        "frame_alignment": "one_synthetic_frame_per_source_frame",
                        "joint_action_source": "synthetic_source_target",
                    },
                )
                try:
                    for index, position in enumerate(source.positions):
                        writer.append(
                            CollectionSample(
                                position=position,
                                action=position,
                                timestamp_ns=1 + round(index / (source.source_hz * second.speed_scale) * 1e9),
                                source_frame_index=index,
                            )
                        )
                    writer.finalize()
                except BaseException:
                    writer.abort()
                    raise
            print("离线合成 HDF5 已生成（sim=True, backend=offline），未连接硬件。")
            return 0
        if not args.execute:
            print("仅预览：未打开 CAN、相机或模型，未写入数据。")
            return 0
    else:
        if not args.no_cameras and args.execute and args.camera_fps < args.fps:
            raise ValueError("Camera FPS must be at least the teleoperation sampling FPS")
        outputs = [
            args.output_dir / f"episode_{args.start_index + index:06d}.hdf5"
            for index in range(args.num_episodes)
        ]
        for output in outputs:
            if os.path.lexists(output):
                raise FileExistsError(output)
        print(f"左右主从共享 {args.left_can}/{args.right_can}；固件 1:1 跟随，采集进程仅监听。")
        print(
            f"episodes={args.num_episodes}, frames/episode={math.ceil(args.duration * args.fps)}, output={args.output_dir}"
        )
        if not args.execute:
            print("仅预览：未打开 CAN 或相机，未写入数据。")
            return 0
    _preflight(args)
    with ExitStack() as stack:
        cameras = _connect_cameras(args, stack)
        if args.mode == "replay":
            session = BiPiperCollectionSession(
                left_can=args.left_can,
                right_can=args.right_can,
                gripper_max_m=args.gripper_max_m,
                speed_percent=args.speed_percent,
                feedback_timeout_s=args.feedback_timeout_s,
                left_installation_pos=args.left_installation_pos,
                right_installation_pos=args.right_installation_pos,
                control_hz=args.control_hz,
                move_joint_velocity=args.move_joint_velocity,
                move_gripper_velocity=args.move_gripper_velocity,
                settle_timeout_s=args.settle_timeout_s,
                joint_tolerance=args.joint_tolerance,
                gripper_tolerance_m=args.gripper_tolerance_m,
            )
            stack.callback(_close, session)
            input("双遍回放将移动双臂。确认工作空间和急停后，按 Enter 连接并开始：")
            session.connect()
            for source, first, second, output in prepared:
                result = run_two_pass_episode(
                    session,
                    source,
                    first,
                    second,
                    output,
                    initial_position=args.initial_state,
                    gripper_max_m=args.gripper_max_m,
                    cameras=cameras,
                    max_lateness_s=args.max_lateness_s,
                    metadata={
                        "task": args.task,
                        "first_pass_gripper_mode": "always_open"
                        if args.first_pass_no_grasp
                        else "trajectory",
                    },
                )
                print(f"已保存：{result}")
                input("双臂已回初态。按 Enter 进入下一条或结束：")
        else:
            from lerobot.robots.bi_piper.shared_can import SharedCANBiPiper

            session = SharedCANBiPiper(
                left_can=args.left_can,
                right_can=args.right_can,
                feedback_timeout_s=args.feedback_timeout_s,
                gripper_max_m=args.gripper_max_m,
            )
            stack.callback(_close, session)
            session.connect()
            for output in outputs:
                input(f"按 Enter 开始采集 {output.name}（{args.duration:g} 秒）；主从跟随始终由硬件运行：")
                result = collect_shared_can_episode(
                    session,
                    output,
                    duration_s=args.duration,
                    fps=args.fps,
                    gripper_max_m=args.gripper_max_m,
                    cameras=cameras,
                    max_lateness_s=args.max_lateness_s,
                    metadata={"task": args.task, "left_can": args.left_can, "right_can": args.right_can},
                )
                print(f"已保存：{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
