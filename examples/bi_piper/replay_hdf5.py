#!/usr/bin/env python

"""Validate and replay the requested bimanual Piper HDF5 trajectory (default 15 Hz).

The default mode only inspects the file.  Real execution requires both
``--execute`` and a fixed confirmation phrase.  Both CAN interfaces are checked
before either arm is enabled, and the measured start pose must be close to the
selected first frame.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

try:
    from .common import (
        DEFAULT_TRAJECTORY,
        add_can_arguments,
        resolve_repo_path,
        validate_can_interfaces,
        validate_can_names,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        DEFAULT_TRAJECTORY,
        add_can_arguments,
        resolve_repo_path,
        validate_can_interfaces,
        validate_can_names,
    )


logger = logging.getLogger(__name__)
JOINT_VECTOR_INDICES = tuple(index for index in range(14) if index not in (6, 13))
EXECUTION_CONFIRMATION = "EXECUTE BI_PIPER HDF5 REPLAY"


def _max_adjacent_steps(actions: np.ndarray) -> tuple[float, float]:
    if len(actions) < 2:
        return 0.0, 0.0
    delta = np.abs(np.diff(actions, axis=0))
    return float(delta[:, JOINT_VECTOR_INDICES].max()), float(delta[:, (6, 13)].max())


def _validate_slice(start_frame: int, end_frame: int | None, num_frames: int) -> int:
    stop = num_frames if end_frame is None else end_frame
    if not 0 <= start_frame < num_frames:
        raise ValueError(f"start_frame must be in [0, {num_frames}), got {start_frame}")
    if not start_frame < stop <= num_frames:
        raise ValueError(f"end_frame must satisfy {start_frame} < end_frame <= {num_frames}, got {stop}")
    return stop


def _make_robot(args: argparse.Namespace):
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.bi_piper import BiPiperConfig, PiperArmConfig

    def arm_config(can_name: str, installation_pos: int) -> PiperArmConfig:
        return PiperArmConfig(
            can_name=can_name,
            speed_percent=args.speed_percent,
            installation_pos=installation_pos,
            enable_on_connect=True,
            disable_on_disconnect=args.disable_on_disconnect,
            gripper_unit="normalized",
            gripper_range_m=(0.0, 0.1),
            # HDF5 actions are absolute joint targets.  Applying a relative
            # feedback-based clamp here changes the recorded trajectory when
            # the arm lags behind, and the altered target may already have
            # been sent before replay can report it.  Adjacent-frame limits
            # are validated below instead.
            max_relative_target=None,
        )

    config = BiPiperConfig(
        id=args.robot_id,
        left_arm_config=arm_config(args.left_can, args.left_installation_pos),
        right_arm_config=arm_config(args.right_can, args.right_installation_pos),
    )
    return make_robot_from_config(config)


def _assert_start_pose(robot, first_action: dict[str, float], args: argparse.Namespace) -> None:
    observation = robot.get_observation()
    violations: list[str] = []
    for name, target in first_action.items():
        present = float(observation[name])
        limit = args.max_gripper_step if "gripper" in name else args.max_joint_step_rad
        error = abs(target - present)
        if not math.isfinite(present) or not math.isfinite(target) or not math.isfinite(error):
            violations.append(f"{name}: measured={present}, first={target}, delta is non-finite")
        elif error > limit:
            violations.append(
                f"{name}: measured={present:.5f}, first={target:.5f}, delta={error:.5f}, limit={limit:.5f}"
            )
    if violations:
        raise RuntimeError(
            "Robot is not close enough to the first replay frame. Move it to the recorded start pose "
            "before retrying; no trajectory command was sent:\n- " + "\n- ".join(violations)
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--robot-id", default="bi_piper_hdf5_replay")
    parser.add_argument("--speed-percent", type=int, default=30)
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=0.15,
        help="Per-frame safety bound and maximum allowed initial joint error",
    )
    parser.add_argument(
        "--max-gripper-step",
        type=float,
        default=0.10,
        help="Per-frame normalized gripper safety bound and maximum allowed initial error",
    )
    parser.add_argument(
        "--disable-on-disconnect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable arm and gripper torque after replay (default: false). "
            "Keep torque enabled unless the arms are mechanically supported."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Arm the real-hardware path; also requires the exact --confirm phrase",
    )
    parser.add_argument(
        "--confirm",
        metavar="PHRASE",
        help=f"Required with --execute; exact value: {EXECUTION_CONFIRMATION!r}",
    )
    add_can_arguments(parser)
    args = parser.parse_args(argv)
    validate_can_names(args, parser)
    if not math.isfinite(args.control_hz) or args.control_hz <= 0:
        parser.error("--control-hz must be positive and finite")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1, 100]")
    if (
        not math.isfinite(args.max_joint_step_rad)
        or args.max_joint_step_rad <= 0
        or not math.isfinite(args.max_gripper_step)
        or args.max_gripper_step <= 0
    ):
        parser.error("action step limits must be positive and finite")
    if args.execute and args.confirm != EXECUTION_CONFIRMATION:
        parser.error(f"--execute requires --confirm {EXECUTION_CONFIRMATION!r}")
    if not args.execute and args.confirm is not None:
        parser.error("--confirm is only valid together with --execute")
    args.trajectory = resolve_repo_path(args.trajectory)
    return args


def main(argv: list[str] | None = None) -> int:
    from lerobot.robots.bi_piper.hdf5_replay import load_hdf5_trajectory, replay_hdf5_trajectory

    args = parse_args(argv)
    trajectory = load_hdf5_trajectory(args.trajectory)
    stop = _validate_slice(args.start_frame, args.end_frame, trajectory.num_frames)
    selected = trajectory.actions[args.start_frame : stop]
    joint_step, gripper_step = _max_adjacent_steps(selected)
    if joint_step > args.max_joint_step_rad:
        raise ValueError(
            f"Trajectory maximum adjacent joint step {joint_step:.6f} rad exceeds "
            f"--max-joint-step-rad={args.max_joint_step_rad:.6f}"
        )
    if gripper_step > args.max_gripper_step:
        raise ValueError(
            f"Trajectory maximum adjacent gripper step {gripper_step:.6f} exceeds "
            f"--max-gripper-step={args.max_gripper_step:.6f}"
        )

    summary = {
        "mode": "execute" if args.execute else "validate_only",
        "trajectory": str(trajectory.source.resolve()),
        "frames": int(len(selected)),
        "start_frame": args.start_frame,
        "end_frame_exclusive": stop,
        "control_hz": args.control_hz,
        "expected_duration_s": len(selected) / args.control_hz,
        "gripper_encoding": trajectory.gripper_encoding,
        "joint_units": trajectory.joint_units,
        "vector_order": trajectory.vector_order,
        "max_adjacent_joint_step_rad": joint_step,
        "max_adjacent_gripper_step": gripper_step,
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        print(
            "Validation complete. Real execution additionally requires "
            f"--execute --confirm {EXECUTION_CONFIRMATION!r}."
        )
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_can_interfaces(args.left_can, args.right_can)
    robot = _make_robot(args)
    completed = False
    try:
        robot.connect()
        if not robot.is_connected:
            raise ConnectionError("BiPiper did not report a connected state")
        first_action = trajectory.action(args.start_frame, gripper_unit="normalized")
        _assert_start_pose(robot, first_action, args)
        logger.info("Start pose accepted; replaying %d frames at %.3f Hz", len(selected), args.control_hz)
        sent = replay_hdf5_trajectory(
            robot,
            trajectory,
            control_hz=args.control_hz,
            start_frame=args.start_frame,
            end_frame=stop,
            gripper_unit="normalized",
        )
        logger.info("Replay complete: %d frames sent", sent)
        completed = True
    except Exception:
        # An altered command, feedback fault, or operator interruption must
        # never fall through to automatic torque disable: the arms may be
        # carrying their own weight and need to remain supported until an
        # operator has secured them and recovered the hardware explicitly.
        robot.preserve_torque_for_manual_recovery()
        logger.critical(
            "HDF5 replay aborted. Motor torque will be preserved for manual recovery; "
            "support both arms before handling CAN or torque state."
        )
        raise
    finally:
        try:
            robot.disconnect()
        except Exception:
            if completed:
                raise
            logger.exception("BiPiper disconnect failed while handling an aborted replay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
