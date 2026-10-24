#!/usr/bin/env python

"""Launch the complete BiPiper state-inference-action loop.

The policy can run in a local RTC background thread, on the existing Async
Inference gRPC server, or through an OpenPI JAX policy server over WebSocket.
This launcher validates the Pi0.5 contract and requires
an explicit motion confirmation before checking CAN or delegating to
``lerobot-rollout``.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from .common import (
        DEFAULT_CHECKPOINT,
        REPO_ROOT,
        add_camera_arguments,
        add_can_arguments,
        camera_settings,
        validate_camera_arguments,
        validate_can_interfaces,
        validate_can_names,
        validate_checkpoint,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        DEFAULT_CHECKPOINT,
        REPO_ROOT,
        add_camera_arguments,
        add_can_arguments,
        camera_settings,
        validate_camera_arguments,
        validate_can_interfaces,
        validate_can_names,
        validate_checkpoint,
    )


MOTION_CONFIRMATION = "I UNDERSTAND THIS WILL MOVE BOTH PIPER ARMS"


def _default_log_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return REPO_ROOT / "outputs" / "bi_piper_inference_logs" / f"closed_loop_{timestamp}.jsonl"


def build_rollout_command(args: argparse.Namespace, checkpoint: Path, camera_resolutions) -> list[str]:
    cameras = camera_settings(args, camera_resolutions)
    per_arm_delta = {f"j{index}": args.max_joint_delta_rad for index in range(1, 7)}
    per_arm_delta["gripper"] = args.max_gripper_delta
    disable_on_disconnect = str(args.disable_on_disconnect).lower()

    if args.backend == "rtc":
        inference_args = [
            "--inference.type=rtc",
            f"--inference.rtc.execution_horizon={args.execution_horizon}",
            f"--inference.rtc.max_guidance_weight={args.max_guidance_weight}",
            f"--inference.rtc.prefix_attention_schedule={args.prefix_attention_schedule}",
            f"--inference.queue_threshold={args.queue_threshold}",
        ]
        rollout_device = args.device
    elif args.backend == "remote":
        server_policy_path = args.server_policy_path or str(checkpoint)
        inference_args = [
            "--inference.type=remote",
            f"--inference.server_address={args.server_address}",
            f"--inference.policy_path={server_policy_path}",
            f"--inference.policy_device={args.server_policy_device}",
            f"--inference.actions_per_chunk={args.actions_per_chunk}",
            f"--inference.chunk_size_threshold={args.chunk_size_threshold}",
            f"--inference.aggregate_fn_name={args.aggregate_fn_name}",
            f"--inference.connect_timeout_s={args.connect_timeout_s}",
            f"--inference.setup_timeout_s={args.setup_timeout_s}",
            f"--inference.rpc_timeout_s={args.rpc_timeout_s}",
            f"--inference.first_action_timeout_s={args.first_action_timeout_s}",
        ]
        # Remote inference does not load policy weights on the robot computer.
        rollout_device = "cpu"
    else:
        inference_args = [
            "--inference.type=openpi_websocket",
            f"--inference.server_address={args.openpi_server_address}",
            f"--inference.api_key={args.openpi_api_key or ''}",
            f"--inference.action_horizon={args.openpi_action_horizon}",
            f"--inference.open_loop_horizon={args.openpi_open_loop_horizon}",
            f"--inference.image_size={args.openpi_image_size}",
            f"--inference.connect_timeout_s={args.openpi_connect_timeout_s}",
            f"--inference.request_timeout_s={args.openpi_request_timeout_s}",
            f"--inference.first_action_timeout_s={args.openpi_first_action_timeout_s}",
            f"--inference.image_key={args.openpi_image_key}",
            f"--inference.left_wrist_image_key={args.openpi_left_wrist_image_key}",
            f"--inference.right_wrist_image_key={args.openpi_right_wrist_image_key}",
            f"--inference.state_key={args.openpi_state_key}",
            f"--inference.prompt_key={args.openpi_prompt_key}",
        ]
        # OpenPI owns the JAX model; the robot process is CPU-only.
        rollout_device = "cpu"

    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_rollout",
        "--strategy.type=base",
        f"--policy.path={checkpoint}",
        *inference_args,
        f"--robot.type={args.robot_type}",
        f"--robot.id={args.robot_id}",
        f"--robot.left_arm_config.can_name={args.left_can}",
        f"--robot.right_arm_config.can_name={args.right_can}",
        f"--robot.left_arm_config.speed_percent={args.speed_percent}",
        f"--robot.right_arm_config.speed_percent={args.speed_percent}",
        f"--robot.left_arm_config.installation_pos={args.left_installation_pos}",
        f"--robot.right_arm_config.installation_pos={args.right_installation_pos}",
        f"--robot.left_arm_config.disable_on_disconnect={disable_on_disconnect}",
        f"--robot.right_arm_config.disable_on_disconnect={disable_on_disconnect}",
        "--robot.left_arm_config.gripper_unit=normalized",
        "--robot.right_arm_config.gripper_unit=normalized",
        "--robot.left_arm_config.gripper_range_m=[0.0,0.1]",
        "--robot.right_arm_config.gripper_range_m=[0.0,0.1]",
        f"--robot.left_arm_config.gripper_closed_effort_threshold={args.gripper_closed_effort_threshold}",
        f"--robot.right_arm_config.gripper_closed_effort_threshold={args.gripper_closed_effort_threshold}",
        "--robot.left_arm_config.max_relative_target=" + json.dumps(per_arm_delta, separators=(",", ":")),
        "--robot.right_arm_config.max_relative_target=" + json.dumps(per_arm_delta, separators=(",", ":")),
        "--robot.cameras=" + json.dumps(cameras, separators=(",", ":")),
        f"--fps={args.fps}",
        f"--duration={args.duration}",
        f"--device={rollout_device}",
        f"--task={args.task}",
        f"--inference_log_path={args.log_file}",
        f"--display_data={str(args.display_data).lower()}",
        f"--return_to_initial_position={str(args.return_to_initial_position).lower()}",
        "--use_torch_compile=false",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--backend",
        choices=("rtc", "remote", "openpi"),
        default="rtc",
        help="Local RTC, Async Inference gRPC, or OpenPI JAX WebSocket inference",
    )
    parser.add_argument(
        "--robot-type",
        choices=("bi_piper",),
        default="bi_piper",
        help="Use the physical Piper joint order without joint remapping",
    )
    parser.add_argument("--robot-id")
    parser.add_argument("--task", default="Stack blocks")
    parser.add_argument(
        "--log-file",
        type=Path,
        help="JSONL path for per-control-cycle timestamp/state/action telemetry",
    )
    parser.add_argument("--device", default="cuda", help="Policy device for the local RTC backend")
    parser.add_argument("--fps", type=float, default=25.0, help="Robot control-loop frequency")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds; 0 runs until interrupted")
    parser.add_argument("--speed-percent", type=int, default=25)
    parser.add_argument("--execution-horizon", type=int, default=25)
    parser.add_argument("--queue-threshold", type=int, default=45)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--prefix-attention-schedule",
        choices=("ZEROS", "ONES", "LINEAR", "EXP"),
        default="EXP",
    )
    parser.add_argument("--server-address", default="localhost:8080")
    parser.add_argument(
        "--server-policy-path",
        help=(
            "Checkpoint path or Hub id as seen by the remote server; defaults to the resolved "
            "local --checkpoint path"
        ),
    )
    parser.add_argument("--server-policy-device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument(
        "--aggregate-fn-name",
        choices=("weighted_average", "latest_only", "average", "conservative"),
        default="weighted_average",
    )
    parser.add_argument("--connect-timeout-s", type=float, default=10.0)
    parser.add_argument("--setup-timeout-s", type=float, default=600.0)
    parser.add_argument("--rpc-timeout-s", type=float, default=30.0)
    parser.add_argument("--first-action-timeout-s", type=float, default=120.0)
    parser.add_argument("--openpi-server-address", default="localhost:8000")
    parser.add_argument("--openpi-api-key")
    parser.add_argument("--openpi-action-horizon", type=int, default=50)
    parser.add_argument("--openpi-open-loop-horizon", type=int, default=1)
    parser.add_argument("--openpi-image-size", type=int, default=224)
    parser.add_argument("--openpi-connect-timeout-s", type=float, default=10.0)
    parser.add_argument("--openpi-request-timeout-s", type=float, default=30.0)
    parser.add_argument("--openpi-first-action-timeout-s", type=float, default=60.0)
    parser.add_argument("--openpi-image-key", default="cam_high")
    parser.add_argument("--openpi-left-wrist-image-key", default="cam_left_wrist")
    parser.add_argument("--openpi-right-wrist-image-key", default="cam_right_wrist")
    parser.add_argument("--openpi-state-key", default="state")
    parser.add_argument("--openpi-prompt-key", default="prompt")
    parser.add_argument("--max-joint-delta-rad", type=float, default=1.0)
    parser.add_argument(
        "--max-gripper-delta",
        type=float,
        default=0.50,
        help="Maximum normalized [0,1] gripper change accepted per control cycle",
    )
    parser.add_argument("--gripper-closed-effort-threshold", type=float, default=1.0)
    parser.add_argument("--display-data", action="store_true")
    parser.add_argument(
        "--return-to-initial-position",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-on-disconnect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Defaults to false so the arms retain support after CAN disconnect",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow the generated rollout command to connect to and move the real arms",
    )
    parser.add_argument(
        "--confirm-motion",
        help=f"Required with --execute; must exactly equal: {MOTION_CONFIRMATION}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the command without opening cameras, CAN, or a policy connection",
    )
    add_can_arguments(parser)
    add_camera_arguments(parser)
    args = parser.parse_args(argv)
    validate_can_names(args, parser)
    if args.robot_id is None:
        args.robot_id = args.robot_type
    if not math.isfinite(args.fps) or args.fps <= 0 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--fps must be positive and finite; --duration must be non-negative and finite")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1, 100]")
    if args.backend == "rtc":
        if args.execution_horizon <= 0 or args.queue_threshold < args.execution_horizon:
            parser.error("--queue-threshold must be at least the positive --execution-horizon")
        if not math.isfinite(args.max_guidance_weight) or args.max_guidance_weight <= 0:
            parser.error("--max-guidance-weight must be positive and finite")
    elif args.backend == "remote":
        args.server_address = args.server_address.strip()
        args.server_policy_device = args.server_policy_device.strip()
        if not args.server_address:
            parser.error("--server-address cannot be empty")
        if not args.server_policy_device:
            parser.error("--server-policy-device cannot be empty")
        if args.server_policy_path is not None and not args.server_policy_path.strip():
            parser.error("--server-policy-path cannot be empty")
        if args.actions_per_chunk <= 0:
            parser.error("--actions-per-chunk must be positive")
        if not math.isfinite(args.chunk_size_threshold) or not 0 <= args.chunk_size_threshold <= 1:
            parser.error("--chunk-size-threshold must be finite and in [0, 1]")
        timeouts = (
            args.connect_timeout_s,
            args.setup_timeout_s,
            args.rpc_timeout_s,
            args.first_action_timeout_s,
        )
        if any(not math.isfinite(value) or value <= 0 for value in timeouts):
            parser.error("remote inference timeouts must be positive and finite")
    else:
        args.openpi_server_address = args.openpi_server_address.strip()
        if not args.openpi_server_address:
            parser.error("--openpi-server-address cannot be empty")
        if (
            args.openpi_action_horizon <= 0
            or args.openpi_open_loop_horizon <= 0
            or args.openpi_image_size <= 0
        ):
            parser.error("OpenPI horizons and image size must be positive")
        timeouts = (
            args.openpi_connect_timeout_s,
            args.openpi_request_timeout_s,
            args.openpi_first_action_timeout_s,
        )
        if any(not math.isfinite(value) or value <= 0 for value in timeouts):
            parser.error("OpenPI inference timeouts must be positive and finite")
    if (
        not math.isfinite(args.max_joint_delta_rad)
        or args.max_joint_delta_rad <= 0
        or not math.isfinite(args.max_gripper_delta)
        or args.max_gripper_delta <= 0
    ):
        parser.error("per-cycle action limits must be positive and finite")
    if not math.isfinite(args.gripper_closed_effort_threshold) or args.gripper_closed_effort_threshold < 0:
        parser.error("--gripper-closed-effort-threshold must be finite and non-negative")
    validate_camera_arguments(args, parser)
    args.log_file = _default_log_path() if args.log_file is None else args.log_file.expanduser()
    if not args.log_file.is_absolute():
        args.log_file = (REPO_ROOT / args.log_file).resolve()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    if args.execute and args.confirm_motion != MOTION_CONFIRMATION:
        parser.error(f"--execute requires --confirm-motion={MOTION_CONFIRMATION!r}")
    if not args.execute and args.confirm_motion is not None:
        parser.error("--confirm-motion is only valid together with --execute")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation_kwargs = (
        {
            "execution_horizon": args.execution_horizon,
            "queue_threshold": args.queue_threshold,
        }
        if args.backend == "rtc"
        else {}
    )
    contract = validate_checkpoint(args.checkpoint, **validation_kwargs)
    if args.backend == "remote" and args.actions_per_chunk > contract.config["chunk_size"]:
        raise ValueError(
            f"--actions-per-chunk={args.actions_per_chunk} exceeds the checkpoint "
            f"chunk_size={contract.config['chunk_size']}"
        )
    requested_resolution = (args.camera_width, args.camera_height)
    mismatched = {
        key: resolution
        for key, resolution in contract.camera_resolutions.items()
        if resolution != requested_resolution
    }
    if mismatched:
        raise ValueError(
            f"Camera resolution {requested_resolution} does not match checkpoint features: {mismatched}"
        )
    command = build_rollout_command(args, contract.path, contract.camera_resolutions)
    if not args.execute:
        print(f"Validated Pi0.5 checkpoint for BiPiper {args.backend} rollout (no hardware opened):")
        print(shlex.join(command))
        if not args.dry_run:
            print(
                "Preview only. To move the real robot, add --execute "
                f"--confirm-motion {shlex.quote(MOTION_CONFIRMATION)}"
            )
        return 0

    validate_can_interfaces(args.left_can, args.right_can)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
