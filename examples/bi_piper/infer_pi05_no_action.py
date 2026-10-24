#!/usr/bin/env python

"""Read three D435 streams and dual-Piper feedback, then log Pi0.5 predictions.

This is the first hardware qualification phase.  It opens CAN in feedback-only
mode, never enables either arm, and never dispatches a robot action.  Each JSONL
record contains the measured 14-D state and the full postprocessed action chunk.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .common import (
        DEFAULT_CHECKPOINT,
        REPO_ROOT,
        add_camera_arguments,
        add_can_arguments,
        camera_settings,
        make_realsense_configs,
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
        make_realsense_configs,
        validate_camera_arguments,
        validate_can_interfaces,
        validate_can_names,
        validate_checkpoint,
    )


logger = logging.getLogger(__name__)


def _parse_openpi_address(address: str) -> tuple[str, int]:
    value = address.strip()
    parsed = urlparse(value if "://" in value else f"ws://{value}")
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
        raise ValueError(f"Invalid OpenPI server address: {address!r}")
    port = parsed.port or 8000
    return parsed.hostname, port


def _default_log_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return REPO_ROOT / "outputs" / "bi_piper_inference_logs" / f"predictions_{timestamp}.jsonl"


def _make_read_only_robot(args: argparse.Namespace, camera_configs: dict[str, Any]):
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.bi_piper import BiPiperConfig, PiperArmConfig

    def arm_config(can_name: str, installation_pos: int) -> PiperArmConfig:
        return PiperArmConfig(
            can_name=can_name,
            installation_pos=installation_pos,
            enable_on_connect=False,
            disable_on_disconnect=False,
            gripper_unit="normalized",
            gripper_range_m=(0.0, 0.1),
        )

    config = BiPiperConfig(
        id=args.robot_id,
        read_only=True,
        left_arm_config=arm_config(args.left_can, args.left_installation_pos),
        right_arm_config=arm_config(args.right_can, args.right_installation_pos),
        cameras=camera_configs,
        emergency_stop_on_error=False,
    )
    return make_robot_from_config(config)


def _load_policy(checkpoint: Path, device: str):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    config.device = device
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(checkpoint, config=config).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def _dataset_features(robot) -> dict[str, dict]:
    from lerobot.utils.feature_utils import hw_to_dataset_features

    policy_inputs = {
        key: value
        for key, value in robot.observation_features.items()
        if isinstance(value, tuple) or (value is float and key.endswith(".pos"))
    }
    return hw_to_dataset_features(policy_inputs, "observation", use_video=False)


def _predict_chunk(
    *,
    policy,
    preprocessor,
    postprocessor,
    raw_observation: dict[str, Any],
    dataset_features: dict[str, dict],
    device: str,
    task: str,
    robot_type: str,
):
    import torch

    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.feature_utils import build_dataset_frame

    frame = build_dataset_frame(dataset_features, raw_observation, prefix="observation")
    model_input = prepare_observation_for_inference(
        frame,
        torch.device(device),
        task=task,
        robot_type=robot_type,
    )
    autocast = (
        torch.autocast(device_type="cuda")
        if torch.device(device).type == "cuda" and policy.config.use_amp
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        processed_input = preprocessor(model_input)
        relative_chunk = policy.predict_action_chunk(processed_input)
        absolute_chunk = postprocessor(relative_chunk)
    return absolute_chunk.squeeze(0).detach().float().cpu()


def _predict_openpi_chunk(
    *,
    client,
    raw_observation: dict[str, Any],
    state: Any,
    task: str,
    image_size: int,
    image_key: str,
    left_wrist_image_key: str,
    right_wrist_image_key: str,
    state_key: str,
    prompt_key: str,
    timeout_s: float,
):
    """Send one read-only observation to an OpenPI server and validate actions."""
    import numpy as np

    from lerobot.rollout.inference.openpi_websocket import _as_chw_uint8

    def image(name: str):
        return _as_chw_uint8(raw_observation[name], image_size)

    state_array = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_array.size != 14:
        raise ValueError(f"OpenPI state must contain 14 values, got {state_array.size}")
    result = client.infer(
        {
            "images": {
                image_key: image("cam_high"),
                left_wrist_image_key: image("cam_left_wrist"),
                right_wrist_image_key: image("cam_right_wrist"),
            },
            state_key: state_array,
            prompt_key: task,
        },
        timeout_s,
    )
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"OpenPI server must return actions with shape [T, 14], got {actions.shape}")
    if actions.shape[0] == 0 or not np.isfinite(actions).all():
        raise ValueError("OpenPI server returned empty or non-finite actions")
    return actions


def _image_summary(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        image = observation[key]
        summary[key] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min": int(image.min()),
            "max": int(image.max()),
        }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--backend",
        choices=("local", "openpi"),
        default="local",
        help="Run local LeRobot inference or query an OpenPI JAX policy server over WebSocket",
    )
    parser.add_argument("--robot-id", default="bi_piper_read_only")
    parser.add_argument("--task", default="Stack blocks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openpi-server-address", default="localhost:8000")
    parser.add_argument("--openpi-api-key")
    parser.add_argument("--openpi-connect-timeout-s", type=float, default=10.0)
    parser.add_argument("--openpi-request-timeout-s", type=float, default=30.0)
    parser.add_argument("--openpi-image-size", type=int, default=224)
    parser.add_argument("--openpi-image-key", default="cam_high")
    parser.add_argument("--openpi-left-wrist-image-key", default="cam_left_wrist")
    parser.add_argument("--openpi-right-wrist-image-key", default="cam_right_wrist")
    parser.add_argument("--openpi-state-key", default="state")
    parser.add_argument("--openpi-prompt-key", default="prompt")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds; 0 disables the limit")
    parser.add_argument("--num-samples", type=int, default=0, help="0 disables the sample-count limit")
    parser.add_argument(
        "--inference-interval-s",
        type=float,
        default=0.0,
        help="Minimum interval between inference starts; 0 runs as fast as the model permits",
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate configuration without loading hardware"
    )
    add_can_arguments(parser)
    add_camera_arguments(parser)
    args = parser.parse_args(argv)
    validate_can_names(args, parser)
    if (
        not math.isfinite(args.duration)
        or args.duration < 0
        or args.num_samples < 0
        or not math.isfinite(args.inference_interval_s)
        or args.inference_interval_s < 0
    ):
        parser.error(
            "--duration and --inference-interval-s must be finite and non-negative; "
            "--num-samples must be non-negative"
        )
    if args.backend == "openpi":
        args.openpi_server_address = args.openpi_server_address.strip()
        if not args.openpi_server_address:
            parser.error("--openpi-server-address cannot be empty")
        if args.openpi_image_size <= 0:
            parser.error("--openpi-image-size must be positive")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (args.openpi_connect_timeout_s, args.openpi_request_timeout_s)
        ):
            parser.error("OpenPI timeouts must be positive and finite")
    if args.duration == 0 and args.num_samples == 0:
        parser.error("at least one of --duration or --num-samples must be non-zero")
    validate_camera_arguments(args, parser)
    args.log_file = _default_log_path() if args.log_file is None else args.log_file.expanduser()
    if not args.log_file.is_absolute():
        args.log_file = (REPO_ROOT / args.log_file).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = validate_checkpoint(args.checkpoint)
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
    settings = camera_settings(args)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "read_only_no_action",
                    "backend": args.backend,
                    "checkpoint": str(contract.path),
                    "robot_type": "bi_piper",
                    "cameras": settings,
                    "log_file": str(args.log_file),
                    "openpi_server_address": args.openpi_server_address if args.backend == "openpi" else None,
                },
                indent=2,
            )
        )
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_can_interfaces(args.left_can, args.right_can)
    policy = preprocessor = postprocessor = None
    openpi_client = None
    if args.backend == "local":
        logger.info("Loading Pi0.5 policy from %s", contract.path)
        policy, preprocessor, postprocessor = _load_policy(contract.path, args.device)
    else:
        from lerobot.rollout.inference.openpi_websocket import _OpenPIClient

        host, port = _parse_openpi_address(args.openpi_server_address)
        openpi_client = _OpenPIClient(
            host,
            port,
            args.openpi_api_key,
            args.openpi_connect_timeout_s,
        )
        logger.info("Connected to OpenPI policy server at %s", args.openpi_server_address)
    robot = None
    sample_index = 0
    robot_connected = False
    try:
        robot = _make_read_only_robot(args, make_realsense_configs(args))
        dataset_features = _dataset_features(robot)
        state_names = list(dataset_features["observation.state"]["names"])
        action_names = list(robot.action_features)
        checkpoint_action_names = list(contract.config.get("action_feature_names") or action_names)
        if len(state_names) != 14 or len(action_names) != 14:
            raise ValueError(
                f"BiPiper must expose 14 state/action values, got {len(state_names)}/{len(action_names)}"
            )

        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        robot.connect()
        robot_connected = True
        logger.info("Feedback-only hardware connected; no motion or torque command was sent")
        started_at = time.perf_counter()
        with args.log_file.open("a", encoding="utf-8", buffering=1) as stream:
            while True:
                if args.duration and time.perf_counter() - started_at >= args.duration:
                    break
                if args.num_samples and sample_index >= args.num_samples:
                    break
                cycle_started = time.perf_counter()
                observation = robot.get_observation()
                inference_started = time.perf_counter()
                state = [float(observation[name]) for name in state_names]
                if args.backend == "local":
                    chunk = _predict_chunk(
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        raw_observation=observation,
                        dataset_features=dataset_features,
                        device=args.device,
                        task=args.task,
                        robot_type=robot.robot_type,
                    )
                    chunk_values = chunk.numpy()
                else:
                    chunk_values = _predict_openpi_chunk(
                        client=openpi_client,
                        raw_observation=observation,
                        state=state,
                        task=args.task,
                        image_size=args.openpi_image_size,
                        image_key=args.openpi_image_key,
                        left_wrist_image_key=args.openpi_left_wrist_image_key,
                        right_wrist_image_key=args.openpi_right_wrist_image_key,
                        state_key=args.openpi_state_key,
                        prompt_key=args.openpi_prompt_key,
                        timeout_s=args.openpi_request_timeout_s,
                    )
                inference_s = time.perf_counter() - inference_started
                record = {
                    "timestamp_ns": time.time_ns(),
                    "sample_index": sample_index,
                    "task": args.task,
                    "robot_type": robot.robot_type,
                    "backend": args.backend,
                    "state_names": state_names,
                    "state": state,
                    "checkpoint_action_names": checkpoint_action_names,
                    "robot_action_names": action_names,
                    "predicted_action": chunk_values[0].tolist(),
                    "predicted_action_chunk": chunk_values.tolist(),
                    "chunk_shape": list(chunk_values.shape),
                    "inference_latency_s": inference_s,
                    "images": _image_summary(observation),
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                logger.info(
                    "sample=%d inference=%.3fs first_action=%s",
                    sample_index,
                    inference_s,
                    [round(value, 4) for value in record["predicted_action"]],
                )
                sample_index += 1
                remaining = args.inference_interval_s - (time.perf_counter() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        logger.info("Interrupted after %d samples", sample_index)
    finally:
        try:
            if robot is not None and robot_connected:
                robot.disconnect()
        finally:
            if openpi_client is not None:
                openpi_client.close()
    logger.info("Wrote %d prediction records to %s", sample_index, args.log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
