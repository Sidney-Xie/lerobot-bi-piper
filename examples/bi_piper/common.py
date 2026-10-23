"""Shared, side-effect-free validation for the BiPiper deployment examples."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path("outputs/pi05_robotwin_piper_x_sft_step40000")
DEFAULT_TRAJECTORY = Path("outputs/stack_blocks_two_replay/episode_0000000.hdf5")
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_DIM = 14
GRIPPER_INDICES = (6, 13)
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
CAN_ERROR_COUNTERS = ("re-started", "bus-errors", "arbit-lost", "error-warn", "error-pass", "bus-off")


@dataclass(frozen=True)
class CheckpointContract:
    path: Path
    config: dict[str, Any]
    camera_resolutions: dict[str, tuple[int, int]]


def resolve_repo_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _camera_resolution(shape: object, feature_name: str) -> tuple[int, int]:
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
    ):
        raise ValueError(f"{feature_name} must have a positive RGB shape, got {shape}")
    first, second, third = shape
    if first == 3 and third != 3:
        return third, second
    if third == 3 and first != 3:
        return second, first
    raise ValueError(f"{feature_name} must be unambiguous RGB CHW or HWC, got {shape}")


def _find_processor_step(document: dict[str, Any], name: str) -> dict[str, Any] | None:
    steps = document.get("steps", [])
    if not isinstance(steps, list):
        return None
    return next(
        (step for step in steps if isinstance(step, dict) and step.get("registry_name") == name),
        None,
    )


def validate_checkpoint(
    path: Path,
    *,
    execution_horizon: int | None = None,
    queue_threshold: int | None = None,
) -> CheckpointContract:
    """Validate the Pi0.5 feature and processor contract without loading weights."""

    checkpoint = resolve_repo_path(path)
    missing = [name for name in CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Checkpoint {checkpoint} is incomplete; missing: {', '.join(missing)}")

    config = _load_json(checkpoint / "config.json")
    preprocessor = _load_json(checkpoint / "policy_preprocessor.json")
    postprocessor = _load_json(checkpoint / "policy_postprocessor.json")
    errors: list[str] = []

    if config.get("type") != "pi05":
        errors.append(f"policy type must be pi05, got {config.get('type')}")

    input_features = config.get("input_features")
    output_features = config.get("output_features")
    if not isinstance(input_features, dict):
        input_features = {}
        errors.append("input_features must be a JSON object")
    if not isinstance(output_features, dict):
        output_features = {}
        errors.append("output_features must be a JSON object")

    state = input_features.get("observation.state")
    if not isinstance(state, dict) or state.get("shape") != [ACTION_DIM] or state.get("type") != "STATE":
        errors.append(f"observation.state must be STATE with shape [{ACTION_DIM}]")
    action = output_features.get("action")
    if not isinstance(action, dict) or action.get("shape") != [ACTION_DIM] or action.get("type") != "ACTION":
        errors.append(f"action must be ACTION with shape [{ACTION_DIM}]")

    camera_resolutions: dict[str, tuple[int, int]] = {}
    for camera_key in CAMERA_KEYS:
        feature_name = f"observation.images.{camera_key}"
        feature = input_features.get(feature_name)
        if not isinstance(feature, dict) or feature.get("type") != "VISUAL":
            errors.append(f"checkpoint is missing VISUAL feature {feature_name}")
            continue
        try:
            camera_resolutions[camera_key] = _camera_resolution(feature.get("shape"), feature_name)
        except ValueError as exc:
            errors.append(str(exc))
    unexpected_cameras = sorted(
        key
        for key in input_features
        if key.startswith("observation.images.")
        and key.removeprefix("observation.images.") not in CAMERA_KEYS
    )
    if unexpected_cameras:
        errors.append(f"unsupported checkpoint camera features: {unexpected_cameras}")

    chunk_size = config.get("chunk_size")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        errors.append(f"chunk_size must be a positive integer, got {chunk_size}")
        chunk_size = 0
    if execution_horizon is not None and not 1 <= execution_horizon <= chunk_size:
        errors.append(f"execution_horizon must be in [1, {chunk_size}], got {execution_horizon}")
    if (
        queue_threshold is not None
        and execution_horizon is not None
        and not execution_horizon <= queue_threshold < chunk_size
    ):
        errors.append(
            "queue_threshold must be at least execution_horizon and less than "
            f"chunk_size={chunk_size}, got {queue_threshold}"
        )

    relative = config.get("use_relative_actions")
    if not isinstance(relative, bool):
        errors.append(f"use_relative_actions must be boolean, got {relative}")
    relative_step = _find_processor_step(preprocessor, "relative_actions_processor")
    absolute_step = _find_processor_step(postprocessor, "absolute_actions_processor")
    if relative_step is None:
        errors.append("policy preprocessor is missing relative_actions_processor")
    elif relative_step.get("config", {}).get("enabled") is not relative:
        errors.append("relative action preprocessor does not match config.json")
    if relative and absolute_step is None:
        errors.append("relative policy is missing absolute_actions_processor")
    elif absolute_step is not None and absolute_step.get("config", {}).get("enabled") is not relative:
        errors.append("absolute action postprocessor does not match config.json")

    required_norm_keys = {"VISUAL", "STATE", "ACTION"}
    norm_map = config.get("normalization_mapping")
    if not isinstance(norm_map, dict) or set(norm_map) != required_norm_keys:
        errors.append(f"normalization_mapping must define exactly {sorted(required_norm_keys)}")
    else:
        if norm_map["VISUAL"] != "IDENTITY":
            errors.append("VISUAL normalization must be IDENTITY")
        if norm_map["STATE"] not in {"MEAN_STD", "QUANTILES"}:
            errors.append(f"unsupported STATE normalization {norm_map['STATE']}")
        if norm_map["ACTION"] not in {"MEAN_STD", "QUANTILES"}:
            errors.append(f"unsupported ACTION normalization {norm_map['ACTION']}")

    if errors:
        raise ValueError("Incompatible BiPiper Pi0.5 checkpoint:\n- " + "\n- ".join(errors))
    return CheckpointContract(checkpoint, config, camera_resolutions)


def add_can_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--left-can", default="can_left")
    parser.add_argument("--right-can", default="can_right")
    parser.add_argument("--left-installation-pos", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--right-installation-pos", type=int, choices=(0, 1, 2, 3), default=1)


def add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--head-camera", required=True, help="Serial number for cam_high")
    parser.add_argument("--left-wrist-camera", required=True, help="Serial number for cam_left_wrist")
    parser.add_argument("--right-wrist-camera", required=True, help="Serial number for cam_right_wrist")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-s", type=int, default=2)
    parser.add_argument("--exposure", type=int, help="Manual exposure applied to all cameras")
    parser.add_argument("--gain", type=int, help="Manual gain applied to all cameras")
    for role in ("head", "left-wrist", "right-wrist"):
        dest = role.replace("-", "_")
        parser.add_argument(
            f"--{role}-exposure",
            dest=f"{dest}_exposure",
            type=int,
            help=f"Override --exposure for the {role} camera",
        )
        parser.add_argument(
            f"--{role}-gain",
            dest=f"{dest}_gain",
            type=int,
            help=f"Override --gain for the {role} camera",
        )


def validate_can_names(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.left_can = args.left_can.strip()
    args.right_can = args.right_can.strip()
    if not args.left_can or not args.right_can or args.left_can == args.right_can:
        parser.error("--left-can and --right-can must be non-empty and distinct")


def camera_serials(args: argparse.Namespace) -> tuple[str, str, str]:
    serials = tuple(
        value.strip() for value in (args.head_camera, args.left_wrist_camera, args.right_wrist_camera)
    )
    if any(not serial for serial in serials):
        raise ValueError("All three RealSense serial numbers must be non-empty")
    if len(set(serials)) != 3:
        raise ValueError("Each camera role must use a distinct RealSense serial number")
    return serials  # type: ignore[return-value]


def camera_settings(
    args: argparse.Namespace,
    resolutions: dict[str, tuple[int, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    serials = camera_serials(args)
    role_prefixes = ("head", "left_wrist", "right_wrist")
    result: dict[str, dict[str, Any]] = {}
    for key, serial, role in zip(CAMERA_KEYS, serials, role_prefixes, strict=True):
        width, height = (
            resolutions[key] if resolutions is not None else (args.camera_width, args.camera_height)
        )
        exposure = getattr(args, f"{role}_exposure")
        gain = getattr(args, f"{role}_gain")
        result[key] = {
            "type": "intelrealsense",
            "serial_number_or_name": serial,
            "width": width,
            "height": height,
            "fps": args.camera_fps,
            "use_depth": False,
            "warmup_s": args.camera_warmup_s,
            "exposure": args.exposure if exposure is None else exposure,
            "gain": args.gain if gain is None else gain,
        }
    return result


def validate_camera_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    dimensions = {
        "--camera-width": args.camera_width,
        "--camera-height": args.camera_height,
        "--camera-fps": args.camera_fps,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions.values()
    ):
        parser.error(f"camera width, height, and FPS must be positive finite integers: {dimensions}")
    if (
        not isinstance(args.camera_warmup_s, int)
        or isinstance(args.camera_warmup_s, bool)
        or args.camera_warmup_s < 0
    ):
        parser.error("--camera-warmup-s must be a non-negative finite integer")

    for role in (None, "head", "left_wrist", "right_wrist"):
        prefix = "" if role is None else f"{role}_"
        option_prefix = "" if role is None else f"{role.replace('_', '-')}-"
        exposure = getattr(args, f"{prefix}exposure")
        gain = getattr(args, f"{prefix}gain")
        if exposure is not None and (
            not isinstance(exposure, int) or isinstance(exposure, bool) or exposure <= 0
        ):
            parser.error(f"--{option_prefix}exposure must be a positive finite integer")
        if gain is not None and (not isinstance(gain, int) or isinstance(gain, bool) or gain < 0):
            parser.error(f"--{option_prefix}gain must be a non-negative finite integer")


def make_realsense_configs(
    args: argparse.Namespace,
    resolutions: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Create typed camera configs lazily so dry-runs need no RealSense SDK."""

    from lerobot.cameras.realsense import RealSenseCameraConfig

    settings = camera_settings(args, resolutions)
    return {
        key: RealSenseCameraConfig(**{name: value for name, value in values.items() if name != "type"})
        for key, values in settings.items()
    }


def validate_can_interfaces(left_can: str, right_can: str) -> None:
    """Fail before enabling either arm when SocketCAN is absent or unhealthy."""

    if left_can == right_can:
        raise ValueError("The two arms must use distinct SocketCAN interfaces")
    for can_name in (left_can, right_can):
        try:
            socket.if_nametoindex(can_name)
        except OSError as exc:
            raise RuntimeError(f"SocketCAN interface not found: {can_name}") from exc
        sysfs = Path("/sys/class/net") / can_name
        try:
            interface_type = int((sysfs / "type").read_text(encoding="utf-8").strip())
            state = (sysfs / "operstate").read_text(encoding="utf-8").strip()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not inspect SocketCAN interface {can_name}: {exc}") from exc
        if interface_type != 280:
            raise RuntimeError(f"{can_name} is not SocketCAN (ARPHRD type={interface_type})")
        if state not in {"up", "unknown"}:
            raise RuntimeError(f"SocketCAN interface {can_name} is not up (operstate={state})")

    ip_command = shutil.which("ip")
    if ip_command is None:
        raise RuntimeError("The `ip` command is required for SocketCAN health checks")
    snapshots: dict[str, dict[str, int]] = {}
    for can_name in (left_can, right_can):
        snapshots[can_name] = _read_can_health(ip_command, can_name)
    time.sleep(0.25)
    for can_name in (left_can, right_can):
        latest = _read_can_health(ip_command, can_name)
        increases = {
            key: latest[key] - snapshots[can_name][key]
            for key in CAN_ERROR_COUNTERS
            if latest[key] > snapshots[can_name][key]
        }
        if increases:
            raise RuntimeError(f"{can_name} accumulated CAN errors during preflight: {increases}")


def _read_can_health(ip_command: str, can_name: str) -> dict[str, int]:
    result = subprocess.run(
        [ip_command, "-details", "-statistics", "link", "show", "dev", can_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect {can_name}: {result.stderr.strip()}")
    state_match = re.search(r"can state ([A-Z-]+)", result.stdout)
    if state_match is None or state_match.group(1) != "ERROR-ACTIVE":
        state = state_match.group(1) if state_match else "UNKNOWN"
        raise RuntimeError(f"SocketCAN controller {can_name} is unhealthy (state={state})")
    labels = r"\s+".join(re.escape(name) for name in CAN_ERROR_COUNTERS)
    values = r"\s+".join(r"(\d+)" for _ in CAN_ERROR_COUNTERS)
    counters = re.search(labels + r"\s+" + values, result.stdout)
    if counters is None:
        raise RuntimeError(f"Could not parse SocketCAN error counters for {can_name}")
    return dict(zip(CAN_ERROR_COUNTERS, (int(value) for value in counters.groups()), strict=True))


def validate_positive_finite(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite, got {value}")
