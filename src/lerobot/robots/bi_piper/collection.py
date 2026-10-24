"""Two-pass replay and passive leader/follower HDF5 collection workflows."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config_bi_piper import PIPER_MOTOR_NAMES
from .hdf5_recording import CollectionSample, HDF5EpisodeWriter
from .hdf5_replay import VECTOR_ORDER, HDF5Trajectory
from .replay_collection import ReplayPlan, ReplaySource

POSITION_KEYS = tuple(f"{side}_{motor}.pos" for side in ("left", "right") for motor in PIPER_MOTOR_NAMES)
JOINT_COLUMNS = tuple(index for index in range(14) if index not in (6, 13))
logger = logging.getLogger(__name__)


def validate_position(position: np.ndarray, gripper_max_m: float) -> np.ndarray:
    """Check physical units before any motion is authorized."""
    values = np.asarray(position, dtype=np.float64)
    if values.shape != (14,):
        raise ValueError("A physical dual-arm position must contain exactly 14 values")
    HDF5Trajectory(values[None], "meters", "radians", VECTOR_ORDER, Path("collection_target"))
    if np.any(values[[6, 13]] > gripper_max_m):
        raise ValueError("A gripper target exceeds the configured physical jaw range")
    return values


class BiPiperCollectionSession:
    """Use physical feedback without policy-specific gripper effort masking."""

    def __init__(
        self,
        *,
        left_can: str = "can_left",
        right_can: str = "can_right",
        gripper_max_m: float = 0.1,
        speed_percent: int = 30,
        feedback_timeout_s: float = 0.5,
        left_installation_pos: int = 1,
        right_installation_pos: int = 1,
        control_hz: float = 30,
        move_joint_velocity: float = 0.5,
        move_gripper_velocity: float = 0.05,
        settle_timeout_s: float = 5,
        joint_tolerance: float = 0.08,
        gripper_tolerance_m: float = 0.005,
    ) -> None:
        from .bi_piper import BiPiper
        from .config_bi_piper import BiPiperConfig, PiperArmConfig

        parameters = (
            control_hz,
            move_joint_velocity,
            move_gripper_velocity,
            settle_timeout_s,
            joint_tolerance,
            gripper_tolerance_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in parameters):
            raise ValueError("Movement rates, timeouts and tolerances must be positive and finite")
        if not math.isfinite(gripper_max_m) or not 0 < gripper_max_m <= 0.1:
            raise ValueError("Piper jaw range must be in (0, 0.1] meters")
        self.gripper_max_m = gripper_max_m
        self.control_hz = control_hz
        self.move_joint_velocity = move_joint_velocity
        self.move_gripper_velocity = move_gripper_velocity
        self.settle_timeout_s = settle_timeout_s
        self.joint_tolerance = joint_tolerance
        self.gripper_tolerance_m = gripper_tolerance_m
        arms = {
            side: PiperArmConfig(
                can_name=can_name,
                gripper_unit="meters",
                gripper_range_m=(0, gripper_max_m),
                max_relative_target=None,
                disable_on_disconnect=False,
                speed_percent=speed_percent,
                feedback_timeout_s=feedback_timeout_s,
                installation_pos=installation,
            )
            for side, can_name, installation in (
                ("left", left_can, left_installation_pos),
                ("right", right_can, right_installation_pos),
            )
        }
        self.robot = BiPiper(
            BiPiperConfig(
                id="bi_piper_collection",
                left_arm_config=arms["left"],
                right_arm_config=arms["right"],
            )
        )

    def connect(self) -> None:
        self.robot.connect()

    def read_position(self) -> np.ndarray:
        # The ordinary policy observation intentionally maps a loaded gripper to
        # zero. Dataset feedback must retain its actual encoder opening instead.
        values = {
            f"{side}_{motor}.pos": value
            for side in ("left", "right")
            for motor, value in self.robot._read_arm_positions(side, for_policy=False).items()
        }
        return np.asarray([values[key] for key in POSITION_KEYS], dtype=np.float64)

    def read_endposes(self) -> dict[str, np.ndarray]:
        from .shared_can import read_endpose

        return {
            side: read_endpose(client, self.robot._arm_configs[side].feedback_timeout_s)
            for side, client in self.robot._clients.items()
        }

    def send_position(self, position: np.ndarray) -> np.ndarray:
        target = validate_position(position, self.gripper_max_m)
        action = dict(zip(POSITION_KEYS, target.tolist(), strict=True))
        sent = self.robot.send_action(action)
        if set(sent) != set(action):
            raise RuntimeError("Robot changed collection action keys")
        values = np.asarray([sent[key] for key in POSITION_KEYS])
        if not np.allclose(values, target, rtol=0, atol=1e-9):
            raise RuntimeError("Robot clipped a collection target; refusing to silently alter the trajectory")
        return values

    def move_to(self, target: np.ndarray) -> None:
        target = validate_position(target, self.gripper_max_m)
        present = self.read_position()
        delta = np.abs(target - present)
        duration = max(
            0.5,
            float(delta[list(JOINT_COLUMNS)].max()) / self.move_joint_velocity,
            float(delta[[6, 13]].max()) / self.move_gripper_velocity,
        )
        steps = max(1, math.ceil(duration * self.control_hz))
        started = time.monotonic()
        for step in range(1, steps + 1):
            time.sleep(max(0, started + duration * step / steps - time.monotonic()))
            self.send_position(present + (target - present) * step / steps)
        deadline = time.monotonic() + self.settle_timeout_s
        while time.monotonic() < deadline:
            measured = self.read_position()
            error = np.abs(measured - target)
            if (
                error[list(JOINT_COLUMNS)].max() <= self.joint_tolerance
                and error[[6, 13]].max() <= self.gripper_tolerance_m
            ):
                return
            self.send_position(target)
            time.sleep(1 / self.control_hz)
        raise TimeoutError("Both Piper arms and grippers did not reach the requested collection pose")

    def stop(self) -> None:
        try:
            self.robot.emergency_stop()
        finally:
            self.robot.preserve_torque_for_manual_recovery()

    def disconnect(self) -> None:
        self.robot.disconnect()


def _wait_for_command(started: float, target_time: float, max_lateness_s: float, clock: Any) -> None:
    remaining = started + target_time - clock.monotonic()
    if remaining < -max_lateness_s:
        raise TimeoutError(
            "Collection missed its command deadline; stopping instead of skipping source frames"
        )
    if remaining > 0:
        clock.sleep(remaining)


def _run_replay(
    session: Any,
    plan: ReplayPlan,
    *,
    writer: HDF5EpisodeWriter | None = None,
    cameras: Any | None = None,
    max_lateness_s: float = 0.5,
    clock: Any = time,
) -> None:
    source_by_command = {
        int(command): source for source, command in enumerate(plan.source_frame_command_indices)
    }
    started = clock.monotonic()
    for index, target in enumerate(plan.positions):
        _wait_for_command(started, float(plan.command_times_s[index]), max_lateness_s, clock)
        sent = session.send_position(target)
        if writer is not None and index in source_by_command:
            position = session.read_position()
            timestamp_ns = clock.monotonic_ns()
            endposes = session.read_endposes()
            images = cameras.read() if cameras is not None else {}
            writer.append(
                CollectionSample(
                    position=position,
                    timestamp_ns=timestamp_ns,
                    action=sent,
                    source_frame_index=source_by_command[index],
                    endposes=endposes,
                    cameras=images,
                )
            )


def run_two_pass_episode(
    session: Any,
    source: ReplaySource,
    first_plan: ReplayPlan,
    second_plan: ReplayPlan,
    output: Path,
    *,
    initial_position: np.ndarray,
    gripper_max_m: float = 0.1,
    cameras: Any | None = None,
    prompt: Callable[[str], Any] = input,
    max_lateness_s: float = 0.5,
    metadata: dict[str, Any] | None = None,
    clock: Any = time,
) -> Path:
    """Replay once, reset, wait, replay+record, reset and atomically publish."""
    if output.exists():
        raise FileExistsError(output)
    initial_position = validate_position(initial_position, gripper_max_m)
    if not math.isfinite(max_lateness_s) or max_lateness_s <= 0:
        raise ValueError("Maximum command lateness must be positive and finite")
    info = dict(metadata or {})
    info.update(
        workflow="bi_piper_two_pass_replay_collect",
        backend="hardware",
        sim=False,
        source_episode=str(source.path),
        source_frame_count=len(source.positions),
        frame_alignment="one_real_frame_per_source_frame",
        first_pass_recorded=False,
        first_pass_speed_scale=first_plan.speed_scale,
        second_pass_speed_scale=second_plan.speed_scale,
        capture_target_fps=source.source_hz * second_plan.speed_scale,
        post_replay_initial_state=initial_position,
        post_replay_initial_state_units="radians,meters",
        recording_scope="both_physical_follower_feedback",
        action_source="sent_replay_target",
    )
    writer = None
    try:
        session.move_to(first_plan.positions[0])
        _run_replay(session, first_plan, max_lateness_s=max_lateness_s, clock=clock)
        session.move_to(initial_position)
        prompt("第一遍结束，双臂已回初态。复位物体后按 Enter 开始第二遍采集：")
        session.move_to(second_plan.positions[0])
        writer = HDF5EpisodeWriter(
            output,
            fps=source.source_hz,
            gripper_max_m=gripper_max_m,
            expected_frames=len(source.positions),
            metadata=info,
        )
        _run_replay(
            session, second_plan, writer=writer, cameras=cameras, max_lateness_s=max_lateness_s, clock=clock
        )
        session.move_to(initial_position)
        return writer.finalize()
    except BaseException:
        try:
            if writer is not None:
                writer.abort()
        except Exception:
            logger.exception("Aborting the incomplete recording failed")
        finally:
            try:
                session.stop()
            except Exception:
                logger.exception("Stopping Piper after collection failure also failed")
        raise


def collect_shared_can_episode(
    session: Any,
    output: Path,
    *,
    duration_s: float = 60,
    fps: float = 25,
    gripper_max_m: float = 0.1,
    cameras: Any | None = None,
    max_lateness_s: float = 0.5,
    metadata: dict[str, Any] | None = None,
    clock: Any = time,
) -> Path:
    """Record hardware 1:1 following without emitting any motor command."""
    if any(not math.isfinite(value) or value <= 0 for value in (duration_s, fps, max_lateness_s)):
        raise ValueError("Episode duration, sampling FPS and maximum lateness must be positive and finite")
    frames = max(1, math.ceil(duration_s * fps))
    info = dict(metadata or {})
    info.update(
        workflow="bi_piper_shared_can_teleop",
        backend="hardware",
        sim=False,
        action_source="leader_can_control",
        following="firmware_direct_1_to_1",
        recording_scope="both_physical_follower_feedback",
        capture_target_fps=fps,
    )
    writer = HDF5EpisodeWriter(
        output, fps=fps, gripper_max_m=gripper_max_m, expected_frames=frames, metadata=info
    )
    try:
        started = clock.monotonic()
        for index in range(frames):
            _wait_for_command(started, index / fps, max_lateness_s, clock)
            sample = session.read_sample()
            timestamp_ns = clock.monotonic_ns()
            images = cameras.read() if cameras is not None else {}
            writer.append(
                CollectionSample(
                    position=sample.follower_position,
                    action=sample.leader_action,
                    timestamp_ns=timestamp_ns,
                    endposes=sample.endposes,
                    cameras=images,
                )
            )
        return writer.finalize()
    except BaseException:
        # The passive observer must not send an emergency stop or disable into
        # the shared leader/follower bus, including during failure/teardown.
        writer.abort()
        raise
