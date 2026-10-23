from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.bi_piper import rollout_pi05_async_rtc as launcher
from examples.bi_piper.infer_pi05_no_action import parse_args as parse_read_only_args

CAMERA_ARGS = [
    "--head-camera=HEAD",
    "--left-wrist-camera=LEFT",
    "--right-wrist-camera=RIGHT",
]
CAMERA_RESOLUTIONS = {
    "cam_high": (640, 480),
    "cam_left_wrist": (640, 480),
    "cam_right_wrist": (640, 480),
}


@pytest.mark.parametrize(
    "argument",
    ["--duration=nan", "--duration=inf", "--inference-interval-s=nan"],
)
def test_read_only_cli_rejects_nonfinite_timing(argument: str) -> None:
    with pytest.raises(SystemExit):
        parse_read_only_args([*CAMERA_ARGS, argument])


@pytest.mark.parametrize(
    "argument",
    ["--exposure=0", "--head-exposure=-1", "--gain=-1", "--right-wrist-gain=-1"],
)
def test_camera_cli_rejects_invalid_manual_sensor_values(argument: str) -> None:
    with pytest.raises(SystemExit):
        launcher.parse_args([*CAMERA_ARGS, argument, "--dry-run"])


def test_build_rtc_rollout_command_uses_local_policy_device() -> None:
    args = launcher.parse_args([*CAMERA_ARGS, "--backend=rtc", "--device=cuda:1", "--dry-run"])

    command = launcher.build_rollout_command(args, Path("/client/checkpoint"), CAMERA_RESOLUTIONS)

    assert "--inference.type=rtc" in command
    assert "--inference.rtc.execution_horizon=25" in command
    assert "--inference.queue_threshold=45" in command
    assert "--device=cuda:1" in command
    assert not any(argument.startswith("--inference.server_address=") for argument in command)
    assert any(argument.startswith("--inference_log_path=") for argument in command)


def test_generated_rtc_command_decodes_as_rollout_config(monkeypatch) -> None:
    import draccus

    from lerobot.configs import PreTrainedConfig, parser
    from lerobot.rollout import RolloutConfig
    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.scripts import lerobot_rollout as _rollout_registrations  # noqa: F401

    args = launcher.parse_args(
        [
            *CAMERA_ARGS,
            "--backend=rtc",
            "--device=cuda:1",
            "--execution-horizon=12",
            "--queue-threshold=24",
            "--max-guidance-weight=8",
            "--prefix-attention-schedule=LINEAR",
            "--dry-run",
        ]
    )
    command = launcher.build_rollout_command(args, Path("/client/checkpoint"), CAMERA_RESOLUTIONS)
    cli_args = command[3:]
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", *cli_args])
    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: SimpleNamespace(device="cuda:1")),
    )

    config = draccus.parse(
        RolloutConfig,
        args=parser.filter_path_args(RolloutConfig.__get_path_fields__(), cli_args),
    )

    assert config.robot.type == "bi_piperx"
    assert set(config.robot.cameras) == set(CAMERA_RESOLUTIONS)
    assert isinstance(config.inference, RTCInferenceConfig)
    assert config.inference.rtc.execution_horizon == 12
    assert config.inference.queue_threshold == 24
    assert config.inference.rtc.max_guidance_weight == 8
    assert config.inference.rtc.prefix_attention_schedule.value == "LINEAR"
    assert config.device == "cuda:1"


def test_build_remote_rollout_command_uses_server_policy_contract() -> None:
    args = launcher.parse_args(
        [
            *CAMERA_ARGS,
            "--backend=remote",
            "--server-address=policy.example:9000",
            "--server-policy-path=/srv/policies/piper",
            "--server-policy-device=cuda:2",
            "--actions-per-chunk=40",
            "--chunk-size-threshold=0.25",
            "--aggregate-fn-name=latest_only",
            "--connect-timeout-s=7",
            "--setup-timeout-s=500",
            "--rpc-timeout-s=20",
            "--first-action-timeout-s=90",
            "--dry-run",
        ]
    )

    command = launcher.build_rollout_command(args, Path("/client/checkpoint"), CAMERA_RESOLUTIONS)

    expected = {
        "--inference.type=remote",
        "--inference.server_address=policy.example:9000",
        "--inference.policy_path=/srv/policies/piper",
        "--inference.policy_device=cuda:2",
        "--inference.actions_per_chunk=40",
        "--inference.chunk_size_threshold=0.25",
        "--inference.aggregate_fn_name=latest_only",
        "--inference.connect_timeout_s=7.0",
        "--inference.setup_timeout_s=500.0",
        "--inference.rpc_timeout_s=20.0",
        "--inference.first_action_timeout_s=90.0",
        "--device=cpu",
    }
    assert expected.issubset(command)
    assert not any(argument.startswith("--inference.rtc.") for argument in command)


def test_real_execution_requires_exact_confirmation_phrase() -> None:
    with pytest.raises(SystemExit):
        launcher.parse_args([*CAMERA_ARGS, "--execute"])

    args = launcher.parse_args(
        [
            *CAMERA_ARGS,
            "--execute",
            f"--confirm-motion={launcher.MOTION_CONFIRMATION}",
        ]
    )

    assert args.execute is True

    with pytest.raises(SystemExit):
        launcher.parse_args(
            [
                *CAMERA_ARGS,
                "--dry-run",
                "--execute",
                f"--confirm-motion={launcher.MOTION_CONFIRMATION}",
            ]
        )


def test_remote_dry_run_never_opens_hardware_or_starts_rollout(monkeypatch, capsys) -> None:
    contract = SimpleNamespace(
        path=Path("/client/checkpoint"),
        config={"chunk_size": 50},
        camera_resolutions=CAMERA_RESOLUTIONS,
    )
    monkeypatch.setattr(launcher, "validate_checkpoint", lambda *args, **kwargs: contract)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("dry-run reached a hardware or subprocess boundary")

    monkeypatch.setattr(launcher, "validate_can_interfaces", unexpected_call)
    monkeypatch.setattr(launcher.subprocess, "run", unexpected_call)

    result = launcher.main([*CAMERA_ARGS, "--backend=remote", "--dry-run"])

    assert result == 0
    assert "no hardware opened" in capsys.readouterr().out


def test_generated_remote_command_decodes_as_rollout_config(monkeypatch) -> None:
    import draccus

    from lerobot.configs import PreTrainedConfig, parser
    from lerobot.rollout import RolloutConfig
    from lerobot.rollout.inference import RemoteInferenceConfig
    from lerobot.scripts import lerobot_rollout as _rollout_registrations  # noqa: F401

    args = launcher.parse_args(
        [
            *CAMERA_ARGS,
            "--backend=remote",
            "--server-address=policy.example:9000",
            "--server-policy-path=/srv/policies/piper",
            "--server-policy-device=cuda:2",
            "--actions-per-chunk=40",
            "--first-action-timeout-s=90",
            "--dry-run",
        ]
    )
    command = launcher.build_rollout_command(args, Path("/client/checkpoint"), CAMERA_RESOLUTIONS)
    cli_args = command[3:]
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", *cli_args])
    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: SimpleNamespace(device="cpu")),
    )

    config = draccus.parse(
        RolloutConfig,
        args=parser.filter_path_args(RolloutConfig.__get_path_fields__(), cli_args),
    )

    assert config.robot.type == "bi_piperx"
    assert set(config.robot.cameras) == set(CAMERA_RESOLUTIONS)
    assert isinstance(config.inference, RemoteInferenceConfig)
    assert config.inference.server_address == "policy.example:9000"
    assert config.inference.policy_path == "/srv/policies/piper"
    assert config.inference.policy_device == "cuda:2"
    assert config.inference.actions_per_chunk == 40
    assert config.inference.first_action_timeout_s == 90
