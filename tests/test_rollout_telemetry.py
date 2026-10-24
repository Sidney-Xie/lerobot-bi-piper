from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from lerobot.rollout.telemetry import JsonlTelemetryWriter


def test_jsonl_telemetry_writer_serializes_numpy_values(tmp_path):
    path = tmp_path / "closed_loop.jsonl"
    writer = JsonlTelemetryWriter(path)
    writer.start()
    assert writer.submit(
        {
            "timestamp_ns": np.int64(123),
            "state": {"left_j1.pos": np.float32(0.25)},
            "action": np.asarray([1.0, 2.0]),
        }
    )
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "timestamp_ns": 123,
            "state": {"left_j1.pos": 0.25},
            "action": [1.0, 2.0],
        }
    ]


def test_control_cycle_log_includes_inference_telemetry(tmp_path):
    from lerobot.rollout import BaseStrategy, BaseStrategyConfig

    path = tmp_path / "closed_loop.jsonl"
    writer = JsonlTelemetryWriter(path)
    writer.start()
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._telemetry = writer
    strategy._telemetry_state_names = ["joint.pos"]
    strategy._engine = SimpleNamespace(
        get_inference_telemetry=lambda: {
            "inference_started_ns": 100,
            "inference_finished_ns": 120,
            "inference_latency_ms": 0.02,
            "queue_size": 24,
            "chunk_id": 3,
        }
    )
    strategy._runtime_context = SimpleNamespace(
        last_processed_action={"joint.pos": 0.5},
        last_sent_action={"joint.pos": 0.5},
        last_send_success=True,
        last_send_error=None,
    )

    strategy._log_control_cycle(
        step=7,
        obs_raw={"joint.pos": 0.25},
        action_dict={"joint.pos": 0.5},
        control_dt=0.001,
    )
    writer.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["inference_started_ns"] == 100
    assert record["inference_finished_ns"] == 120
    assert record["inference_latency_ms"] == 0.02
    assert record["queue_size"] == 24
    assert record["chunk_id"] == 3
