"""Low-overhead JSONL telemetry for hardware rollouts."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Any

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    """Convert tensors and NumPy values without importing either package."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


class JsonlTelemetryWriter:
    """Best-effort asynchronous JSONL writer.

    The control loop only performs a bounded queue operation.  A full queue
    drops telemetry rather than blocking robot control, and writer failures are
    logged without masking the hardware exception that caused shutdown.
    """

    def __init__(self, path: str | Path, max_queue_size: int = 256) -> None:
        self.path = Path(path).expanduser()
        self._queue: Queue[dict[str, Any] | None] = Queue(maxsize=max_queue_size)
        self._stop = Event()
        self._thread = Thread(target=self._run, name="rollout-jsonl", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        self._started = True

    def submit(self, record: Mapping[str, Any]) -> bool:
        if not self._started or self._stop.is_set():
            return False
        try:
            self._queue.put_nowait(_jsonable(dict(record)))
        except Full:
            logger.warning("Rollout telemetry queue is full; dropping a record")
            return False
        return True

    def close(self) -> None:
        if not self._started:
            return
        self._stop.set()
        try:
            self._queue.put(None, timeout=1.0)
        except Full:
            logger.warning("Could not signal rollout telemetry writer before timeout")
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.warning("Rollout telemetry writer did not stop within timeout")
        self._started = False

    def _run(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                while True:
                    record = self._queue.get()
                    if record is None:
                        break
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
        except Exception:
            logger.exception("Rollout telemetry writer failed for %s", self.path)
