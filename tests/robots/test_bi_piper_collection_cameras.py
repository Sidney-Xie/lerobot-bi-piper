"""Camera collection tests use synthetic framesets and never open devices."""

from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.bi_piper import collection_cameras
from lerobot.robots.bi_piper.collection_cameras import RealSenseCollectionCameras
from lerobot.robots.bi_piper.hdf5_recording import CameraSample


def sample(number: int, *, age_ns: int = 0) -> CameraSample:
    return CameraSample(
        rgb=np.full((2, 3, 3), number, dtype=np.uint8),
        depth_mm=np.full((2, 3), number, dtype=np.float64),
        timestamp_ns=time.monotonic_ns() - age_ns,
        frame_number=number,
        device_timestamp_ms=float(number),
    )


class FakeFrame:
    def __init__(self, number: int, data: np.ndarray) -> None:
        self.number, self.data = number, data

    def get_frame_number(self) -> int:
        return self.number

    def get_data(self) -> np.ndarray:
        return self.data

    def get_timestamp(self) -> float:
        return self.number * 10.0


class FakeFrames:
    def __init__(self, color_number: int, depth_number: int, depth_value: int) -> None:
        self.color = FakeFrame(color_number, np.full((2, 3, 3), color_number, dtype=np.uint8))
        self.depth = FakeFrame(depth_number, np.full((2, 3), depth_value, dtype=np.uint16))
        self.aligned = False

    def get_color_frame(self) -> FakeFrame:
        return self.color

    def get_depth_frame(self) -> FakeFrame:
        assert self.aligned, "Depth must be aligned to the color image before export"
        return self.depth


class FakeCapturePipeline:
    def __init__(self, cameras: RealSenseCollectionCameras, frames: list[FakeFrames | Exception]) -> None:
        self.cameras, self.frames = cameras, deque(frames)

    def try_wait_for_frames(self, timeout_ms: int) -> tuple[bool, FakeFrames | None]:
        assert timeout_ms == 200
        if not self.frames:
            self.cameras._stop.set()
            return False, None
        frame = self.frames.popleft()
        if isinstance(frame, Exception):
            raise frame
        return True, frame


class FakeAlign:
    def __init__(self, stream: str) -> None:
        assert stream == "color"

    def process(self, frames: FakeFrames) -> FakeFrames:
        frames.aligned = True
        return frames


@pytest.fixture
def fake_rs(monkeypatch):
    intrinsics = SimpleNamespace(
        width=3, height=2, fx=2.0, fy=2.5, ppx=1.0, ppy=0.5, model="none", coeffs=[0.0] * 5
    )
    video_profile = SimpleNamespace(
        as_video_stream_profile=lambda: video_profile,
        get_intrinsics=lambda: intrinsics,
        fps=lambda: 30,
        get_extrinsics_to=lambda _: SimpleNamespace(
            # librealsense stores rotations column by column (90-degree yaw).
            rotation=[0, 1, 0, -1, 0, 0, 0, 0, 1],
            translation=[0.01, 0, 0],
        ),
    )
    options = []
    sensor = SimpleNamespace(
        get_info=lambda _: "RGB Camera", set_option=lambda option, value: options.append((option, value))
    )
    device = SimpleNamespace(
        first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.00025),
        query_sensors=lambda: [sensor],
    )
    profile = SimpleNamespace(get_device=lambda: device, get_stream=lambda _: video_profile)

    class FakeConfig:
        def __init__(self) -> None:
            self.serial = None
            self.streams = []

        def enable_device(self, serial: str) -> None:
            self.serial = serial

        def enable_stream(self, *args) -> None:
            self.streams.append(args)

    class FakePipeline:
        def __init__(self) -> None:
            self.config = None
            self.started = False
            self.stop_calls = 0
            self.fail_start = False

        def start(self, config):
            self.config = config
            if self.fail_start:
                raise OSError("second camera cannot start")
            self.started = True
            return profile

        def stop(self) -> None:
            self.stop_calls += 1
            if not self.started:
                raise RuntimeError("stop called on unstarted pipeline")
            self.started = False

    class FakeThread:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.name = kwargs["name"]
            self.ident: int | None = None
            self.started = False
            self.joined = False
            self.fail_start = False
            self.survives_join = False

        def start(self) -> None:
            if self.fail_start:
                raise OSError("capture worker cannot start")
            self.started = True
            self.ident = id(self)

        def join(self, timeout: float) -> None:
            assert self.ident is not None, "Cannot join a worker that never started"
            assert timeout == 1.0
            self.joined = True

        def is_alive(self) -> bool:
            return self.started and (not self.joined or self.survives_join)

    pipelines = []

    def new_pipeline():
        pipeline = FakePipeline()
        pipelines.append(pipeline)
        return pipeline

    namespace = SimpleNamespace(
        pipeline=new_pipeline,
        config=FakeConfig,
        align=FakeAlign,
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(rgb8="rgb8", z16="z16"),
        camera_info=SimpleNamespace(name="name"),
        option=SimpleNamespace(enable_auto_exposure="auto_exposure", exposure="exposure", gain="gain"),
    )
    monkeypatch.setattr(collection_cameras, "rs", namespace)
    monkeypatch.setattr(collection_cameras, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(collection_cameras, "Thread", FakeThread)
    return namespace, pipelines, options


def test_each_camera_frame_is_consumed_only_once():
    cameras = RealSenseCollectionCameras({"head": "123"}, frame_timeout_s=0.1)
    first = sample(7)
    cameras._latest["head"] = first
    assert cameras.read()["head"] is first
    with pytest.raises(TimeoutError, match="fresh frames"):
        cameras.read(timeout_s=0.002)
    second = sample(8)
    cameras._latest["head"] = second
    assert cameras.read()["head"] is second


def test_every_camera_must_advance_before_consuming_any():
    cameras = RealSenseCollectionCameras({"left": "123", "right": "456"}, frame_timeout_s=0.1)
    cameras._latest.update(left=sample(7), right=sample(9))
    assert set(cameras.read()) == {"left", "right"}
    cameras._latest["left"] = sample(8)
    with pytest.raises(TimeoutError):
        cameras.read(timeout_s=0.002)
    assert cameras._last_numbers == {"left": 7, "right": 9}
    cameras._latest["right"] = sample(10)
    result = cameras.read()
    assert result["left"].frame_number == 8
    assert result["right"].frame_number == 10


@pytest.mark.parametrize("age_ns", [1_000_000_000, -1_000_000_000])
def test_rejects_old_or_future_host_capture_timestamps(age_ns):
    cameras = RealSenseCollectionCameras({"head": "123"}, frame_timeout_s=0.002)
    cameras._latest["head"] = sample(1, age_ns=age_ns)
    with pytest.raises(TimeoutError):
        cameras.read()


def test_stopping_wakes_reader_and_does_not_return_consumed_sample():
    cameras = RealSenseCollectionCameras({"head": "123"})
    cameras._latest["head"] = sample(1)
    cameras.read()
    cameras.disconnect()
    with pytest.raises(TimeoutError):
        cameras.read()


def test_stopped_camera_rejects_even_unconsumed_ready_frame():
    cameras = RealSenseCollectionCameras({"head": "123"})
    cameras._latest["head"] = sample(1)
    cameras.disconnect()
    with pytest.raises(TimeoutError, match="stopped"):
        cameras.read()
    assert cameras._last_numbers == {}


def test_capture_preserves_rgbd_frameset_pairing_and_converts_depth_scale(fake_rs):
    cameras = RealSenseCollectionCameras({"head": "123"})
    frames = [FakeFrames(10, 1, 1000), FakeFrames(11, 1, 2000), FakeFrames(12, 2, 3000)]
    calibration = {"depth_scale_m": 0.00025, "serial": "123"}
    pipeline = FakeCapturePipeline(cameras, frames)
    cameras._capture("head", pipeline, calibration)
    # The synchronous fake has stopped the capture loop after its finite input.
    frame = cameras._latest["head"]
    assert frame.frame_number == 12
    assert frame.device_timestamp_ms == 120.0
    assert frame.timestamp_ns <= time.monotonic_ns()
    np.testing.assert_array_equal(frame.rgb, np.full((2, 3, 3), 12, dtype=np.uint8))
    np.testing.assert_array_equal(frame.depth_mm, np.full((2, 3), 750.0))
    # Export owns its pixels instead of referencing mutable device buffers.
    frames[-1].color.data[:] = 0
    frames[-1].depth.data[:] = 0
    assert frame.rgb[0, 0, 0] == 12
    assert frame.depth_mm[0, 0] == 750.0
    assert frame.calibration is calibration


def test_worker_skips_a_repeated_depth_frame(fake_rs):
    cameras = RealSenseCollectionCameras({"head": "123"})
    pipeline = FakeCapturePipeline(cameras, [FakeFrames(10, 1, 1000), FakeFrames(11, 1, 2000)])
    cameras._capture("head", pipeline, {"depth_scale_m": 0.001})
    frame = cameras._latest["head"]
    assert frame.frame_number == 10
    assert frame.depth_mm[0, 0] == 1000.0


def test_rgb_only_worker_does_not_align_or_access_depth(fake_rs, monkeypatch):
    namespace, _, _ = fake_rs

    def forbidden_align(*args):
        raise AssertionError("RGB-only capture attempted depth alignment")

    monkeypatch.setattr(namespace, "align", forbidden_align)
    cameras = RealSenseCollectionCameras({"head": "123"}, use_depth=False)
    frames = FakeFrames(10, 1, 1000)
    cameras._capture("head", FakeCapturePipeline(cameras, [frames]), {})
    assert cameras._latest["head"].depth_mm is None
    assert not frames.aligned


def test_capture_worker_exception_propagates_original_cause(fake_rs):
    cameras = RealSenseCollectionCameras({"head": "123"})
    cause = OSError("USB camera disconnected")
    cameras._capture("head", FakeCapturePipeline(cameras, [cause]), {"depth_scale_m": 0.001})
    with pytest.raises(RuntimeError, match="camera head failed") as caught:
        cameras.read()
    assert caught.value.__cause__ is cause


def test_alignment_setup_exception_is_reported_as_worker_failure(fake_rs, monkeypatch):
    namespace, _, _ = fake_rs

    def bad_align(*args):
        raise OSError("alignment initialization failed")

    monkeypatch.setattr(namespace, "align", bad_align)
    cameras = RealSenseCollectionCameras({"head": "123"})
    cameras._capture("head", FakeCapturePipeline(cameras, []), {})
    with pytest.raises(RuntimeError, match="camera head failed") as caught:
        cameras.read()
    assert str(caught.value.__cause__) == "alignment initialization failed"


def test_connect_exports_calibration_and_stops_each_started_pipeline(fake_rs, monkeypatch):
    _, pipelines, options = fake_rs
    cameras = RealSenseCollectionCameras(
        {"head": "123", "wrist": "456"}, width=3, height=2, exposure=200, gain=16
    )
    monkeypatch.setattr(cameras, "read", lambda **kwargs: {})
    cameras.connect()
    threads = list(cameras._threads)
    assert all(thread.started for thread in threads)
    assert [pipeline.config.serial for pipeline in pipelines] == ["123", "456"]
    assert all(
        [stream[0] for stream in pipeline.config.streams] == ["color", "depth"] for pipeline in pipelines
    )
    calibration = threads[0].kwargs["args"][2]
    assert calibration["serial"] == "123"
    assert calibration["depth_scale_m"] == 0.00025
    np.testing.assert_allclose(calibration["intrinsic_cv"], [[2.0, 0, 1.0], [0, 2.5, 0.5], [0, 0, 1]])
    np.testing.assert_allclose(calibration["depth_to_color_translation_m"], [0.01, 0, 0])
    np.testing.assert_allclose(calibration["depth_to_color_rotation"], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert options == [("auto_exposure", 0), ("exposure", 200), ("gain", 16)] * 2
    cameras.disconnect()
    assert all(pipeline.stop_calls == 1 for pipeline in pipelines)
    assert all(thread.joined for thread in threads)
    assert cameras._pipelines == {}
    assert cameras._threads == []
    cameras.disconnect()
    assert all(pipeline.stop_calls == 1 for pipeline in pipelines)


def test_partial_start_failure_cleans_all_pipelines_and_preserves_original_error(fake_rs, monkeypatch):
    namespace, pipelines, _ = fake_rs
    original_factory = namespace.pipeline

    def failing_factory():
        pipeline = original_factory()
        pipeline.fail_start = len(pipelines) == 2
        return pipeline

    monkeypatch.setattr(namespace, "pipeline", failing_factory)
    cameras = RealSenseCollectionCameras({"head": "123", "wrist": "456"})
    with pytest.raises(OSError, match="second camera cannot start"):
        cameras.connect()
    assert len(pipelines) == 2
    assert all(pipeline.stop_calls == 1 for pipeline in pipelines)
    assert not pipelines[0].started
    assert cameras._pipelines == {}
    assert cameras._threads == []
    assert cameras._stop.is_set()


def test_failed_thread_start_skips_unstarted_join_and_releases_pipeline(fake_rs, monkeypatch):
    _, pipelines, _ = fake_rs
    original_thread = collection_cameras.Thread
    created = []

    def failing_thread(**kwargs):
        thread = original_thread(**kwargs)
        thread.fail_start = True
        created.append(thread)
        return thread

    monkeypatch.setattr(collection_cameras, "Thread", failing_thread)
    cameras = RealSenseCollectionCameras({"head": "123"})
    with pytest.raises(OSError, match="capture worker cannot start"):
        cameras.connect()
    assert len(created) == 1
    assert created[0].ident is None
    assert not created[0].joined
    assert pipelines[0].stop_calls == 1
    assert cameras._threads == []
    assert cameras._pipelines == {}


def test_live_worker_after_join_blocks_reconnect_and_preserves_stop_signal(fake_rs, monkeypatch):
    _, pipelines, _ = fake_rs
    cameras = RealSenseCollectionCameras({"head": "123"})
    monkeypatch.setattr(cameras, "read", lambda **kwargs: {})
    cameras.connect()
    worker = cameras._threads[0]
    worker.survives_join = True
    with pytest.raises(RuntimeError, match="worker did not terminate"):
        cameras.disconnect()
    assert cameras._threads == [worker]
    assert cameras._stop.is_set()
    with pytest.raises(RuntimeError, match="already connected"):
        cameras.connect()
    assert cameras._stop.is_set()
    assert len(pipelines) == 1
    worker.survives_join = False
    cameras.disconnect()
    assert cameras._threads == []


@pytest.mark.parametrize("timeout_s", [0, -1, float("nan"), float("inf")])
def test_invalid_read_timeout_is_rejected(timeout_s):
    cameras = RealSenseCollectionCameras({"head": "123"})
    with pytest.raises(ValueError):
        cameras.read(timeout_s=timeout_s)
