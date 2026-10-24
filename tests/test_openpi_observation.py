"""Hardware-free regression coverage for OpenPI rollout observations."""

import unittest

import numpy as np
import pytest

from lerobot.rollout.inference.factory import OpenPIWebSocketInferenceConfig
from lerobot.rollout.inference.openpi_websocket import OpenPIWebSocketInferenceEngine
from lerobot.utils.feature_utils import hw_to_dataset_features

pytest.importorskip("openpi_client", reason="Install the client from the OpenPI submodule")


class TestOpenPIObservation(unittest.TestCase):
    def test_notify_aggregates_hardware_fields_in_feature_order(self):
        names = [
            f"{side}_{joint}.pos"
            for side in ("left", "right")
            for joint in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
        ]
        cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        features = dict.fromkeys(names, float)
        features.update(dict.fromkeys(cameras, (8, 8, 3)))
        engine = OpenPIWebSocketInferenceEngine(
            OpenPIWebSocketInferenceConfig(image_size=8),
            hw_features=hw_to_dataset_features(features, "observation"),
            ordered_action_keys=names,
            task="test task",
            fps=25,
        )
        # Activate notification without starting a socket or any hardware.
        engine._started = True
        engine._active.set()
        observation = {name: float(index) for index, name in reversed(list(enumerate(names)))}
        for index, name in enumerate(cameras):
            observation[name] = np.full((8, 8, 3), index * 50, dtype=np.uint8)

        engine.notify_observation(observation)

        payload = engine._latest_observation
        np.testing.assert_array_equal(payload["state"], np.arange(14, dtype=np.float32))
        self.assertEqual(payload["state"].dtype, np.float32)
        self.assertEqual(payload["prompt"], "test task")
        for index, name in enumerate(cameras):
            np.testing.assert_array_equal(
                payload["images"][name], np.full((3, 8, 8), index * 50, dtype=np.uint8)
            )
        self.assertTrue(engine._request_event.is_set())
        self.assertNotIn("observation.state", observation)


if __name__ == "__main__":
    unittest.main()
