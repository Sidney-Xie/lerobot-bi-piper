#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lerobot.lerobot_types import RobotAction, RobotObservation

from ..bi_piper import BiPiper
from .config_bi_piperx import BiPiperXConfig

_SIDES = ("left", "right")


def _exchanged_joint_key(key: str) -> str:
    """Map a policy-side j4/j5 feature key to its physical Piper key and back."""
    for side in _SIDES:
        j4_key = f"{side}_j4.pos"
        j5_key = f"{side}_j5.pos"
        if key == j4_key:
            return j5_key
        if key == j5_key:
            return j4_key
    return key


class BiPiperX(BiPiper):
    """BiPiper variant whose policy-facing j4 and j5 are exchanged on both arms.

    Public features retain the standard BiPiper names and order. A policy j4
    command drives physical j5, while policy j5 drives physical j4. Joint
    feedback is exchanged symmetrically so observations use the same convention.
    """

    config_class = BiPiperXConfig
    name = "bi_piperx"

    def __init__(self, config: BiPiperXConfig):
        super().__init__(config)
        self.config = config

    def get_observation(self) -> RobotObservation:
        observation = super().get_observation()
        # Assign values rather than renaming keys to preserve the public feature order.
        for side in _SIDES:
            j4_key = f"{side}_j4.pos"
            j5_key = f"{side}_j5.pos"
            observation[j4_key], observation[j5_key] = observation[j5_key], observation[j4_key]
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        # Rename only at the hardware boundary. BiPiper then applies the physical
        # joint's own absolute and per-cycle limits before sending the CAN command.
        physical_action = {_exchanged_joint_key(key): value for key, value in action.items()}
        physical_sent = super().send_action(physical_action)
        logical_sent = {_exchanged_joint_key(key): value for key, value in physical_sent.items()}

        # Keep the result in the caller's feature order, including partial actions.
        return {key: logical_sent[key] for key in action if key in logical_sent}
