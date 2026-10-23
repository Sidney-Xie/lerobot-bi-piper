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

from dataclasses import dataclass

from ..bi_piper import BiPiperConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("bi_piperx")
@dataclass(kw_only=True)
class BiPiperXConfig(BiPiperConfig):
    """BiPiper configuration with policy-facing j4 and j5 exchanged."""

    id: str | None = "bi_piperx"
