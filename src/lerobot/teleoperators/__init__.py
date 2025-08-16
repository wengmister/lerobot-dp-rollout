#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator
from .utils import make_teleoperator_from_config
from .vr_teleoperator import VRTeleoperator, VRTeleoperatorConfig

# Import teleoperator modules to trigger registration
from . import (  # noqa: F401
    bi_so100_leader,
    franka_fer_vr,
    homunculus,
    koch_leader,
    so100_leader,
    so101_leader,
    vr_teleoperator,
    xhand_vr,
)
