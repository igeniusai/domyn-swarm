# Copyright 2025 iGenius S.p.A
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

"""Make the Sphinx extensions under ``docs/_ext`` importable by the docs tests.

Runs at collection time, before the test modules are imported.
"""

from pathlib import Path
import sys

_EXT_DIR = Path(__file__).resolve().parents[2] / "docs" / "_ext"
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))
