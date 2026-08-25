# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Make the Sphinx extensions under ``docs/_ext`` importable by the docs tests.

Runs at collection time, before the test modules are imported.
"""

from pathlib import Path
import sys

_EXT_DIR = Path(__file__).resolve().parents[2] / "docs" / "_ext"
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))
