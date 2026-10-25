"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
*
* pyslam.slam binds its classes to one core on first import, so a pytest process tests one core.
*
* `pytest tests` runs on the C++ core and skips tests/python_core.
* `PYSLAM_USE_CPP=false pytest tests/python_core tests/slam` runs the Python core.
"""

import os
from pathlib import Path

PYTHON_CORE_DIR = Path(__file__).resolve().parent / "python_core"


def _python_core_selected():
    return os.environ.get("PYSLAM_USE_CPP", "").strip().lower() in ("0", "false", "no", "off")


def pytest_ignore_collect(collection_path, config):
    if _python_core_selected():
        return None
    path = Path(collection_path).resolve()
    if path == PYTHON_CORE_DIR or PYTHON_CORE_DIR in path.parents:
        return True
    return None
