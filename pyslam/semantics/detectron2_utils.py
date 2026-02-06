"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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
"""

import importlib

_DETECTRON2_IMPORT_CHECKED = False


def check_detectron2_import() -> None:
    """Check detectron2 importability and warn on common ABI issues."""
    global _DETECTRON2_IMPORT_CHECKED
    if _DETECTRON2_IMPORT_CHECKED:
        return
    try:
        importlib.import_module("detectron2._C")
    except Exception as e:
        _warn_detectron2_import_error(e)
        raise
    _DETECTRON2_IMPORT_CHECKED = True


def _warn_detectron2_import_error(error: Exception) -> None:
    error_text = str(error)
    if "detectron2/_C" in error_text or "undefined symbol" in error_text:
        try:
            from pyslam.utilities.logging import Printer

            Printer.red(
                "Detectron2 C++ extension failed to load. This usually means it was "
                "compiled against a different PyTorch version. Rebuild detectron2 after "
                "installing the desired torch/torchvision versions, for example:\n"
                "  cd thirdparty/detectron2\n"
                "  python -m pip install --no-build-isolation -e . --force-reinstall\n"
                "If the problem persists, remove any old build artifacts and retry."
            )
        except Exception:
            pass
