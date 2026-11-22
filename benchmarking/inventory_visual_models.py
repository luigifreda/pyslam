#!/usr/bin/env python3
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
* Capture source/patch provenance and literal wrapper evidence before migrating models.
"""

import argparse
import ast
import configparser
from datetime import datetime, timezone
import math
import hashlib
from pathlib import Path
import re
import sys

from benchmark_common import ROOT, command, enum_snapshot, machine_info, provenance, sha256, write_json

EVIDENCE = re.compile(r"https?://|weights|checkpoint|torch\.hub|torch\.load|load_state|RLock|Lock\(|"
                      r"no_grad|inference_mode|non_blocking|pin_memory|cudnn|ThreadPool|set_lib|"
                      r"is_opencv_version|KeyPoint|match_lighterglue|n_layers=|num_workers")


def evidence(path):
    return [{"line": number, "text": line.strip()}
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
            if EVIDENCE.search(line)]


def contract_snapshot():
    # FeatureInfo remains the authority. Do not reproduce its numerical tables in code.
    sys.path.insert(0, str(ROOT))
    from pyslam.local_features.feature_types import FeatureInfo
    result = {"enums": enum_snapshot(), "FeatureInfo": {}}
    for name in ("norm_type", "max_descriptor_distance", "need_mask_management"):
        table = getattr(FeatureInfo, name)
        result["FeatureInfo"][name] = {
            f"{type(key).__name__}.{key.name}": ("Infinity" if isinstance(value, float) and math.isinf(value) else value)
            for key, value in table.items()
        }
    result["FeatureInfo"]["custom_detector_types"] = [key.name for key in FeatureInfo.custom_detector_types]
    return result


def inventory():
    modules = []
    sources = sorted((ROOT / "pyslam/local_features").glob("feature_*.py"))
    sources += sorted((ROOT / "pyslam/loop_closing").glob("*feature*.py"))
    sources += [ROOT / "pyslam/loop_closing/loop_detector_vpr.py"]
    for path in sources:
        tree = ast.parse(path.read_text())
        modules.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                        "classes": [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
                        "evidence": evidence(path)})
    submodules = []
    config = configparser.ConfigParser()
    config.read(ROOT / ".gitmodules")
    for section in config.sections():
        path = config[section]["path"]
        folder = ROOT / path
        licenses = [p for p in folder.glob("*")
                    if p.is_file() and p.name.lower().startswith(("license", "licence", "copying"))]
        local_diff = command(["git", "diff", "--binary", "HEAD"], cwd=folder)
        submodules.append({"path": path, "url": config[section]["url"],
                           "pinned_gitlink": command(["git", "ls-tree", "HEAD", path]),
                           "checkout_sha": command(["git", "rev-parse", "HEAD"], cwd=folder),
                           "local_diff_stat": command(["git", "diff", "--stat"], cwd=folder),
                           "local_diff": local_diff,
                           "local_diff_sha256": hashlib.sha256((local_diff or "").encode()).hexdigest(),
                           "license_evidence": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p),
                                                 "heading": p.read_text(errors="replace")[:300]} for p in licenses],
                           "license_review": "pending; code and weight terms must be checked separately"})
    patches = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                "files": re.findall(r"^\+\+\+ (.+)$", path.read_text(errors="replace"), re.M),
                "migration_status": "preserved in place; no migration attempted"}
               for path in sorted((ROOT / "thirdparty").glob("*.patch"))]
    installers = sorted((ROOT / "scripts").glob("install*.sh"))
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance(), "machine": machine_info(), "enums": enum_snapshot(),
        "modules": modules, "submodules": submodules, "patches": patches,
        "installers": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                        "pins": [line.strip() for line in path.read_text(errors="replace").splitlines()
                                 if re.search(r"==|numpy<|setuptools<|typing_extensions|tensorflow_hub<", line)]}
                       for path in installers],
        "history": command(["git", "log", "-80", "--format=%h %s", "--",
                            "pyslam/local_features", "pyslam/loop_closing", "thirdparty/*.patch", "scripts/install*.sh"]),
        "pip_freeze": command([sys.executable, "-m", "pip", "freeze"], timeout=60),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-contract", type=Path, help="Create the original golden contract; refuses to overwrite it")
    args = parser.parse_args()
    if args.write_contract and args.write_contract.exists():
        parser.error("The contract already exists; do not regenerate it to make a compatibility failure pass")
    write_json(args.output, inventory())
    if args.write_contract:
        write_json(args.write_contract, contract_snapshot())


if __name__ == "__main__":
    main()
