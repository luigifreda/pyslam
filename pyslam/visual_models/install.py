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

Registry-driven installation (prefetch) of visual models.

Models integrated through pinned source snapshots / hub artifacts (no pip changes) are
"installed" by constructing them once on the CPU with downloads allowed: this fetches
the pinned code and every checkpoint into the standard caches (torch.hub, Hugging Face),
including indirect ones such as DINOv2 backbones. Registered SHA-256 values are then
verified. Afterwards the models run offline (PYSLAM_OFFLINE=1).

Existing integrations keep their installers (scripts/install_*.sh, git submodules, pip
pins); for them this module only reports where they come from.
"""

from dataclasses import dataclass
import importlib
import os
from pathlib import Path

from .model_registry import get_spec, list_models
from .weights import registered_weights, verify


@dataclass(frozen=True)
class Installer:
    spec_kind: str
    spec_name: str
    constructor: str  # "module:callable", called with keyword arguments on CPU
    kwargs: tuple = ()


# Self-installing integrations (pinned snapshots / hub artifacts). Name -> installer.
INSTALLERS = {
    "DEDODE": Installer("local_feature", "DEDODE", "pyslam.local_features.feature_dedode:DeDoDeFeature2D",
                        (("num_features", 16), ("device", "cpu"))),
    "RDD": Installer("local_feature", "RDD", "pyslam.local_features.feature_rdd:RddFeature2D",
                     (("num_features", 16), ("device", "cpu"))),
    "ROMA": Installer("pair_matcher", "ROMA", "pyslam.local_features.feature_matcher_detector_free:RoMaMatcher",
                      (("device", "cpu"),)),
    "TINY_ROMA": Installer("pair_matcher", "TINY_ROMA",
                           "pyslam.local_features.feature_matcher_detector_free:TinyRoMaMatcher",
                           (("device", "cpu"),)),
    "SALAD": Installer("global_descriptor", "SALAD", "pyslam.loop_closing.global_feature_torchhub:GlobalFeatureSalad",
                       (("device", "cpu"),)),
    "BOQ": Installer("global_descriptor", "BOQ", "pyslam.loop_closing.global_feature_torchhub:GlobalFeatureBoQ",
                     (("device", "cpu"),)),
    "MEGALOC": Installer("global_descriptor", "MEGALOC", "pyslam.loop_closing.global_feature_megaloc:GlobalFeatureMegaloc",
                         (("device", "cpu"),)),
    "LOFTR": Installer("pair_matcher", "LOFTR", "pyslam.local_features.feature_matcher:LoFTRMatcher", ()),
}


def artifacts_for(spec):
    """Registered checksum-pinned artifacts of a spec: {artifact name: entry}."""
    target = f"{spec.kind}:{spec.name}"
    return {name: entry for name, entry in registered_weights().items() if entry.get("model") == target}


def artifact_locations(entry):
    """Candidate local paths of an artifact (torch.hub checkpoints, pyslam cache)."""
    import torch

    from .weights import WeightResolver, WeightSource

    paths = []
    for source in entry["sources"]:
        name = Path(source["filename"]).name
        paths.append(Path(torch.hub.get_dir()) / "checkpoints" / name)
        if source["kind"] == "hf":
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(source["repo_id"], source["filename"], revision=source.get("revision"))
            if isinstance(cached, str):
                paths.append(Path(cached))
        paths.append(WeightResolver().cached_path(WeightSource(**source)))
    return paths


def artifact_status(entry, verify_checksum=False):
    """'present', 'verified', 'checksum mismatch' or 'missing'."""
    digest = entry["sources"][0]["sha256"]
    for path in artifact_locations(entry):
        if path.is_file():
            if not verify_checksum:
                return "present"
            try:
                verify(path, digest)
                return "verified"
            except Exception:
                return "checksum mismatch"
    return "missing"


def plan(names=None, group=None):
    """Specs to install: explicit names, a dependency group, or every self-installing model."""
    if names:
        return [INSTALLERS[name.upper()] if name.upper() in INSTALLERS else name for name in names]
    selected = []
    for name, installer in INSTALLERS.items():
        spec = get_spec(installer.spec_name, installer.spec_kind)
        if group is None or spec.group == group:
            selected.append(installer)
    if group is not None:  # report the group's other members (installed by legacy scripts)
        covered = {(i.spec_kind, i.spec_name) for i in INSTALLERS.values()}
        selected += [spec for spec in list_models(group=group)
                     if spec.wrapper and (spec.kind, spec.name) not in covered]
    return selected


def install(installer, dry_run=False):
    """Construct once (fetching code + weights), then verify registered checksums."""
    spec = get_spec(installer.spec_name, installer.spec_kind)
    report = {"name": spec.name, "kind": spec.kind, "group": spec.group, "action": installer.constructor}
    if not dry_run:
        os.environ.pop("PYSLAM_OFFLINE", None)
        module, _, symbol = installer.constructor.partition(":")
        import contextlib
        import sys

        # Model constructors print progress: keep stdout for the machine-readable report.
        with contextlib.redirect_stdout(sys.stderr):
            constructor = getattr(importlib.import_module(module), symbol)
            instance = constructor(**dict(installer.kwargs))
            del instance
    report["artifacts"] = {name: (artifact_status(entry, verify_checksum=not dry_run) if not dry_run
                                  else artifact_status(entry))
                           for name, entry in artifacts_for(spec).items()}
    report["status"] = ("planned" if dry_run else
                        "failed" if any(v in ("missing", "checksum mismatch") for v in report["artifacts"].values())
                        else "installed")
    return report


def legacy_report(spec):
    source = spec.source()
    return {"name": spec.name, "kind": spec.kind, "group": spec.group, "status": "legacy installer",
            "installer": source.get("pin_source") or "scripts/install_thirdparty.sh / scripts/install_git_modules.sh"}
