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
* Read-only visual-model inspection. No constructor runs and no weights download.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .model_registry import get_spec, list_models
from .weights import default_cache, offline_requested, registered_weights, resolve_registered_weights, WeightsUnavailable

ROOT = Path(__file__).resolve().parents[2]


def inspect(spec):
    wrapper = spec.wrapper
    module = wrapper.split(":", 1)[0] if wrapper else None
    if not module:
        code = "sentinel; use tracker API"
    elif module.startswith("pyslam."):
        code = "wrapper found; import not tested" if (ROOT / (module.replace(".", "/") + ".py")).is_file() else "missing wrapper"
    elif module.startswith("vpr."):
        code = "vendor file found; import not tested" if (ROOT / "thirdparty" / (module.replace(".", "/") + ".py")).is_file() else "missing vendor file"
    else:
        code = "package found; import not tested" if importlib.util.find_spec(module.split(".")[0]) else "missing package"
    source = spec.source()
    artifacts = sorted(name for name, entry in registered_weights().items()
                       if entry.get("model") == f"{spec.kind}:{spec.name}")
    if artifacts:
        weights = (f"checksum-pinned artifacts registered: {', '.join(artifacts)}; "
                   "verify with `prefetch NAME --offline` (legacy constructors still use their own paths)")
    elif source.get("weights"):
        weights = "legacy paths; presence and checksum not yet verified"
    else:
        weights = "not declared"
    return {
        "name": spec.name, "kind": spec.kind, "group": spec.group, "code": code,
        "provides_descriptors": spec.provides_descriptors,
        "weights": weights,
        "license": spec.license() or "review pending",
        "revision": source.get("revision"), "wrapper": wrapper,
    }


def markdown_table():
    lines = ["| Model | Kind | Per-keypoint descriptors | Group | License |",
             "| --- | --- | --- | --- | --- |"]
    for spec in list_models():
        if spec.wrapper:
            review = spec.license()
            license_text = ("existing integration (not re-reviewed)"
                            if not review or review.startswith(("existing", "pending")) else review)
            lines.append(f"| {spec.name} | {spec.kind} | {'yes' if spec.provides_descriptors else 'no'} | "
                         f"{spec.group} | {license_text} |")
    return "\n".join(lines)


def run_install(args, parser):
    from .install import INSTALLERS, Installer, install, legacy_report, plan

    if args.list:
        for name, installer in INSTALLERS.items():
            print(f"{name:10s} {installer.spec_kind:18s} {get_spec(installer.spec_name, installer.spec_kind).group}")
        return 0
    reports, failed = [], False
    for item in plan(args.names, args.group):
        if isinstance(item, Installer):
            try:
                reports.append(install(item, dry_run=args.dry_run))
            except Exception as exc:  # report and continue with the other models
                reports.append({"name": item.spec_name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        else:
            try:
                spec = item if not isinstance(item, str) else get_spec(item, "local_feature")
            except ValueError:
                parser.error(f"Unknown model {item!r}; see `install --list` and `check`")
            reports.append(legacy_report(spec))
        failed |= reports[-1]["status"] == "failed"
    print(json.dumps(reports, indent=2))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Inspect metadata/files without importing or constructing models")
    check.add_argument("name", nargs="?")
    check.add_argument("--kind", choices=("local_feature", "pair_matcher", "global_descriptor"), default="local_feature")
    check.add_argument("--group", choices=("core", "dense_matchers", "tf_models", "vpr"))
    check.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    check.add_argument("--strict", action="store_true", help="Fail while dependency/weight/license verification is incomplete")
    check.add_argument("--artifacts", action="store_true",
                       help="Also report whether registered artifacts are cached (imports torch; no download)")
    sub.add_parser("table", help="Emit the model table from the registry")
    install = sub.add_parser("install", help="Prefetch code snapshots and weights of self-installing models, "
                                             "then verify registered checksums (legacy models: report their installer)")
    install.add_argument("names", nargs="*", help="Model names (default: every self-installing model)")
    install.add_argument("--group", choices=("core", "dense_matchers", "tf_models", "vpr"))
    install.add_argument("--list", action="store_true", help="List self-installing models and exit")
    install.add_argument("--dry-run", action="store_true", help="Show the plan and the cache status only")
    prefetch = sub.add_parser("prefetch", help="Explicitly fetch a registered checksum-pinned artifact")
    prefetch.add_argument("name", choices=tuple(registered_weights()))
    prefetch.add_argument("--cache-dir", type=Path)
    prefetch.add_argument("--offline", action="store_true", default=None)
    args = parser.parse_args()
    if args.command == "table":
        print(markdown_table())
        return 0
    if args.command == "install":
        return run_install(args, parser)
    if args.command == "prefetch":
        try:
            print(resolve_registered_weights(args.name, cache_dir=args.cache_dir, offline=args.offline,
                                             allow_download=True))
            return 0
        except WeightsUnavailable as exc:
            parser.exit(2, str(exc) + "\n")
    try:
        specs = (get_spec(args.name, args.kind),) if args.name else list_models(group=args.group)
    except ValueError as exc:
        parser.error(str(exc))
    device = "CPU selected; no GPU runtime initialized"
    if args.device == "cuda":
        # Check in a child process so an explicit CUDA check does not import torch
        # into a caller that is only inspecting the catalog.
        try:
            result = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                                    text=True, capture_output=True, timeout=30)
            device = "CUDA available" if result.returncode == 0 and result.stdout.strip() == "True" else "CUDA unavailable"
        except subprocess.TimeoutExpired:
            device = "CUDA probe timed out"
    models = [inspect(spec) for spec in specs]
    if args.artifacts:
        from .install import artifact_status, artifacts_for
        for spec, model in zip(specs, models):
            model["artifacts"] = {name: artifact_status(entry) for name, entry in artifacts_for(spec).items()}
    print(json.dumps({"models": models, "device": device,
                      "cache": str(default_cache()), "offline": offline_requested(),
                      "note": "Metadata inspection is not an inference, install, or license validation"}, indent=2))
    return 2 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
