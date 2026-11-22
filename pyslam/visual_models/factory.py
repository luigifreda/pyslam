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
* Resolve constructors lazily; model ownership stays with the caller/pipeline.
*
* Do not globally cache extractor instances: several wrappers remember the last image,
* keypoints and mutable num_features. Existing wrappers keep their own model reuse/locks.
"""

from functools import lru_cache
import importlib

from .model_registry import get_spec


class ModelUnavailable(ImportError):
    """A selected optional integration cannot be imported."""


@lru_cache(maxsize=None)
def resolve_symbol(target):
    module, separator, symbol = target.partition(":")
    if not separator or not module or not symbol:
        raise ValueError(f"Expected module:symbol, got {target!r}")
    value = importlib.import_module(module)
    for name in symbol.split("."):
        value = getattr(value, name)
    return value


@lru_cache(maxsize=None)
def _set_lib_once(lib_name):
    # config.cfg.set_lib does not deduplicate sys.path entries; register each vendor
    # path once per process instead of once per construction.
    importlib.import_module("pyslam.config").cfg.set_lib(lib_name, prepend=True)


def _unavailable(spec, exc):
    return ModelUnavailable(f"{spec.name}: optional integration {spec.wrapper} is unavailable. "
                            f"Dependency group: {spec.group}. See docs/visual_features.md. "
                            f"Original error: {exc}")


def _construct(name, kind, kwargs):
    spec = get_spec(name, kind)
    if spec.wrapper is None:
        raise ValueError(f"{spec.name} is a sentinel, not a constructible {kind}")
    # VPR remains in its caller's loop-detecting process. Merely listing/checking specs
    # never constructs a model or starts a worker.
    if kind == "global_descriptor" and spec.source_key in {"vpr", "patch_netvlad"}:
        _set_lib_once("vpr")
    try:
        constructor = resolve_symbol(spec.wrapper)
    except (ImportError, AttributeError) as exc:
        raise _unavailable(spec, exc) from exc
    try:
        return constructor(**kwargs)
    except ImportError as exc:
        # A lazily imported optional package inside the constructor. Other errors
        # (AttributeError, TypeError, ...) are genuine wrapper bugs and propagate.
        raise _unavailable(spec, exc) from exc


def get_local_feature(name, **kwargs):
    """Construct a local wrapper with its existing constructor arguments."""
    return _construct(name, "local_feature", kwargs)


def get_pair_matcher(name, **kwargs):
    return _construct(name, "pair_matcher", kwargs)


def get_global_descriptor(name, **kwargs):
    return _construct(name, "global_descriptor", kwargs)
