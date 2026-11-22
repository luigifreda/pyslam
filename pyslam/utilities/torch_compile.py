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

Opt-in torch.compile for learned networks (Parameters.kTorchCompile, PYSLAM_TORCH_COMPILE=1).

Backend: Parameters.kTorchCompileBackend / PYSLAM_TORCH_COMPILE_BACKEND, "inductor"
(default, needs triton) or "tensorrt" (needs torch_tensorrt). If the backend is missing,
or compilation fails at the first call, the original function is used from then on and
a warning is printed: enabling the option never breaks a run. The first calls are slow
(compilation); outputs can differ by float rounding.
"""

import functools
import importlib.util
import warnings

from pyslam.config_parameters import Parameters


def _backend():
    backend = Parameters.kTorchCompileBackend
    if backend == "tensorrt" and importlib.util.find_spec("torch_tensorrt") is None:
        warnings.warn("PYSLAM_TORCH_COMPILE_BACKEND=tensorrt but torch_tensorrt is not installed: not compiling")
        return None
    if backend == "inductor" and importlib.util.find_spec("triton") is None:
        warnings.warn("torch.compile inductor backend needs triton: not compiling")
        return None
    return backend


def maybe_compile(fn, name="model"):
    """Return a compiled version of fn when the option is enabled, else fn unchanged."""
    if not Parameters.kTorchCompile:
        return fn
    backend = _backend()
    if backend is None:
        return fn
    import torch

    if backend == "tensorrt":
        import torch_tensorrt  # noqa: F401  (registers the backend)
    compiled = torch.compile(fn, backend=backend)
    state = {"failed": False}

    @functools.wraps(fn)
    def call(*args, **kwargs):
        if not state["failed"]:
            try:
                return compiled(*args, **kwargs)
            except Exception as exc:  # fall back once, permanently
                state["failed"] = True
                warnings.warn(f"torch.compile failed for {name} ({type(exc).__name__}: {exc}); using eager mode")
        return fn(*args, **kwargs)

    print(f"torch.compile enabled for {name} (backend={backend}); first calls compile")
    return call


def compile_module_forward(module, name="model"):
    """Compile module.forward in place (calls through methods like .extract() benefit too)."""
    if Parameters.kTorchCompile and module is not None:
        module.forward = maybe_compile(module.forward, name)
    return module
