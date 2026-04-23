from __future__ import annotations

import builtins
import importlib

wandb_import_error = None

try:
    _wandb = importlib.import_module("wandb")
except Exception as exc:
    _wandb = None
    wandb_import_error = exc


class _DisabledWandbTable:
    def __init__(self, columns=None):
        self.columns = columns or []
        self.rows = []

    def add_data(self, *args):
        self.rows.append(args)


class _DisabledWandb:
    Table = _DisabledWandbTable
    run = None

    def init(self, *args, **kwargs):
        return None

    def define_metric(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


wandb = _wandb if _wandb is not None else _DisabledWandb()

_has_warned = False


def warn_if_wandb_unavailable(print_fun=None):
    global _has_warned

    if wandb_import_error is None or _has_warned:
        return False

    _has_warned = True
    printer = print_fun if print_fun is not None else builtins.print
    printer(
        f"WARNING: wandb import failed ({wandb_import_error}). W&B logging is disabled."
    )
    return True
