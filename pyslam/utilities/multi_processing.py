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
"""

import sys
import platform

from .data_management import SafeQueue


# Utilities for multiprocessing


class MultiprocessingManager:
    # NOTE: The usage of the multiprocessing Manager().Queue() generates pickling problems
    #       when we use set_start_method('spawn') with torch.multiprocessing (which is needed by torch with CUDA).
    #       Thereofore, we use this MultiprocessingManager to manage queues in slightly different way.
    #       In general, the usage of the multiprocessing Manager() seem to return smoother interactions. For this
    #       reason, we use it by default.
    def __init__(self, use_manager=True, verbose=False):
        import torch.multiprocessing as mp

        self.manager = None
        self.start_method = mp.get_start_method()
        self.verbose = verbose
        if verbose:
            print(f"MultiprocessingManager: start method: {self.start_method}")
        if use_manager and self.start_method != "spawn":
            self.manager = mp.Manager()  # use a memory manager when start method is not 'spawn'

    @staticmethod
    def is_start_method_spawn():
        import torch.multiprocessing as mp

        return mp.get_start_method() == "spawn"

    def Queue(self, maxsize=0):
        import torch.multiprocessing as mp

        if self.manager is not None:
            # the start method is not 'spawn' => we prefer to use the multiprocessing manager
            return self.manager.Queue(maxsize=maxsize)
        else:
            # the start method is 'spawn' => we prefer to use standard multiprocessing Queue
            if platform.system() == "Darwin":
                return SafeQueue(maxsize=maxsize)
            else:
                return mp.Queue()

    def Value(self, typecode_or_type, *args, lock=True):
        import torch.multiprocessing as mp

        return mp.Value(typecode_or_type=typecode_or_type, *args, lock=lock)

    def Dict(self):
        if self.manager is not None:
            return self.manager.dict()
        else:
            return {}

    def shutdown(self):
        """
        Shutdown the multiprocessing manager if it exists.
        This is important for clean shutdown to terminate the SyncManager process.
        """
        if self.manager is not None:
            try:
                self.manager.shutdown()
                if self.verbose:
                    print("MultiprocessingManager: manager shut down successfully.")
            except Exception as e:
                if self.verbose:
                    print(f"MultiprocessingManager: Warning: Error shutting down manager: {e}")
            finally:
                self.manager = None
