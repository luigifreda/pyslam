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

import torch.multiprocessing as mp
import threading as th
import queue as queue_module
from queue import Empty as QueueEmpty

import traceback
import platform
import copy

from pyslam.utilities.logging import Printer


kPrintTrackebackDetails = True


def clone_obj(obj):
    """Clone an object recursively."""
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj


class Value:
    def __init__(self, type, value):
        self.type = type
        self.value = value


# Base class to inherit to get singleton at each constructor call
class SingletonBase:
    _instances = {}

    @classmethod
    def get_instance(cls, *args):
        # Create a key from the arguments passed to the constructor
        key = tuple(args)
        if key not in cls._instances:
            # If no instance exists with these arguments, create one
            instance = cls(*args)
            cls._instances[key] = instance
        return cls._instances[key]


class AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = th.Lock()

    def increment(self):
        with self._lock:
            self._value += 1

    def value(self):
        with self._lock:
            return self._value

    def increment_and_get(self):
        with self._lock:
            self._value += 1
            return self._value


# ================================
# Queue helper functions
# ================================


def push_to_front(queue, item):
    temp_list = [item]
    # Use get(block=False) to avoid blocking
    while True:
        try:
            temp_list.append(queue.get(block=False))
        except:
            break
    # Put all items back in the queue, starting with the new item at the front
    for i in temp_list:
        try:
            # Try non-blocking put first
            queue.put(i, block=False)
        except:
            # If queue is full, fall back to blocking put with timeout
            try:
                queue.put(i, timeout=1.0)
            except:
                # If timeout fails, log and continue (item will be lost)
                if kPrintTrackebackDetails:
                    Printer.orange(f"push_to_front: failed to put item in queue, item may be lost")


# empty a queue before exiting from the consumer thread/process for safety
def empty_queue(queue, verbose=True):
    # if platform.system() == 'Darwin':
    try:
        while not queue.empty():
            queue.get_nowait()
    except QueueEmpty:
        # Queue is empty, which is expected - just return silently
        pass
    except Exception as e:
        if verbose:
            Printer.red(f"EXCEPTION in empty_queue: {e}")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                print(f"\t traceback details: {traceback_details}")


def get_last_item_from_queue(queue):
    """Get the last item from a queue without blocking."""
    last_item = None
    while True:
        try:
            last_item = queue.get_nowait()
        except Exception as e:
            break
    return last_item


class FixedSizeQueue:
    def __init__(self, maxsize):
        self.queue = mp.Queue()
        self.maxsize = maxsize
        self.size = mp.Value("i", 0)

    def put(self, item):
        with self.size.get_lock():
            if self.size.value >= self.maxsize:
                # pop the oldest element from the queue without using it
                item = self.queue.get()
                self.size.value -= 1
            self.queue.put(item)
            self.size.value += 1

    def get(self):
        with self.size.get_lock():
            if self.size.value > 0:
                item = self.queue.get()
                self.size.value -= 1
                return item
            else:
                raise IndexError("Queue is empty.")

    def qsize(self):
        with self.size.get_lock():
            return self.size.value


class SafeQueue:
    def __init__(self, maxsize=0):
        """
        A wrapper around multiprocessing.Queue with a custom qsize method.

        :param maxsize: The maximum size of the queue (default: 0 for unlimited size).
        """
        self.queue = mp.Queue(maxsize)
        self._size = mp.Value("i", 0)  # Shared integer to track the size of the queue
        self._lock = mp.Lock()  # Lock to ensure thread-safe operations

    def put(self, item, block=True, timeout=None):
        """Put an item into the queue."""
        with self._lock:
            self._size.value += 1
        return self.queue.put(item, block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue."""
        item = self.queue.get(block=block, timeout=timeout)
        with self._lock:
            self._size.value -= 1
        return item

    def get_nowait(self):
        """Remove and return an item from the queue without blocking."""
        item = self.queue.get_nowait()
        with self._lock:
            if item:
                self._size.value -= 1
        return item

    def qsize(self):
        """Return the current size of the queue."""
        with self._lock:
            return self._size.value

    def empty(self):
        """Check if the queue is empty."""
        with self._lock:
            return self._size.value == 0

    def full(self):
        """Check if the queue is full."""
        with self._lock:
            if self.queue._maxsize > 0:
                return self._size.value >= self.queue._maxsize
            return False

    def close(self):
        """Close the underlying queue."""
        self.queue.close()

    def join_thread(self):
        """Join the queue's worker thread."""
        self.queue.join_thread()

    def cancel_join_thread(self):
        """Cancel the queue's worker thread."""
        self.queue.cancel_join_thread()


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        return None

    def qsize(self):
        return 0

    def empty(self):
        return True


# ================================
# Dictionary helper functions
# ================================


def static_fields_to_dict(cls):
    return {
        key: value
        for key, value in vars(cls).items()
        if not key.startswith("__") and not callable(value)
    }


# Recursively merge two dictionaries without data sharing between them.
# No side effects on the two input dictionaries.
def merge_dicts(a, b):
    merged_dict = {}
    for key in a:
        if key in b:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merged_dict[key] = merge_dicts(a[key].copy(), b[key].copy())
            elif isinstance(a[key], list) and isinstance(b[key], list):
                merged_dict[key] = []
                merged_dict[key].extend([x for x in a[key] if x not in merged_dict[key]])
                merged_dict[key].extend([x for x in b[key] if x not in merged_dict[key]])
            elif a[key] == b[key]:
                merged_dict[key] = copy.deepcopy(a[key])  # same leaf value
            else:
                merged_dict[key] = copy.deepcopy(b[key])  # replace value
        else:
            merged_dict[key] = copy.deepcopy(a[key])  # add key
    for key in b:
        if key not in a:
            merged_dict[key] = copy.deepcopy(b[key])
    return merged_dict
