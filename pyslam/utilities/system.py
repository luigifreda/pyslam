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
import os
import numpy as np
import logging
from termcolor import colored
import cv2

import threading
import logging
from logging.handlers import QueueHandler, QueueListener

# import multiprocessing as mp
import torch.multiprocessing as mp


from pathlib import Path
import gdown
import requests  # Use requests for general HTTP downloads
from tqdm import tqdm  # Import tqdm for progress bars

import shutil
import tempfile
import atexit
import time
import signal
import traceback
from typing import Optional


def getchar():
    print("press enter to continue:")
    a = input("").split(" ")[0]
    print(a)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False


# This function check and exec 'from module import name' and directly return the 'name'.'method'.
# The method is used to directly return a 'method' of 'name' (i.e. 'module'.'name'.'method')
# N.B.: if a method is needed you CAN'T
#   from module import name.method
# since method is an attribute of name!
def import_from(module, name, method=None, debug=False):
    from .logging import Printer

    try:
        imported_module = __import__(module, fromlist=[name])
        imported_name = getattr(imported_module, name)
        if method is None:
            return imported_name
        else:
            return getattr(imported_name, method)
    except:
        if method is not None:
            name = name + "." + method
        Printer.orange(
            "WARNING: cannot import "
            + name
            + " from "
            + module
            + ", check the file docs/TROUBLESHOOTING.md"
        )
        if debug:
            Printer.orange(traceback.format_exc())
        return None


def get_opencv_version():
    opencv_major = int(cv2.__version__.split(".")[0])
    opencv_minor = int(cv2.__version__.split(".")[1])
    opencv_build = int(cv2.__version__.split(".")[2])
    return (opencv_major, opencv_minor, opencv_build)


def is_opencv_version_greater_equal(a, b, c):
    opencv_version = get_opencv_version()
    return (
        opencv_version[0] * 1000 + opencv_version[1] * 100 + opencv_version[2]
        >= a * 1000 + b * 100 + c
    )


def check_if_main_thread(message=""):
    if threading.current_thread() is threading.main_thread():
        print(f"This is the main thread. {message}")
        return True
    else:
        print(f"This is NOT the main thread. {message}")
        return False


# Set the limit of open files. This is useful when using multiprocessing and socket management
# returns the error: OSError: [Errno 24] Too many open files.
def set_rlimit():
    import resource

    # Check the current soft and hard limits
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

    # Set the new limit
    new_soft_limit = 4096
    if new_soft_limit > soft_limit:
        print(f"set_rlimit(): Current soft limit: {soft_limit}, hard limit: {hard_limit}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))

        # Confirm the change
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"set_rlimit(): Updated soft limit: {soft_limit}, hard limit: {hard_limit}")


# To fix this issue under linux: https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found
def locally_configure_qt_environment():
    ci_and_not_headless = False  # Default value in case the import fails
    try:
        from cv2.version import ci_build, headless

        ci_and_not_headless = ci_build and not headless
    except ImportError:
        pass  # Handle the case where cv2.version does not exist

    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)  # Remove if exists
        os.environ.pop("QT_QPA_FONTDIR", None)  # Remove if exists


def force_kill_all_and_exit(code=0, verbose=True):
    """
    Force kill all remaining processes and exit the program.
    """
    # print("[!] Force shutdown initiated.")

    # Log active threads (excluding main)
    active_threads = [t for t in threading.enumerate() if t != threading.main_thread()]
    if active_threads:
        if verbose:
            print(f"[!] Active threads: {[t.name for t in active_threads]}")

    # Attempt to stop threads (cannot forcibly kill threads in Python)
    for t in active_threads:
        if verbose:
            print(f"[!] Thread {t.name} is still running and cannot be force-killed.")

    # Terminate all active multiprocessing children
    active_children = mp.active_children()
    if active_children:
        if verbose:
            print(f"[!] Active child processes: {active_children}")
    for p in active_children:
        try:
            if verbose:
                print(f"[!] Terminating process PID {p.pid}...")
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                if verbose:
                    print(f"[!] Killing stubborn process PID {p.pid}...")
                os.kill(p.pid, signal.SIGKILL)
        except Exception as e:
            if verbose:
                print(f"[!] Failed to terminate process PID {p.pid}: {e}")
            traceback.print_exc()

    # Wait briefly to allow processes to shut down
    time.sleep(0.5)

    if verbose:
        print("[✓] All processes attempted to terminate. Exiting.")
        print("[!] Note: Remaining threads (if any) will be terminated with the process.")

    # Force exit immediately - this bypasses all Python cleanup and thread cleanup
    # Note: The threads reported above are just warnings - os._exit() will terminate
    # the entire process regardless of any remaining threads. The process will exit
    # immediately even if QueueFeederThread or other threads are still running.
    os._exit(code)  # Bypass cleanup and exit immediately
