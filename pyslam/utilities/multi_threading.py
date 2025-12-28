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
import traceback

from .data_management import SafeQueue
from .logging import Printer


# Utilities for multithreading

import threading
import time


kVerbose = False


# A simple task timer that can be started and stopped. Similar to QTimer in Qt.
# It works in a separate thread and calls a callback function at regular intervals.
class SimpleTaskTimer:
    def __init__(self, interval, callback, single_shot=False, name=""):
        """
        Initializes the task timer.
        :param interval: Interval in seconds between timer triggers.
        :param callback: Function to be called when the timer fires.
        :param single_shot: If True, the timer will only fire once.
        """
        self.name = name
        self.interval = interval
        self.callback = callback
        self.single_shot = single_shot
        self._thread = None
        self._stop_event = threading.Event()

    def __del__(self):
        if self._thread is not None and self._thread.is_alive():
            try:
                self.stop()
            except:
                pass

    def _run(self):
        """Internal method to handle the timer functionality."""
        if self.single_shot:
            time.sleep(self.interval)
            if not self._stop_event.is_set():
                if kVerbose:
                    print(f"SimpleTaskTimer {self.name}: single shot timer fired")
                try:
                    self.callback()
                except Exception as e:
                    Printer.red(f"SimpleTaskTimer {self.name}: error in callback: {e}")
                    traceback.print_exc()
        else:
            while not self._stop_event.is_set():
                time.sleep(self.interval)
                if not self._stop_event.is_set():
                    if kVerbose:
                        print(f"SimpleTaskTimer {self.name}: timer fired")
                    try:
                        self.callback()
                    except Exception as e:
                        Printer.red(f"SimpleTaskTimer {self.name}: error in callback: {e}")
                        traceback.print_exc()

    def start(self):
        """Starts the timer."""
        if self._thread is not None and self._thread.is_alive():
            print(f"SimpleTaskTimer {self.name} is already running!")
            return
        print(f"SimpleTaskTimer {self.name}: starting timer")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stops the timer."""
        self._stop_event.set()
        if self._thread is not None:
            if self._thread.is_alive():
                self._thread.join()
            self._thread = None
