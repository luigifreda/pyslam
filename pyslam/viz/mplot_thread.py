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
import time
import os
import numpy as np

import platform

import matplotlib

# matplotlib.use('Agg')  # running non-interactive mode
# matplotlib.use("TkAgg")  # or "Agg" if you're just saving plots

import matplotlib.pyplot as plt

import logging

# import multiprocessing as mp
import torch.multiprocessing as mp
from pyslam.utilities.logging import Logging
from pyslam.utilities.system import locally_configure_qt_environment
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.config_parameters import Parameters

kPlotSleep = 0.04
kVerbose = False
kDebugAndPrintToFile = True

kSetDaemon = True  # from https://docs.python.org/3/library/threading.html#threading.Thread.daemon
# The entire Python program exits when no alive non-daemon threads are left.

kUseFigCanvasDrawIdle = True


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
# kLogsFolder = kRootFolder + '/logs'


if kVerbose and kDebugAndPrintToFile:
    # redirect the prints of local mapping to the file logs/local_mapping.log
    # you can watch the output in separate shell by running:
    # $ tail -f logs/mplot_thread.log
    logging_file = Parameters.kLogsFolder + "/mplot_thread.log"
    local_logger = Logging.setup_file_logger(
        "mplot_thread_logger", logging_file, formatter=Logging.simple_log_formatter
    )

    def print(*args, **kwargs):
        message = " ".join(
            str(arg) for arg in args
        )  # Convert all arguments to strings and join with spaces
        return local_logger.info(message, **kwargs)


kUsePlotPause = not kUseFigCanvasDrawIdle  # this should be set True under macOS
if platform.system() == "Darwin":
    kUsePlotPause = True

if kUseFigCanvasDrawIdle and platform.system() != "Darwin":
    plt.ion()

# NOTE: Here we are using processes instead of threads.
# The file name `mplot_thread.py` is still here since the classes are used as parallel drawing threads.


class FigureNum:
    figure_num = 0

    @staticmethod
    def getFigureNum():
        FigureNum.figure_num += 1
        return FigureNum.figure_num


# empty a queue before exiting from the consumer thread/process for safety
def empty_queue(queue):
    while not queue.empty():
        try:
            queue.get(timeout=0.001)
        except:
            pass


# global lock for drawing with matplotlib
class SharedSingletonLock:
    _instance = None  # Placeholder for singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedSingletonLock, cls).__new__(cls)
            cls._instance.lock = mp.Lock()  # Create the lock only once
        return cls._instance

    @property
    def get_lock(self):
        return self.lock


# use mplotlib figure to draw in 2d dynamic data
class Mplot2d:
    def __init__(self, xlabel: str = "", ylabel: str = "", title: str = ""):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self.data = None
        self.got_data = False
        self.handle_map = {}
        self.fig = None

        self.axis_computed = False
        self.xlim = [float("inf"), float("-inf")]
        self.ylim = [float("inf"), float("-inf")]
        self.xmin = float("inf")
        self.xmax = float("-inf")
        self.ymin = float("inf")
        self.ymax = float("-inf")

        self.key = mp.Value("i", 0)
        self.is_running = mp.Value("i", 1)

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue()
        self.key_queue = self.mp_manager.Queue()

        self.figure_num = mp.Value("i", int(FigureNum.getFigureNum()))
        print(
            f"Mplot2d: starting process {self.title} on figure: {self.figure_num.value}, backend: {matplotlib.get_backend()}"
        )

        self.lock = SharedSingletonLock().get_lock

        self.initialized = False  # New flag to track figure initialization

        args = (
            self.figure_num,
            self.queue,
            self.lock,
            self.key,
            self.is_running,
            self.key_queue,
        )
        self.process = mp.Process(target=self.run, args=args)
        # self.process.daemon = kSetDaemon
        self.process.start()

    def quit(self):
        print(f'Mplot2d "{self.title}" closing...')
        self.is_running.value = 0
        timeout = 5 if mp.get_start_method() != "spawn" else 10
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            print(
                f'Warning: Mplot2d "{self.title}" process did not terminate in time, forced kill.'
            )
            self.process.terminate()
        print(f'Mplot2d "{self.title}" closed')

    def init(self, figure_num, lock):
        lock.acquire()
        locally_configure_qt_environment()
        if kVerbose:
            print(mp.current_process().name, "initializing...")
        self.fig = plt.figure(figure_num)
        if kUseFigCanvasDrawIdle:
            self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        # self.ax = self.fig.gca(projection='3d')
        # self.ax = self.fig.gca()
        self.ax = self.fig.add_subplot(111)
        if self.title != "":
            self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if matplotlib.get_backend() == "Qt5Agg":
            self.ax.grid(
                visible=True, which="both", axis="both", color="gray", linestyle="-", linewidth=0.21
            )
        else:
            self.ax.grid()
        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        lock.release()
        self.initialized = True  # Set the flag to True after initialization

    def run(self, figure_num, queue, lock, key, is_running, key_queue):
        if kVerbose:
            print(f'Mplot2d "{self.title}": starting run on figure ', figure_num.value)
        self.key_queue_thread = key_queue
        # self.init(figure_num.value, lock)
        while is_running.value == 1:
            if kVerbose:
                print("Mplot2d: drawer_refresh step")
            self.drawer_refresh(queue, lock)
            if kUseFigCanvasDrawIdle:
                time.sleep(kPlotSleep)
        empty_queue(queue)  # empty the queue before exiting
        print(mp.current_process().name, f' - Mplot2d "{self.title}": closing fig {self.fig}')
        plt.close(self.fig)
        print(f"Mplot2d {self.title}: run: closed")

    def drawer_refresh(self, queue, lock):
        while not queue.empty():
            self.got_data = True
            self.data = queue.get()
            xy_signal, name, color, marker, linestyle, append = self.data

            # Initialize figure upon receiving the first data
            if not self.initialized:
                self.init(self.figure_num.value, lock)

            # print(mp.current_process().name,"refreshing : signal ", name)
            if name in self.handle_map:
                handle = self.handle_map[name]
                if append:
                    handle.set_xdata(np.append(handle.get_xdata(), xy_signal[0]))
                    handle.set_ydata(np.append(handle.get_ydata(), xy_signal[1]))
                else:
                    handle.set_xdata(xy_signal[0])
                    handle.set_ydata(xy_signal[1])
            else:
                (handle,) = self.ax.plot(
                    xy_signal[0],
                    xy_signal[1],
                    c=color,
                    marker=marker,
                    linestyle=linestyle,
                    label=name,
                )
                self.handle_map[name] = handle
        # print(mp.current_process().name,"got data: ", self.got_data)
        if self.got_data is True:
            self.plot_refresh(lock)

    def on_key_press(self, event):
        print(
            mp.current_process().name,
            f' - Mplot2d "{self.title}": key event pressed...  {event.key}',
        )
        if event.key == "escape":
            self.key.value = 27  # escape key
        elif len(event.key) == 1:
            self.key.value = ord(event.key)
        else:
            self.key.value = 0  # unrecognized or non-character key
        self.key_queue_thread.put(self.key.value)

    def on_key_release(self, event):
        print(
            mp.current_process().name,
            f' - Mplot2d "{self.title}": key event released... {event.key}',
        )
        self.key.value = 0  # reset to no key symbol

    def on_close(self, event):
        self.is_running.value = 0
        print(mp.current_process().name, f' - Mplot2d "{self.title}" closed figure')

    def get_key(self):
        if not self.key_queue.empty():
            key_val = self.key_queue.get()
            if isinstance(key_val, int) and 0 <= key_val <= 255:
                return chr(key_val)
            else:
                return ""
        else:
            return ""

    def setAxis(self):
        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        if not kUseFigCanvasDrawIdle:
            self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw(self, xy_signal, name, color="r", marker=".", linestyle="-", append=True):
        if self.queue is None:
            return
        if kVerbose:
            print(mp.current_process().name, f'Mplot2d "{self.title}" draw ')
        self.queue.put((xy_signal, name, color, marker, linestyle, append))

    def updateMinMax(self, np_signal):
        xmax, ymax = np.amax(np_signal, axis=0)
        xmin, ymin = np.amin(np_signal, axis=0)
        cx = 0.5 * (xmax + xmin)
        cy = 0.5 * (ymax + ymin)
        if False:
            # update maxs
            if xmax > self.xlim[1]:
                self.xlim[1] = xmax
            if ymax > self.ylim[1]:
                self.ylim[1] = ymax
            # update minsS
            if xmin < self.xlim[0]:
                self.xlim[0] = xmin
            if ymin < self.ylim[0]:
                self.ylim[0] = ymin
        # make axis actually squared
        if True:
            smin = min(xmin, ymin)
            smax = max(xmax, ymax)
            delta = 0.5 * (smax - smin)
            self.xlim = [cx - delta, cx + delta]
            self.ylim = [cy - delta, cy + delta]
        self.axis_computed = True

    def plot_refresh(self, lock):
        if kVerbose:
            print(mp.current_process().name, f'Mplot2d "{self.title}" refreshing ')
        lock.acquire()
        if self.is_running.value == 1:
            self.setAxis()
        # if not kUseFigCanvasDrawIdle:
        if kUsePlotPause:
            plt.pause(kPlotSleep)
        lock.release()


# use mplotlib figure to draw in 3D trajectories
class Mplot3d:
    def __init__(self, title: str = ""):
        self.title = title

        self.data = None
        self.got_data = False

        self.axis_computed = False
        self.xlim = [float("inf"), float("-inf")]
        self.ylim = [float("inf"), float("-inf")]
        self.zlim = [float("inf"), float("-inf")]
        self.xmin = float("inf")
        self.xmax = float("-inf")
        self.ymin = float("inf")
        self.ymax = float("-inf")
        self.zmin = float("inf")
        self.zmax = float("-inf")

        self.handle_map = {}

        self.key = mp.Value("i", 0)
        self.is_running = mp.Value("i", 1)

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue()
        self.key_queue = self.mp_manager.Queue()

        self.lock = SharedSingletonLock().get_lock

        self.figure_num = mp.Value("i", int(FigureNum.getFigureNum()))
        print(f"Mplot3d: starting the process on figure: {self.figure_num.value}")

        args = (
            self.figure_num,
            self.queue,
            self.lock,
            self.key,
            self.is_running,
            self.key_queue,
        )
        self.process = mp.Process(target=self.run, args=args)
        # self.process.daemon = kSetDaemon
        self.process.start()

    def quit(self):
        print(f'Mplot3d "{self.title}" closing...')
        self.is_running.value = 0
        timeout = 5 if mp.get_start_method() != "spawn" else 10
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            print('Warning: Mplot3d "{self.title}" process did not terminate in time, forced kill.')
            self.process.terminate()

    def init(self, figure_num, lock):
        lock.acquire()
        locally_configure_qt_environment()
        if kVerbose:
            print(mp.current_process().name, "initializing...")
        self.fig = plt.figure(figure_num)
        if kUseFigCanvasDrawIdle:
            self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.ax = self.fig.add_subplot(111, projection="3d")  # Adjusted line
        # self.ax = self.fig.gca(projection='3d')
        if self.title != "":
            self.ax.set_title(self.title)
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.set_zlabel("Z axis")
        self.setAxis()
        lock.release()

    def run(self, figure_num, queue, lock, key, is_running, key_queue):
        if kVerbose:
            print("Mplot3d: starting run on figure ", figure_num.value)
        self.key_queue_thread = key_queue
        self.init(figure_num.value, lock)
        while is_running.value == 1:
            self.drawer_refresh(queue, lock)
            if kUseFigCanvasDrawIdle:
                time.sleep(kPlotSleep)
        empty_queue(queue)  # empty the queue before exiting
        print(mp.current_process().name, f' - Mplot3d "{self.title}": closing fig ', self.fig)
        plt.close(self.fig)
        print(f"Mplot3d: run: closed")

    def drawer_refresh(self, queue, lock):
        while not queue.empty():
            self.got_data = True
            self.data = queue.get()
            traj, name, color, marker = self.data
            np_traj = np.asarray(traj)
            does_label_exist = False
            if name in self.handle_map:
                handle = self.handle_map[name]
                does_label_exist = True
            self.updateMinMax(np_traj[-1, :])
            handle = self.ax.scatter3D(
                np_traj[:, 0], np_traj[:, 1], np_traj[:, 2], c=color, marker=marker
            )
            if not does_label_exist:
                handle.set_label(name)
                self.handle_map[name] = handle
        if self.got_data is True:
            self.plot_refresh(lock)

    def on_key_press(self, event):
        print(
            mp.current_process().name, f' - Mplot3d "{self.title}": key event pressed...', event.key
        )
        if event.key == "escape":
            self.key.value = 27  # escape key
        elif len(event.key) == 1:
            self.key.value = ord(event.key)
        else:
            self.key.value = 0  # unrecognized or non-character key
        self.key_queue_thread.put(self.key.value)

    def on_key_release(self, event):
        print(
            mp.current_process().name,
            f' - Mplot3d "{self.title}": key event released...',
            event.key,
        )
        self.key.value = 0  # reset to no key symbol

    def on_close(self, event):
        self.is_running.value = 0
        print(mp.current_process().name, f' - Mplot3d "{self.title}" closed figure')

    def get_key(self):
        if not self.key_queue.empty():
            return chr(self.key_queue.get())
        else:
            return ""

    def setAxis(self):
        # self.ax.axis('equal')   # this does not work with the new matplotlib 3
        if self.axis_computed:
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.ax.set_zlim(self.zlim)
        self.ax.legend()
        # We need to draw *and* flush
        if not kUseFigCanvasDrawIdle:
            self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw(self, traj, name, color="r", marker="."):
        if self.queue is None:
            return
        self.queue.put((traj, name, color, marker))

    def updateMinMax(self, np_traj):
        # xmax,ymax,zmax = np.amax(np_traj,axis=0)
        # xmin,ymin,zmin = np.amin(np_traj,axis=0)
        x, y, z = np_traj[0], np_traj[1], np_traj[2]
        if self.xmin == float("inf"):
            self.xmin = x
        if self.xmax == float("-inf"):
            self.xmax = x
        if self.ymin == float("inf"):
            self.ymin = y
        if self.ymax == float("-inf"):
            self.ymax = y
        if self.zmin == float("inf"):
            self.zmin = z
        if self.zmax == float("-inf"):
            self.zmax = z

        self.xmin = min(self.xmin, x)
        self.xmax = max(self.xmax, x)
        self.ymin = min(self.ymin, y)
        self.ymax = max(self.ymax, y)
        self.zmin = min(self.zmin, z)
        self.zmax = max(self.zmax, z)

        cx = 0.5 * (self.xmax + self.xmin)
        cy = 0.5 * (self.ymax + self.ymin)
        cz = 0.5 * (self.zmax + self.zmin)
        if False:
            # update maxs
            if xmax > self.xlim[1]:
                self.xlim[1] = xmax
            if ymax > self.ylim[1]:
                self.ylim[1] = ymax
            if zmax > self.zlim[1]:
                self.zlim[1] = zmax
            # update mins
            if xmin < self.xlim[0]:
                self.xlim[0] = xmin
            if ymin < self.ylim[0]:
                self.ylim[0] = ymin
            if zmin < self.zlim[0]:
                self.zlim[0] = zmin
        # make axis actually squared
        if True:
            # smin = min(self.xlim[0],self.ylim[0],self.zlim[0])
            # smax = max(self.xlim[1],self.ylim[1],self.zlim[1])
            smin = min(self.xmin, self.ymin, self.zmin)
            smax = max(self.xmax, self.ymax, self.zmax)
            delta = 0.5 * (smax - smin)
            self.xlim = [cx - delta, cx + delta]
            self.ylim = [cy - delta, cy + delta]
            self.zlim = [cz - delta, cz + delta]
        self.axis_computed = True

    def plot_refresh(self, lock):
        if kVerbose:
            print(mp.current_process().name, "refreshing ", self.title)
        lock.acquire()
        if self.is_running.value == 1:
            self.setAxis()
        # if not kUseFigCanvasDrawIdle:
        if kUsePlotPause:
            plt.pause(kPlotSleep)
        lock.release()
