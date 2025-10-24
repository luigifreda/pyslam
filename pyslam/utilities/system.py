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


def getchar():
    print("press enter to continue:")
    a = input("").split(" ")[0]
    print(a)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False


# Class to print
# colored text and background
# from https://www.geeksforgeeks.org/print-colors-python-terminal/
class TerminalColors:
    """
    TerminalColors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold
    """

    reset = "\033[0m"
    bold = "\033[01m"
    disable = "\033[02m"
    underline = "\033[04m"
    reverse = "\033[07m"
    strikethrough = "\033[09m"
    invisible = "\033[08m"

    class fg:
        black = "\033[30m"
        red = "\033[31m"
        green = "\033[32m"
        orange = "\033[33m"
        blue = "\033[34m"
        purple = "\033[35m"
        cyan = "\033[36m"
        lightgrey = "\033[37m"
        darkgrey = "\033[90m"
        lightred = "\033[91m"
        lightgreen = "\033[92m"
        yellow = "\033[93m"
        lightblue = "\033[94m"
        pink = "\033[95m"
        lightcyan = "\033[96m"

    class bg:
        black = "\033[40m"
        red = "\033[41m"
        green = "\033[42m"
        orange = "\033[43m"
        blue = "\033[44m"
        purple = "\033[45m"
        cyan = "\033[46m"
        lightgrey = "\033[47m"


class Printer(object):
    @staticmethod
    def red(*args, **kwargs):
        print(TerminalColors.fg.red, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def green(*args, **kwargs):
        print(TerminalColors.fg.green, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def blue(*args, **kwargs):
        print(TerminalColors.fg.blue, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def lightblue(*args, **kwargs):
        print(TerminalColors.fg.lightblue, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def cyan(*args, **kwargs):
        print(TerminalColors.fg.cyan, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def orange(*args, **kwargs):
        print(TerminalColors.fg.orange, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def purple(*args, **kwargs):
        print(TerminalColors.fg.purple, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def yellow(*args, **kwargs):
        print(TerminalColors.fg.yellow, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def error(*args, **kwargs):
        print(TerminalColors.fg.red, *args, **kwargs, file=sys.stderr)
        print(TerminalColors.reset, end="")

    @staticmethod
    def warning(*args, **kwargs):
        print(TerminalColors.fg.yellow, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def info(*args, **kwargs):
        print(TerminalColors.fg.cyan, *args, **kwargs)
        print(TerminalColors.reset, end="")

    @staticmethod
    def bold(*args, **kwargs):
        print(TerminalColors.bold, *args, **kwargs)
        print(TerminalColors.reset, end="")

    def bold_blue(*args, **kwargs):
        print(f"{TerminalColors.bold}{TerminalColors.fg.blue}", *args, **kwargs)
        print(TerminalColors.reset, end="")

    def bold_green(*args, **kwargs):
        print(f"{TerminalColors.bold}{TerminalColors.fg.green}", *args, **kwargs)
        print(TerminalColors.reset, end="")

    def bold_purple(*args, **kwargs):
        print(f"{TerminalColors.bold}{TerminalColors.fg.purple}", *args, **kwargs)
        print(TerminalColors.reset, end="")

    def bold_cyan(*args, **kwargs):
        print(f"{TerminalColors.bold}{TerminalColors.fg.cyan}", *args, **kwargs)
        print(TerminalColors.reset, end="")


# test class with termcolor
class Printer_old(object):
    @staticmethod
    def red(input):
        print(colored(input, "red"))

    @staticmethod
    def green(input):
        print(colored(input, "green"))


# return a random RGB color tuple
def random_color():
    color = tuple(np.random.randint(0, 255, 3).tolist())
    return color


# for logging to multiple files, streams, etc.
class Logging(object):
    """
    A class for logging to multiple files, streams, etc.
    Example:
    # first file logger
    logger = Logging.setup_file_logger('first_logger', 'first_logfile.log')
    logger.info('This is just info message')

    # second file logger
    super_logger = Logging.setup_file_logger('second_logger', 'second_logfile.log')
    super_logger.error('This is an error message')
    """

    time_log_formatter = logging.Formatter("%(levelname)s[%(asctime)s] %(message)s")
    notime_log_formatter = logging.Formatter("%(levelname)s %(message)s")
    simple_log_formatter = logging.Formatter("%(message)s")
    thread_log_formatter = logging.Formatter("%(levelname)s] (%(threadName)-10s) %(message)s")

    @staticmethod
    def setup_logger(name, level=logging.INFO, formatter=time_log_formatter):  # to sys.stderr
        """To setup as many loggers as you want with a selected formatter"""
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def setup_file_logger(
        name, log_file, level=logging.INFO, mode="+w", formatter=time_log_formatter
    ):  # to file
        """To setup as many loggers as you want with a selected formatter"""
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        handler = logging.FileHandler(log_file, mode=mode)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def setup_socket_logger(
        name, host, port, level=logging.INFO, formatter=time_log_formatter
    ):  # TCP-IP
        """To setup as many loggers as you want with a selected formatter"""
        handler = logging.SocketHandler(host, port)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def setup_udp_logger(name, host, port, level=logging.INFO, formatter=time_log_formatter):  # UDP
        """To setup as many loggers as you want with a selected formatter"""
        handler = logging.DatagramHandler(host, port)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger


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


class LoggerQueue(SingletonBase):
    """
    A class to manage process-safe logging using a shared Queue and QueueListener.
    Automatically starts the listener on initialization and stops it cleanly on exit.
    """

    def __init__(
        self, log_file, level=logging.INFO, formatter=Logging.simple_log_formatter, datefmt=""
    ):

        # Reset the log file (clear its contents) before initializing the logger
        self.reset_log_file(log_file)

        self.log_file = log_file
        self.level = level
        self.formatter = formatter or logging.Formatter(
            "%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
            datefmt=datefmt,
        )

        # Shared log queue
        self.log_queue = mp.Queue()

        # File handler for the listener
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setFormatter(self.formatter)

        # Queue listener
        self.listener = QueueListener(self.log_queue, self.file_handler)

        # Start the listener
        try:
            self.listener.start()
        except Exception as e:
            print(f"LoggerQueue[{self.log_file}]: Error starting listener: {e}")
        # print(f"LoggerQueue[{self.log_file}]: initialized and started.")

        self.is_closing = False

        # Register stop_listener to be called at program exit
        atexit.register(self.stop_listener)

    def reset_log_file(self, log_file):
        """
        Clears the contents of the log file to reset it at the beginning.
        """
        try:
            with open(log_file, "w"):  # Open the file in write mode, which clears it
                pass
            print(f"LoggerQueue[{log_file}]: Log file reset.")
        except Exception as e:
            print(f"LoggerQueue[{log_file}]: Error resetting log file: {e}")

    # def __del__(self):
    #     """
    #     Destructor to stop the logging listener safely.
    #     """
    #     self.stop_listener()

    def stop_listener(self):
        """
        Stops the QueueListener and flushes the log queue.
        Ensures the resources are properly released.
        """
        if self.is_closing:
            return
        self.is_closing = True
        process_name = mp.current_process().name
        # print(f"LoggerQueue[{self.log_file}]: process: {process_name}, stopping ...")
        try:
            # Stop the listener with timeout
            if hasattr(self, "listener") and self.listener:
                try:
                    self.listener.stop()
                except Exception as e:
                    print(f"LoggerQueue[{self.log_file}]: Error stopping listener: {e}")
                finally:
                    self.listener = None

            # Close the queue properly
            if hasattr(self, "log_queue") and self.log_queue:
                try:
                    self.log_queue.close()
                    self.log_queue.join_thread()
                except Exception as e:
                    print(f"LoggerQueue[{self.log_file}]: Error closing queue: {e}")

            # Close the file handler
            if hasattr(self, "file_handler") and self.file_handler:
                try:
                    self.file_handler.close()
                except Exception as e:
                    print(f"LoggerQueue[{self.log_file}]: Error closing file handler: {e}")

        except Exception as e:
            print(
                f"LoggerQueue[{self.log_file}]: process: {process_name}, Exception during stop: {e}"
            )

        print(f"LoggerQueue[{self.log_file}]: process: {process_name}, stopped.")

    def get_logger(self, name=None):
        """
        Create and return a logger configured to use the shared Queue.

        :param name: Optional logger name.
        :return: Logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.level)

        # Attach a QueueHandler to direct logs to the shared queue
        if not any(isinstance(h, QueueHandler) for h in logger.handlers):
            logger.addHandler(QueueHandler(self.log_queue))

        return logger


# An implementation of a thread- and process-safe file logger
class FileLogger:
    kSafetyLockingTimeout = 0.05

    def __init__(
        self, log_file, level=logging.INFO, formatter=Logging.simple_log_formatter, datefmt=""
    ):
        """
        Initializes a thread- and process-safe logger.

        :param log_file: Path to the log file.
        """
        self.log_file = log_file
        self._process_lock = mp.Lock()
        self._thread_lock = threading.Lock()

        self.reset_log_file(log_file)
        self.log_file = log_file
        self.level = level
        self.formatter = formatter or logging.Formatter(
            "%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
            datefmt=datefmt,
        )

        self._logger = logging.getLogger(log_file)
        self._logger.setLevel(level)

        # Prevent duplicate handlers if the logger is reused.
        if not self._logger.hasHandlers():
            file_handler = logging.FileHandler(log_file)
            # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(self.formatter)
            self._logger.addHandler(file_handler)

        atexit.register(self.close)

    def close(self):
        """Close the logger and release system resources."""
        print(f"FileLogger[{self.log_file}]: closing...")
        if self._logger:
            self._logger.removeHandler(self._logger.handlers[0])
            for h in self._logger.handlers:
                h.close()
            self._logger = None

    def __del__(self):
        self.close()
        print(f"FileLogger[{self.log_file}]: deleted.")

    def reset_log_file(self, log_file):
        """
        Clears the contents of the log file to reset it at the beginning.
        """
        try:
            with open(log_file, "w"):  # Open the file in write mode, which clears it
                pass
            print(f"FileLogger[{log_file}]: Log file reset.")
        except Exception as e:
            print(f"FileLogger[{log_file}]: Error resetting log file: {e}")

    def log(self, level, message):
        """
        Logs a message at the specified level in a thread- and process-safe manner.

        :param level: Logging level (e.g., logging.INFO, logging.ERROR).
        :param message: The log message.
        """
        # with self._thread_lock:
        #     with self._process_lock:
        #         self._logger.log(level, message)

        try:
            if self._thread_lock.acquire(
                timeout=FileLogger.kSafetyLockingTimeout
            ):  # Acquire the thread lock
                try:
                    if self._process_lock.acquire(
                        timeout=FileLogger.kSafetyLockingTimeout
                    ):  # Acquire the process lock
                        try:
                            self._logger.log(level, message)  # Perform logging
                        finally:
                            self._process_lock.release()  # Always release the process lock
                    else:
                        print(
                            f"FileLogger: ERROR while logging: could not acquire process lock in {FileLogger.kSafetyLockingTimeout} seconds"
                        )
                finally:
                    self._thread_lock.release()  # Always release the thread lock
            else:
                print(
                    f"FileLogger: ERROR while logging: could not acquire thread lock in {FileLogger.kSafetyLockingTimeout} seconds"
                )
        except Exception as e:
            # Handle logging exceptions or re-raise
            print(f"FileLogger: ERROR while logging: {e}")

    def info(self, message):
        """Logs an info message."""
        self.log(logging.INFO, message)

    def debug(self, message):
        """Logs a debug message."""
        self.log(logging.DEBUG, message)

    def warning(self, message):
        """Logs a warning message."""
        self.log(logging.WARNING, message)

    def error(self, message):
        """Logs an error message."""
        self.log(logging.ERROR, message)

    def critical(self, message):
        """Logs a critical message."""
        self.log(logging.CRITICAL, message)


# This function check and exec 'from module import name' and directly return the 'name'.'method'.
# The method is used to directly return a 'method' of 'name' (i.e. 'module'.'name'.'method')
# N.B.: if a method is needed you CAN'T
# from module import name.method
# since method is an attribute of name!
def import_from(module, name, method=None, debug=False):
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


def print_options(opt, opt_name="OPTIONS"):
    content_list = []
    args = list(vars(opt))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, " ") + "  " + str(getattr(opt, arg))]
    print_notification(content_list, opt_name)


def print_notification(content_list, notifi_type="NOTIFICATION"):
    print(("---------------------- {0} ----------------------".format(notifi_type)))
    print()
    for content in content_list:
        print(content)
    print()
    print("----------------------------------------------------")


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
    os._exit(code)  # Bypass cleanup and exit immediately
