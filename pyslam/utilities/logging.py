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
    def get_instance(cls, *args, **kwargs):
        # Create a key from the arguments passed to the constructor
        key = (tuple(args), tuple(sorted(kwargs.items())))
        if key not in cls._instances:
            # If no instance exists with these arguments, create one
            instance = cls(*args, **kwargs)
            cls._instances[key] = instance
        return cls._instances[key]


class LoggerQueue(SingletonBase):
    """
    A class to manage process-safe logging using a shared Queue and QueueListener.
    Automatically starts the listener on initialization and stops it cleanly on exit.
    """

    def __init__(
        self,
        log_file,
        level=logging.INFO,
        formatter=Logging.simple_log_formatter,
        datefmt="",
        start_listener: bool = True,
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

        # Register a global cleanup function to stop all LoggerQueue instances
        # This ensures all instances are cleaned up even if atexit handlers
        # are called in an unexpected order
        if not hasattr(LoggerQueue, "_global_cleanup_registered"):
            atexit.register(LoggerQueue.stop_all_instances)
            LoggerQueue._global_cleanup_registered = True

    @classmethod
    def stop_all_instances(cls):
        """
        Stop all LoggerQueue instances. This is useful for ensuring clean shutdown
        of all logging threads before program exit. Called automatically at exit.
        """
        # Stop all instances, even if some are already closing
        # This ensures we clean up all queues and threads
        for instance in cls._instances.values():
            if isinstance(instance, cls) and hasattr(instance, "stop_listener"):
                try:
                    # Call stop_listener even if already closing - it's idempotent
                    # This ensures queues are properly closed and feeder threads are cancelled
                    instance.stop_listener()
                except Exception as e:
                    print(f"Warning: Error stopping LoggerQueue instance: {e}")

        # Give a small moment for threads to exit after cleanup
        # This is especially important for QueueFeederThread threads
        time.sleep(0.1)

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
    #     Destructor to stop the logging listener safely. Not needed anymore since we have atexit.register(self.stop_listener)
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
        print(f"LoggerQueue[{self.log_file}]: process: {process_name}, stopping listener...")
        try:
            # First, prevent new log records from being added to the queue
            # by removing QueueHandlers from all loggers that use this queue
            # This ensures no new items are put into the queue during shutdown
            import logging

            root_logger = logging.getLogger()
            for logger in [root_logger] + list(logging.Logger.manager.loggerDict.values()):
                if isinstance(logger, logging.Logger):
                    # Remove QueueHandlers that use our queue
                    handlers_to_remove = []
                    for handler in logger.handlers[
                        :
                    ]:  # Copy list to avoid modification during iteration
                        if isinstance(handler, QueueHandler):
                            # Check if this handler uses our queue
                            if hasattr(handler, "queue") and handler.queue is self.log_queue:
                                handlers_to_remove.append(handler)
                    for handler in handlers_to_remove:
                        try:
                            logger.removeHandler(handler)
                        except Exception:
                            pass

            # Stop the listener - ensure the thread exits before closing the queue
            if self.listener:
                try:
                    # Get reference to the listener thread before stopping
                    listener_thread = getattr(self.listener, "_thread", None)

                    # Call stop() to set the _stop flag - this tells the thread to exit
                    self.listener.stop()

                    # Wait for the listener thread to actually finish
                    # The thread will exit after it gets the next item from the queue and checks the flag
                    if listener_thread and listener_thread.is_alive():
                        # Put a sentinel to wake up the blocked thread if it's waiting
                        if self.log_queue and not getattr(self.log_queue, "_closed", False):
                            try:
                                # Put a sentinel LogRecord to wake up the thread
                                # The thread will process it, check the _stop flag, and exit cleanly
                                import logging

                                sentinel_record = logging.LogRecord(
                                    name="",
                                    level=0,
                                    pathname="",
                                    lineno=0,
                                    msg="",
                                    args=(),
                                    exc_info=None,
                                )
                                self.log_queue.put_nowait(sentinel_record)
                            except Exception as e:
                                print(
                                    f"LoggerQueue[{self.log_file}]: Error putting sentinel record: {e}"
                                )
                                print(f"\t traceback details: {traceback.format_exc()}")

                        # Wait for thread to finish (with longer timeout to allow clean shutdown)
                        # The thread should exit after processing the sentinel and checking _stop flag
                        listener_thread.join(timeout=2.0)
                        if listener_thread.is_alive():
                            # Thread didn't exit in time - might be stuck
                            # We'll close the queue anyway, which may cause an exception in the thread
                            # but since it's a daemon thread, it won't block shutdown
                            listener_thread.terminate()
                    elif listener_thread is None:
                        # Thread attribute not found - might be using a different QueueListener implementation
                        # Give some time for the listener to stop
                        time.sleep(0.5)
                except Exception as e:
                    print(f"LoggerQueue[{self.log_file}]: Error stopping listener: {e}")
                    print(f"\t traceback details: {traceback.format_exc()}")
                finally:
                    self.listener = None

            # Close the queue properly - only after listener is stopped
            # The proper order is:
            # 1. Close the queue (signals no more data will be put)
            # 2. Join the feeder thread (waits for it to finish processing)
            # 3. Only cancel if join times out
            if self.log_queue:
                try:
                    # Check if queue is already closed
                    if not getattr(self.log_queue, "_closed", False):
                        # Step 1: Close the queue first
                        # This signals that no more items will be put into the queue
                        # and allows the feeder thread to finish processing buffered items
                        self.log_queue.close()

                        # Step 2: Wait for the feeder thread to finish
                        # This ensures all buffered items are flushed to the underlying pipe
                        # The feeder thread should exit after processing all items
                        try:
                            # Use timeout to avoid hanging indefinitely
                            # timeout parameter was added in Python 3.9+
                            if hasattr(self.log_queue, "join_thread"):
                                try:
                                    # Wait up to 2 seconds for feeder thread to finish
                                    self.log_queue.join_thread(timeout=2.0)
                                except TypeError:
                                    # Timeout parameter not supported (Python < 3.9)
                                    # Try without timeout - this may block, so we'll cancel if needed
                                    self.log_queue.join_thread()
                        except Exception as join_error:
                            # If join fails or times out, cancel the join
                            # This tells the queue not to wait for the feeder thread
                            try:
                                if hasattr(self.log_queue, "cancel_join_thread"):
                                    self.log_queue.cancel_join_thread()
                                    print(
                                        f"LoggerQueue[{self.log_file}]: Feeder thread join timed out or failed, cancelled join"
                                    )
                            except Exception:
                                pass  # cancel_join_thread may not be available
                    else:
                        # Queue already closed, but we may still need to join or cancel
                        try:
                            if hasattr(self.log_queue, "join_thread"):
                                try:
                                    # Try to join with timeout
                                    self.log_queue.join_thread(timeout=1.0)
                                except (TypeError, Exception):
                                    # If join fails, cancel it
                                    if hasattr(self.log_queue, "cancel_join_thread"):
                                        self.log_queue.cancel_join_thread()
                        except Exception:
                            pass

                except Exception as e:
                    print(f"LoggerQueue[{self.log_file}]: Error closing queue: {e}")
                    print(f"\t traceback details: {traceback.format_exc()}")

            # Close the file handler
            if self.file_handler:
                try:
                    self.file_handler.close()
                except Exception as e:
                    print(f"LoggerQueue[{self.log_file}]: Error closing file handler: {e}")
                    print(f"\t traceback details: {traceback.format_exc()}")

        except Exception as e:
            print(
                f"LoggerQueue[{self.log_file}]: process: {process_name}, Exception during stop: {e}"
            )
            print(f"\t traceback details: {traceback.format_exc()}")

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


# new experimental logger queue
# it has some issues with the multiprocessing queue and the QueueListener
class LoggerQueueEXP(SingletonBase):
    """
    Process-safe logging using a multiprocessing queue and a QueueListener.

    IMPORTANT DESIGN:
      - Create ONE LoggerQueue in the MAIN process with start_listener=True.
      - Pass logger_queue.log_queue to worker processes and build LoggerQueue there
        with start_listener=False (worker mode).

    This prevents multiple processes from writing to the same file handler and
    prevents worker processes from truncating the log file.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        *,
        log_queue: Optional[mp.Queue] = None,
        level: int = logging.INFO,
        formatter: Optional[logging.Formatter] = Logging.simple_log_formatter,
        datefmt: str = "",
        truncate: bool = True,
        start_listener: bool = True,
    ):
        self.log_file = log_file
        self.level = level
        self.is_closing = False

        self.formatter = formatter or logging.Formatter(
            "%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
            datefmt=datefmt,
        )

        # Queue: either use provided shared queue or create a new one.
        # (Workers should receive an existing queue from the main process.)
        self.log_queue = log_queue if log_queue is not None else mp.Queue()

        # Listener/file handler exist only in "listener mode"
        self.file_handler: Optional[logging.Handler] = None
        self.listener: Optional[QueueListener] = None

        self._is_main_process = mp.current_process().name == "MainProcess"

        if start_listener:
            if not self._is_main_process:
                # You *can* force a listener in a worker, but it usually breaks shared-file logging.
                # We prevent it by default because it’s the common footgun.
                raise RuntimeError(
                    "start_listener=True in a non-main process. "
                    "Create/start the listener only in the main process, "
                    "and pass the queue to workers with start_listener=False."
                )
            if not log_file:
                raise ValueError("log_file must be provided when start_listener=True")

            if truncate:
                self.reset_log_file(log_file)

            self.file_handler = logging.FileHandler(log_file)
            self.file_handler.setLevel(level)
            self.file_handler.setFormatter(self.formatter)

            # respect_handler_level=True means the handler's level is respected for records
            self.listener = QueueListener(
                self.log_queue, self.file_handler, respect_handler_level=True
            )

            try:
                self.listener.start()
            except Exception as e:
                # Avoid using logging here (it might depend on this infrastructure)
                print(f"LoggerQueue[{log_file}]: Error starting listener: {e}")
                raise

            atexit.register(self.stop_listener)

    @staticmethod
    def reset_log_file(log_file: str) -> None:
        """Truncate the log file safely (main process only)."""
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", encoding="utf-8"):
                pass
            print(f"LoggerQueue[{log_file}]: Log file reset.")
        except Exception as e:
            print(f"LoggerQueue[{log_file}]: Error resetting log file: {e}")
            raise

    def stop_listener(self) -> None:
        """
        Stop the QueueListener and close the file handler.
        Idempotent and safe to call multiple times.

        Note: We do NOT force-close the multiprocessing queue here in order to avoid
        issues if other processes are still alive and logging during shutdown.
        """
        if self.is_closing:
            return
        self.is_closing = True

        proc = mp.current_process().name
        lf = self.log_file or "<no-file>"
        print(f"LoggerQueue[{lf}]: process: {proc}, stopping listener...")

        # Stop listener first
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception as e:
                print(f"LoggerQueue[{lf}]: Error stopping listener: {e}")
            finally:
                self.listener = None

        # Close file handler
        if self.file_handler is not None:
            try:
                self.file_handler.close()
            except Exception as e:
                print(f"LoggerQueue[{lf}]: Error closing file handler: {e}")
            finally:
                self.file_handler = None

        print(f"LoggerQueue[{lf}]: process: {proc}, stopped.")

    def get_logger(self, name: Optional[str] = None, *, propagate: bool = False) -> logging.Logger:
        """
        Return a logger configured to send records to the shared queue.

        - Adds exactly one QueueHandler (won’t duplicate on repeated calls).
        - Sets logger level and propagation policy.
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.level)
        logger.propagate = propagate

        # Add QueueHandler only once
        if not any(isinstance(h, QueueHandler) for h in logger.handlers):
            qh = QueueHandler(self.log_queue)
            qh.setLevel(self.level)
            logger.addHandler(qh)

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
