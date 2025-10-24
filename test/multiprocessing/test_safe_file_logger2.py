import logging
import os
from multiprocessing import Lock, Process
from threading import Lock as ThreadLock, Thread

import sys

from pyslam.config import Config

import time
from pyslam.utilities.system import FileLogger, LoggerQueue

if False:
    # NOTE: This seems to fail once in while
    logger = FileLogger("test.log")

    def print(*args, **kwargs):
        message = " ".join(
            str(arg) for arg in args
        )  # Convert all arguments to strings and join with spaces
        return logger.info(message, **kwargs)

else:
    logging_manager = LoggerQueue.get_instance("test.log")
    logger = logging_manager.get_logger("test")

    def print(*args, **kwargs):
        message = " ".join(
            str(arg) for arg in args
        )  # Convert all arguments to strings and join with spaces
        return logger.info(message, **kwargs)


# Function to log messages in threads
def thread_logging(thread_id):
    for i in range(2):
        print(f"Thread {thread_id} - Message {i}")
        time.sleep(1)


# Function to log messages in processes
def process_logging(process_id):
    for i in range(2):
        print(f"Process {process_id} - Message {i}")
        time.sleep(1)


# Usage example with threads and processes
if __name__ == "__main__":

    # Creating threads
    threads = [Thread(target=thread_logging, args=(i,)) for i in range(5)]

    # Creating processes
    processes = [Process(target=process_logging, args=(i,)) for i in range(5)]

    print("Start")

    # Starting threads
    for thread in threads:
        thread.start()

    # Starting processes
    for process in processes:
        process.start()

    # Joining threads
    for thread in threads:
        thread.join()

    # Joining processes
    for process in processes:
        process.join()

    # Closing the logger
    # logger.close()
    print("Done")

    # del logger
