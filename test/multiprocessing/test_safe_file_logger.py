import logging
import os
from multiprocessing import Lock, Process
from threading import Lock as ThreadLock, Thread

import sys 
sys.path.append("../../")
from pyslam.config import Config

import time
from pyslam.utilities.utils_sys import FileLogger, LoggerQueue



# Function to log messages in threads
def thread_logging(logger, thread_id):
    for i in range(2):
        logger.info(f"Thread {thread_id} - Message {i}")
        time.sleep(1)


# Function to log messages in processes
def process_logging(logger, process_id):
    for i in range(2):
        logger.info(f"Process {process_id} - Message {i}")
        time.sleep(1)


# Usage example with threads and processes
if __name__ == "__main__":
    
    if False:
        logger = FileLogger("test.log")
    else: 
        logging_manager = LoggerQueue.get_instance("test.log")
        logger = logging_manager.get_logger("test")

    # Creating threads
    threads = [Thread(target=thread_logging, args=(logger, i)) for i in range(5)]

    # Creating processes
    processes = [Process(target=process_logging, args=(logger, i)) for i in range(5)]

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
    #logger.close()
    print('Done')