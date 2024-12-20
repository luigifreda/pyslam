import time
from multiprocessing import Process

import sys 
sys.path.append("../../")
from config import Config

from utils_sys import LoggerQueue


def worker_task(logging_manager):
    # Get a logger for the worker process
    logger = logging_manager.get_logger("Worker")
    logger.info("Worker process has started.")
    time.sleep(1)
    logger.info("Worker process has finished.")


if __name__ == "__main__":
    log_file = "process_safe_log.log"
    
    # Initialize the LoggingManager (automatically starts the listener)
    logging_manager = LoggerQueue(log_file)
    
    # Get a logger for the main process
    main_logger = logging_manager.get_logger("MainThread")
    main_logger.info("Main thread has started.")
    
    # Start a worker process
    worker = Process(target=worker_task, args=(logging_manager,))
    worker.start()
    worker.join()
    
    main_logger.info("Main thread has finished.")
    
    time.sleep(1)