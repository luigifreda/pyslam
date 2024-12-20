import time
import multiprocessing as mp

import sys 
sys.path.append("../../")
from config import Config

from utils_sys import LoggerQueue


logging_file='test.log'
logging_manager = LoggerQueue.get_instance(logging_file)
logger = logging_manager.get_logger("main")

def print(*args, **kwargs):
    return logger.info(*args, **kwargs)  


def worker_task(logging_manager):
    # Get a logger for the worker process
    print("Worker process has started.")
    time.sleep(1)
    print("Worker process has finished.")


if __name__ == "__main__":
        
    print("Main thread has started.")
    
    # Start a worker process
    worker = mp.Process(target=worker_task, args=(logging_manager,), name="process")
    worker.start()
    worker.join()
    
    print("Main thread has finished.")
    
    time.sleep(1)