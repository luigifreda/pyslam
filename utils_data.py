import torch.multiprocessing as mp
import threading as th    

import traceback

from utils_sys import MultiprocessingManager, Printer


kPrintTrackebackDetails = True

# empty a queue before exiting from the consumer thread/process for safety
def empty_queue(queue):
    try:
        while queue.qsize()>0:
            queue.get(timeout=0.001)
    except Exception as e:
        Printer.red(f'EXCEPTION in empty_queue: {e}')
        if kPrintTrackebackDetails:
            traceback_details = traceback.format_exc()
            print(f'\t traceback details: {traceback_details}')

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
    
    
class FixedSizeQueue:
    def __init__(self, maxsize):
        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue() 
        self.maxsize = maxsize
        self.size = mp.Value('i', 0) 

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
        
        
class AtomicCounter:
    def __init__(self):
        self.value = 0
        self._lock = th.Lock()

    def increment(self):
        with self._lock:
            self.value += 1

    def value(self):
        with self._lock:
            return self.value
    
    def increment_and_get(self):
        with self._lock:
            self.value += 1
            return self.value
    
    
