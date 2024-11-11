import torch.multiprocessing as mp
import threading as th    
import numpy as np

from utils_sys import MultiprocessingManager

# empty a queue before exiting from the consumer thread/process for safety
def empty_queue(queue):
    while queue.qsize()>0:
        try:
            queue.get(timeout=0.001)
        except:
            pass

class Value:
    def __init__(self, type, value):
        self.type = type
        self.value = value


class SingletonMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
    
    
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