import torch.multiprocessing as mp
import threading as th    

import traceback
import platform

from utils_sys import Printer


kPrintTrackebackDetails = True


# empty a queue before exiting from the consumer thread/process for safety
def empty_queue(queue):
    if platform.system() == 'Darwin':
        try:             
            while not queue.empty():
                queue.get(timeout=0.001) 
        except Exception as e:
            Printer.red(f'EXCEPTION in empty_queue: {e}')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                print(f'\t traceback details: {traceback_details}')
    else:
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
    
    
class FixedSizeQueue:
    def __init__(self, maxsize):
        self.queue = mp.Queue() 
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
        

class SafeQueue:
    def __init__(self, maxsize=0):
        """
        A wrapper around multiprocessing.Queue with a custom qsize method.
        
        :param maxsize: The maximum size of the queue (default: 0 for unlimited size).
        """
        self.queue = mp.Queue(maxsize)
        self._size = mp.Value('i', 0)  # Shared integer to track the size of the queue
        self._lock = mp.Lock()  # Lock to ensure thread-safe operations

    def put(self, item, block=True, timeout=None):
        """Put an item into the queue."""
        with self._lock:
            self._size.value += 1
        self.queue.put(item, block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue."""
        item = self.queue.get(block=block, timeout=timeout)
        with self._lock:
            self._size.value -= 1
        return item

    def qsize(self):
        """Return the current size of the queue."""
        with self._lock:
            return self._size.value

    def empty(self):
        """Check if the queue is empty."""
        with self._lock:
            return self._size.value == 0

    def full(self):
        """Check if the queue is full."""
        with self._lock:
            if self.queue._maxsize > 0:
                return self._size.value >= self.queue._maxsize
            return False

    def close(self):
        """Close the underlying queue."""
        self.queue.close()

    def join_thread(self):
        """Join the queue's worker thread."""
        self.queue.join_thread()

    def cancel_join_thread(self):
        """Cancel the queue's worker thread."""
        self.queue.cancel_join_thread()