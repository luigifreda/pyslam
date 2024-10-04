import multiprocessing as mp
import threading as th    

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