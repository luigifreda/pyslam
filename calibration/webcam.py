import numpy as np
import cv2
import time

from multiprocessing import Process, Queue

class Webcam(object):
    def __init__(self, camera_num=0):
        self.cap = cv2.VideoCapture(camera_num)
        self.current_frame = None 
        self.ret = None 
        self.q = Queue(maxsize=2)        
        self.vp = Process(target=self._update_frame, args=(self.q,))
        self.vp.daemon = True

    # create thread for capturing images
    def start(self):
        self.vp.start()        
        
    # process function     
    def _update_frame(self, q):
        while(True):
            self.ret, self.current_frame = self.cap.read()
            if self.ret is True: 
                #self.current_frame= self.cap.read()[1]
                #if q.full():
                #    old_frame = self.q.get()
                self.q.put(self.current_frame)
                print('q.size: ', self.q.qsize())
        time.sleep(0.005)
                  
    # get the current frame
    def get_current_frame(self):
        img = None 
        while not self.q.empty():  # get the last one 
            img = self.q.get()         
        return img