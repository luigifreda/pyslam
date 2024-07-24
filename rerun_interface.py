import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb



class Rerun:
    def __init__(self) -> None:
        rr.init("pyslam_rerun")
        rr.connect()  # Connect to a remote viewer
        
    @staticmethod
    def init() -> None:
        rr.init("pyslam")
        rr.connect()  # Connect to a remote viewer
        rr.spawn()
        
      
    @staticmethod
    def log_2dplot_seq_scalar_data(topic: str, seqId: int, scalar_data) -> None:
        rr.set_time_sequence("seqId", seqId)
        rr.log(topic, rr.Scalar(scalar_data))
        
    @staticmethod
    def log_2dplot_time_scalar_data(topic: str, time_nanos, scalar_data) -> None:
        rr.set_time_nanos("time", time_nanos)
        rr.log(topic, rr.Scalar(scalar_data))   
        
    @staticmethod
    def log_img_seq(topic: str, seqId: int, img, adjust_rgb=True) -> None:
        if adjust_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rr.set_time_sequence("seqId", seqId)
        rr.log(topic, rr.Image(img))
                
    @staticmethod
    def log_img_time(topic: str, time_nanos, img, adjust_rgb=True) -> None:
        if adjust_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        rr.set_time_nanos("time", time_nanos)
        rr.log(topic, rr.Image(img))

   