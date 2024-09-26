import math 
import numpy as np 

from utils_features import descriptor_sigma_mad
from parameters import Parameters 
from frame import Frame   

# experimental 
class SLAMDynamicConfig(object):
    def __init__(self):
        
        self.descriptor_distance_sigma=Parameters.kMaxDescriptorDistance   # Note: Parameters.kMaxDescriptorDistance is initialized by the feature manager  
        self.descriptor_distance_alpha=0.9
        self.descriptor_distance_factor=3
        self.descriptor_distance_max_delta_fraction = 0.5
        self.descriptor_distance_min = Parameters.kMaxDescriptorDistance * (1. - self.descriptor_distance_max_delta_fraction)
        self.descriptor_distance_max = Parameters.kMaxDescriptorDistance * (1. + self.descriptor_distance_max_delta_fraction)
        
        self.reproj_err_frame_map_sigma=Parameters.kMaxReprojectionDistanceMap
        self.reproj_err_frame_map_alpha=0.9  
        self.reproj_err_frame_map_factor=3                
        
    def update_descriptor_stat(self, f_ref, f_cur, idxs_ref, idxs_cur):  
        if len(idxs_cur)>0:
            des_cur = f_cur.des[idxs_cur]
            des_ref = f_ref.des[idxs_ref]
            sigma_mad,_ = descriptor_sigma_mad(des_cur, des_ref, descriptor_distances=Frame.descriptor_distances)
            delta = self.descriptor_distance_factor*sigma_mad
            if self.descriptor_distance_sigma is None:
                self.descriptor_distance_sigma = delta
            else:
                self.descriptor_distance_sigma = self.descriptor_distance_alpha*self.descriptor_distance_sigma + (1.-self.descriptor_distance_alpha)*delta
                self.descriptor_distance_sigma = max(min(self.descriptor_distance_sigma, self.descriptor_distance_max), self.descriptor_distance_min)
            print('descriptor sigma: ', self.descriptor_distance_sigma)
        else:
            self.descriptor_distance_sigma = 0
        return self.descriptor_distance_sigma
    
    def update_reproj_err_map_stat(self, value):
        self.reproj_err_frame_map_sigma = self.reproj_err_frame_map_alpha*self.reproj_err_frame_map_sigma + (1.-self.reproj_err_frame_map_alpha)*value
        self.reproj_err_frame_map_sigma = max(1., self.reproj_err_frame_map_sigma)
        return self.reproj_err_frame_map_sigma
        