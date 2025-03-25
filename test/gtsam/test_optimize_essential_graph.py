import sys 
sys.path.append("../../")
import config

import numpy as np
import random

import unittest
from unittest import TestCase

import numpy as np

from frame import Frame, FeatureTrackerShared
from keyframe import KeyFrame
from map_point import MapPoint
from camera import Camera, PinholeCamera
from utils_geom import rotation_matrix_from_yaw_pitch_roll, pitch_matrix, poseRt, inv_T, Sim3Pose
from utils_gen_synthetic_data import generate_random_points_2d, backproject_points, project_points
from map import Map
from viewer3D import Viewer3D

from collections import defaultdict 
import time 

import optimizer_gtsam 
import optimizer_g2o


class DataGenerator:
    def __init__(self, n=100, radius=10.0, sigma_noise_xyz=0.001, sigma_noise_theta_deg=0.001):
        self.n = n
        self.radius = radius
        self.sigma_noise_xyz = sigma_noise_xyz
        self.sigma_noise_theta_rad = np.deg2rad(sigma_noise_theta_deg)
        
        self.map_obj = Map()
        self.keyframes = []
        self.loop_connections = defaultdict(set)
        self.non_corrected_sim3_map = {}
        self.corrected_sim3_map = {}        
        self.gt_poses = []
        
        seed = 0
        np.random.seed(seed) # make it deterministic
        random.seed(seed)
        
    def generate_loop_data(self):
        n = self.n
        radius = self.radius
        sigma_noise_xyz = self.sigma_noise_xyz
        sigma_noise_theta_rad = self.sigma_noise_theta_rad
        
        delta_angle =  2 * np.pi / n    
        omega = delta_angle    
        velocity = omega * radius

        x2d, y2d, theta = 0, 0, 0
        x2d_n, y2d_n, theta_n = x2d, y2d, theta
        
        for i in range(n+1):
            
            # in 3D with computer vision xyz coordinates (z along optical axis, x right, y down)        
            Rwc = pitch_matrix(-theta_n + np.pi/2)
            twc = np.array([x2d_n, 0, y2d_n])
            Rcw = Rwc.T
            tcw = -Rcw @ twc
            
            gt_Rwc = pitch_matrix(-theta + np.pi/2)
            gt_twc = np.array([x2d, 0, y2d])
            gt_Twc = poseRt(gt_Rwc, gt_twc)
            self.gt_poses.append(gt_Twc)
            
            f = Frame(camera=None, img=None)
            f.update_pose(poseRt(Rcw, tcw))
            
            kf = KeyFrame(frame=f)
            self.keyframes.append(kf)
            if i > 0:
                kf_prev = self.keyframes[i - 1]
                kf.add_connection(kf_prev, 1000)    # fake weight        
                kf.set_parent(kf_prev)
                if i > 1:
                    kf_prev2 = self.keyframes[i - 2]
                    kf.add_connection(kf_prev2, 1000)    # fake weight
            
            self.map_obj.add_keyframe(kf)
            
            # update status 
            
            # 2D classic x,y,theta no noise
            theta += delta_angle                     
            x2d += velocity * np.cos(theta)
            y2d += velocity * np.sin(theta)      
                        
            # 2D classic x,y,theta with noise
            theta_n += delta_angle + random.gauss(0, sigma_noise_theta_rad)            
            x2d_n += velocity * np.cos(theta_n) + random.gauss(0, sigma_noise_xyz) 
            y2d_n += velocity * np.sin(theta_n) + random.gauss(0, sigma_noise_xyz)

            
        last_kf = self.keyframes[-1]
        first_kf = self.keyframes[0]
        last_kf.add_connection(first_kf, 1000)    # fake weight    
        
        
    def add_loop_closure(self): 
        # let's add a loop closure for the loop: last_keyframe -> first_keyframe
        last_kf = self.keyframes[-1]
        first_kf = self.keyframes[0]
                
        # retrieve keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
        current_keyframe = last_kf
        current_keyframe.update_connections()
        
        current_connected_keyframes = current_keyframe.get_connected_keyframes()
        print(f'current keyframe: {current_keyframe.id}, connected keyframes: {[k.id for k in current_connected_keyframes]}')
        current_connected_keyframes.append(current_keyframe)  
        
        # let's use the ground truth as loop closure correction 
        last_gt_Twc = self.gt_poses[-1].copy()
        last_gt_Tcw = inv_T(last_gt_Twc)
        last_Scw = Sim3Pose(last_gt_Tcw[:3, :3], last_gt_Tcw[:3, 3], 1.0)
                
        self.corrected_sim3_map[current_keyframe] = last_Scw
                
        # Iterate over all current connected keyframes and propagate the sim3 correction obtained on current keyframe
        for connected_kfi in current_connected_keyframes:
            Tiw = connected_kfi.Tcw

            if connected_kfi != current_keyframe:
                Tic = Tiw @ last_gt_Twc
                Ric = Tic[:3, :3]
                tic = Tic[:3, 3]
                Sic = Sim3Pose(Ric, tic, 1.0)
                corrected_Siw = Sic @ last_Scw
                # Pose corrected with the Sim3 of the loop closure
                self.corrected_sim3_map[connected_kfi] = corrected_Siw

            Riw = Tiw[:3, :3]
            tiw = Tiw[:3, 3]
            Siw = Sim3Pose(Riw, tiw, 1.0)
            # Pose without correction
            self.non_corrected_sim3_map[connected_kfi] = Siw
            
        # Create a dictionary where each key is a KeyFrame and the value is a set of connected KeyFrames
        loop_connections = defaultdict(set)

        for kfi in current_connected_keyframes:
            # Get previous neighbors (covisible keyframes)
            previous_neighbors = kfi.get_covisible_keyframes()

            # Update connections and get the new ones
            kfi.update_connections()
            loop_connections[kfi] = set(kfi.get_connected_keyframes())

            # # Remove previous neighbors from connections
            # for previous_neighbor in previous_neighbors:
            #     try:
            #         loop_connections[kfi].remove(previous_neighbor)
            #     except:
            #         pass # not found

            # # Remove the current connected keyframes from the connection set
            # for other_kf in current_connected_keyframes:
            #     try:
            #         loop_connections[kfi].remove(other_kf)
            #     except: 
            #         pass # not found
        
        self.loop_keyframe = current_keyframe
        self.loop_connections = loop_connections
        self.current_keyframe = current_keyframe
        self.fix_scale=True

    def compute_ATE(self):
        error = 0
        for i,kf in enumerate(self.keyframes):
            gt_Twc = self.gt_poses[i]
            Twc = kf.Twc
            error += np.linalg.norm(gt_Twc[:3,3] - Twc[:3,3])**2
        return np.sqrt(error/len(self.keyframes))
    

def main():
    optimize_essential_graph = optimizer_gtsam.optimize_essential_graph
    #optimize_essential_graph = optimizer_g2o.optimize_essential_graph
        
    data_generator = DataGenerator(n=50, radius=5.0, sigma_noise_xyz=0.04, sigma_noise_theta_deg=0.1)
    data_generator.generate_loop_data()
    
    est_poses_before = []
    for kf in data_generator.keyframes:
        est_poses_before.append(kf.Twc.copy())

    ATE_before = data_generator.compute_ATE()
    print(f'ATE before: {ATE_before}')
    
    # add loop closure
    data_generator.add_loop_closure()    
    
    print(f'fix_scale: {data_generator.fix_scale}')
    print(f'first_keyframe: {data_generator.keyframes[0].id}, last_keyframe: {data_generator.keyframes[-1].id}, #keyframes: {len(data_generator.keyframes)}')
    print(f'loop_keyframe: {data_generator.loop_keyframe.id}')
    print(f'current_keyframe: {data_generator.current_keyframe.id}')
    print(f'loop_connections: {[kf.id for kf in data_generator.loop_connections]}')
    print(f'#corrected_sim3_map: {len(data_generator.corrected_sim3_map)}, #non_corrected_sim3_map: {len(data_generator.non_corrected_sim3_map)}')
        
    if True: 
        mse = optimize_essential_graph(
            data_generator.map_obj, data_generator.loop_keyframe, data_generator.current_keyframe,
            data_generator.non_corrected_sim3_map, data_generator.corrected_sim3_map,
            data_generator.loop_connections, data_generator.fix_scale, verbose=True
        )
        print("Optimization MSE:", mse)

    est_poses = []
    for kf in data_generator.keyframes:
        est_poses.append(kf.Twc)

    viewer3d = Viewer3D()
    if True:
        viewer3d.draw_cameras([data_generator.gt_poses, est_poses, est_poses_before], [[1,0,0], [0,1,0], [0,0,1]])
    else:
        viewer3d.draw_cameras([est_poses, data_generator.gt_poses], [[0,1,0],[1,0,0]])
    
    ATE_after = data_generator.compute_ATE()
    print(f'ATE after: {ATE_after}')
    
    while not viewer3d.is_closed():
        time.sleep(0.1)
            
            
if __name__ == "__main__":
    main()
