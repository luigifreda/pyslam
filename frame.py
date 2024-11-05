"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import cv2
import numpy as np
import json
#import g2o

from threading import RLock, Thread, current_thread
from scipy.spatial import cKDTree

from dataset import SensorType
from parameters import Parameters  

from camera import Camera, PinholeCamera
from camera_pose import CameraPose

from utils_geom import add_ones, poseRt, normalize, triangulate_points, triangulate_normalized_points
from utils_sys import myjet, Printer

from feature_types import FeatureInfo
from concurrent.futures import ThreadPoolExecutor

from utils_draw import draw_feature_matches
from utils_features import compute_NSAD_between_matched_keypoints, descriptor_sigma_mad

kDrawFeatureRadius = [r*5 for r in range(1,100)]
kDrawOctaveColor = np.linspace(0, 255, 12)


# Base object class for frame info management; 
# it collects methods for managing:
# - camera intrinsics 
# - camera pose 
# - points projections 
# - checking points visibility 
class FrameBase(object):       
    _id = 0                     # shared frame counter    
    _id_lock = RLock()     
    def __init__(self, camera: Camera, pose=None, id=None, timestamp=None, img_id=None):
        self._lock_pose = RLock()
        # frame camera info
        self.camera = camera
        # self._pose is a CameraPose() representing Tcw (pc = Tcw * pw)
        if pose is None: 
            self._pose = CameraPose()      
        else: 
            self._pose = CameraPose(pose) 
        # frame id            
        if id is not None: 
            self.id = id 
        else: 
            with FrameBase._id_lock:
                self.id = FrameBase._id
                FrameBase._id += 1     
        # frame timestamp                
        self.timestamp = timestamp 
        self.img_id = img_id
        
    def __hash__(self):
        return self.id   
    
    def __eq__(self, rhs):
        return (isinstance(rhs, FrameBase) and self.id == rhs.id)
    
    def __lt__(self, rhs):
        return self.id < rhs.id 
     
    def __le__(self, rhs):
        return self.id <= rhs.id         
        
    @property
    def width(self):
        return self.camera.width
    
    @property
    def height(self):
        return self.camera.height    
                
    @property    
    def isometry3d(self):  # pose as g2o.Isometry3d 
        with self._lock_pose:          
            return self._pose.isometry3d
                    
    @property
    def Tcw(self):
        with self._lock_pose:           
            return self._pose.Tcw  
    @property
    def Twc(self):
        with self._lock_pose:           
            return self._pose.get_inverse_matrix()      
    @property
    def Rcw(self):
        with self._lock_pose:           
            return self._pose.Rcw       
    @property
    def Rwc(self):
        with self._lock_pose:           
            return self._pose.Rwc     
    @property
    def tcw(self):
        with self._lock_pose:           
            return self._pose.tcw       
    @property
    def Ow(self):
        with self._lock_pose:           
            return self._pose.Ow          
    @property
    def pose(self):
        with self._lock_pose:   
            return self._pose.Tcw      
    @property    
    def quaternion(self): # g2o.Quaternion(),  quaternion_cw  
        
        with self._lock_pose:           
            return self._pose.quaternion  
    @property    
    def orientation(self): # g2o.Quaternion(),  quaternion_cw  
        with self._lock_pose:           
            return self._pose.orientation  
                
    @property    
    def position(self):    # 3D vector tcw (world origin w.r.t. camera frame) 
        with self._lock_pose:           
            return self._pose.position    
               
    # update pose from transformation matrix or g2o.Isometry3d
    def update_pose(self, pose):
        with self._lock_pose:              
            self._pose.set(pose)    
   # update pose from transformation matrix 
    def update_translation(self, tcw):
        with self._lock_pose:              
            self._pose.set_translation(tcw)           
   # update pose from transformation matrix 
    def update_rotation_and_translation(self, Rcw, tcw):
        with self._lock_pose:          
            self._pose.set_from_rotation_and_translation(Rcw, tcw)            
                
    # transform a world point into a camera point 
    def transform_point(self, pw):
        with self._lock_pose:          
            return (self._pose.Rcw @ pw) + self._pose.tcw # p w.r.t. camera 
    # transform a world points into camera points [Nx3] 
    # out: points  w.r.t. camera frame  [Nx3] 
    def transform_points(self, points):    
        with self._lock_pose:          
            Rcw = self._pose.Rcw
            tcw = self._pose.tcw.reshape((3,1))              
            return (Rcw @ points.T + tcw).T  # get points  w.r.t. camera frame  [Nx3]      
    
    # project an [Nx3] array of map point vectors on this frame 
    # out: [Nx2] image projections (u,v) or [Nx3] array of stereo projections (u,v,ur) in case do_stereo_projet=True,
    #      [Nx1] array of map point depths   
    def project_points(self, points, do_stereo_project=False):                   
        pcs = self.transform_points(points)
        if do_stereo_project:      
            return self.camera.project_stereo(pcs)
        else:
            return self.camera.project(pcs)
    # project a list of N MapPoint objects on this frame
    # out: [Nx2] image projections (u,v) or [Nx3] array of stereo projections (u,v,ur) in case do_stereo_projet=True,
    #      [Nx1] array of map point depths 
    def project_map_points(self, map_points, do_stereo_project=False):    
        points = np.array([p.pt for p in map_points])
        return self.project_points(points, do_stereo_project)

    # project a 3d point vector pw on this frame 
    # out: image point, depth      
    def project_point(self, pw):                
        pc = self.transform_point(pw) # p w.r.t. camera 
        return self.camera.project(pc)   
    # project a MapPoint object on this frame
    # out: image point, depth 
    def project_map_point(self, map_point):           
        return self.project_point(map_point.pt)  
    
    def is_in_image(self, uv, z): 
        return self.camera.is_in_image(uv,z)        
                
    # input: [Nx2] array of uvs, [Nx1] of zs 
    # output: [Nx1] array of visibility flags             
    def are_in_image(self, uvs, zs):
        return self.camera.are_in_image(uvs,zs)   
    
    # input: map_point
    # output: visibility flag, projection uv, depth z
    def is_visible(self, map_point):
        #with self._lock_pose:    (no need, project_map_point already locks the pose)   
        uv,z = self.project_map_point(map_point)
        PO = map_point.pt-self.Ow
                        
        if not self.is_in_image(uv, z):
            return False, uv, z

        dist3D = np.linalg.norm(PO)   
        # point depth must be inside the scale pyramid of the image
        if dist3D < map_point.min_distance or dist3D > map_point.max_distance:
            return False, uv, z
        # viewing angle must be less than 60 deg
        if np.dot(PO,map_point.get_normal()) < Parameters.kViewingCosLimitForPoint * dist3D:
            return False, uv, z
        return True, uv, z 
    
    # input: a list of map points 
    # output: [Nx1] array of visibility flags, 
    #         [Nx2] array of projections (u,v) or [Nx3] array of stereo image points (u,v,ur) in case do_stereo_projet=True, 
    #         [Nx1] array of depths, 
    #         [Nx1] array of distances PO
    # check a) points are in image b) good view angle c) good distance range  
    def are_visible(self, map_points, do_stereo_project=False):
        points = []
        point_normals = []
        min_dists = []
        max_dists = []
        for p in map_points:
            points.append(p.pt)
            point_normals.append(p.get_normal())
            min_dists.append(p.min_distance)
            max_dists.append(p.max_distance)
        points = np.array(points)
        point_normals = np.array(point_normals)
        min_dists = np.array(min_dists)
        max_dists = np.array(max_dists)
        
        #with self._lock_pose:  (no need, project_points already locks the pose) 
        uvs, zs = self.project_points(points, do_stereo_project)    
        POs = points - self.Ow 
        dists   = np.linalg.norm(POs, axis=-1, keepdims=True)    
        POs /= dists
        cos_view = np.sum(point_normals * POs, axis=1)
                
        are_in_image = self.are_in_image(uvs, zs)     
        are_in_good_view_angle = cos_view > Parameters.kViewingCosLimitForPoint         
        dists = dists.reshape(-1,)              
        are_in_good_distance = (dists > min_dists) & (dists < max_dists)        
            
        out_flags = are_in_image & are_in_good_view_angle & are_in_good_distance
        return out_flags, uvs, zs, dists        


def detect_and_compute(img):
    return Frame.tracker.detectAndCompute(img)
     
            
# A Frame mainly collects keypoints, descriptors and their corresponding 3D points 
class Frame(FrameBase):
    tracker         = None      # shared tracker  
    feature_manager = None 
    feature_matcher = None 
    descriptor_distance = None       
    descriptor_distances = None  # norm for vectors     
    is_store_imgs = False         
    def __init__(self, camera: Camera, img, img_right=None, depth=None, pose=None, id=None, timestamp=None, kps_data=None, img_id=None):
        super().__init__(camera, pose=pose, id=id, timestamp=timestamp, img_id=img_id)    
        
        self._lock_features = RLock()  
        self.is_keyframe = False  

        # image keypoints information arrays (unpacked from array of cv::KeyPoint())
        # NOTE: in the stereo case, we assume the images are rectified (and unistortion becomes redundant)
        self.kps       = None      # left keypoint coordinates                                      [Nx2]
        self.kps_r     = None      # righ keypoint coordinates (extracted on right image)           [Nx2]        
        self.kpsu      = None      # [u]ndistorted keypoint coordinates                             [Nx2] 
        self.kpsn      = None      # [n]ormalized keypoint coordinates                              [Nx2] (Kinv * [kp,1])    
        self.octaves   = None      # keypoint octaves                                               [Nx1]
        self.octaves_r = None      # keypoint octaves                                               [Nx1]        
        self.sizes     = None      # keypoint sizes                                                 [Nx1] 
        self.angles    = None      # keypoint sizes                                                 [Nx1]         
        self.des       = None      # keypoint descriptors                                           [NxD] where D is the descriptor length 
        self.des_r     = None      # right keypoint descriptors                                     [NxD] where D is the descriptor length         
        self.depths    = None      # keypoint depths                                                [Nx1]
        self.kps_ur    = None      # corresponding right u-coordinates for left keypoints           [Nx1] (computed with stereo matching and assuming rectified stereo images)

        # map points information arrays 
        self.points   = None     # map points => self.points[idx] (if is not None) is the map point matched with self.kps[idx]
        self.outliers = None     # outliers flags for map points (reset and set by pose_optimization())
        
        self.kf_ref = None       # reference keyframe 
        self.img = None          # image (copy of img if available)
        self.depth_img = None    # depth (copy of depth if available)
                                
        if img is not None:
            #self.H, self.W = img.shape[0:2]                 
            if Frame.is_store_imgs: 
                self.img = img.copy()  
            else: 
                self.img = None    

        if img_right is not None:
            if Frame.is_store_imgs: 
                self.img_r = img_right.copy()  
            else: 
                self.img_r = None                                        

        if depth is not None:
            if self.camera is not None and self.camera.depth_factor != 1.0:
                depth = depth * self.camera.depth_factor             
            if Frame.is_store_imgs: 
                self.depth_img = depth.copy()  
            else: 
                self.depth_img = None   
                                
        if img is not None:                  
            if img_right is not None:
                with ThreadPoolExecutor() as executor:
                    future_l = executor.submit(detect_and_compute, img)
                    future_r = executor.submit(detect_and_compute, img_right)
                    self.kps, self.des = future_l.result()
                    self.kps_r, self.des_r = future_r.result()
                    #print(f'kps: {len(self.kps)}, des: {self.des.shape}, kps_r: {len(self.kps_r)}, des_r: {self.des_r.shape}')
            else: 
                self.kps, self.des = Frame.tracker.detectAndCompute(img)  
                                                                                
            # convert from a list of keypoints to arrays of points, octaves, sizes  
            if self.kps is not None:    
                kps_data = np.array([ [x.pt[0], x.pt[1], x.octave, x.size, x.angle] for x in self.kps ], dtype=np.float32)                     
                self.kps     = kps_data[:,:2] if kps_data is not None else None
                self.octaves = np.uint32(kps_data[:,2]) #print('octaves: ', self.octaves)                    
                self.sizes   = kps_data[:,3]
                self.angles  = kps_data[:,4]  
                if self.camera is not None:
                    self.kpsu = self.camera.undistort_points(self.kps) # convert to undistorted keypoint coordinates             
                    self.kpsn = self.camera.unproject_points(self.kpsu)
                self.points = np.array( [None]*len(self.kpsu) )  # init map points
                self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)
                                
            if self.kps_r is not None: 
                kps_data_r = np.array([ [x.pt[0], x.pt[1], x.octave, x.size, x.angle] for x in self.kps_r ], dtype=np.float32)
                self.kps_r     = kps_data_r[:,:2] if kps_data_r is not None else None
                self.octaves_r = np.uint32(kps_data_r[:,2]) #print('octaves: ', self.octaves)
            
            if self.kps is not None:
                if depth is not None: 
                    self.compute_stereo_from_rgbd(kps_data, depth)
                if img_right is not None:
                    self.depths = np.full(len(self.kps), -1, dtype=np.float)     
                    self.kps_ur = np.full(len(self.kps), -1, dtype=np.float)
                    self.compute_stereo_matches(img, img_right)
            
    def to_json(self):
        ret = {
                'id': int(self.id),
                'timestamp': float(self.timestamp),
                'img_id': int(self.img_id),
                'pose': json.dumps(self.pose.astype(float).tolist()) if self.pose is not None else None,
                'camera': self.camera.to_json(),
                
                'is_keyframe': bool(self.is_keyframe),
                
                'kps': json.dumps(self.kps.astype(float).tolist()),
                'kps_r': json.dumps(self.kps_r.astype(float).tolist() if self.kps_r is not None else None),
                'kpsu': json.dumps(self.kpsu.astype(float).tolist()),
                'kpsn': json.dumps(self.kpsn.astype(float).tolist()),
                'octaves': json.dumps(self.octaves.tolist()),
                'octaves_r': json.dumps(self.octaves_r.tolist() if self.octaves_r is not None else None),
                'sizes': json.dumps(self.sizes.tolist()),
                'angles': json.dumps(self.angles.astype(float).tolist()),
                'des': json.dumps(self.des.tolist()),
                'des_r': json.dumps(self.des_r.astype(float).tolist()) if self.des_r is not None else None,
                'depths': json.dumps(self.depths.astype(float).tolist()) if self.depths is not None else None,
                'kps_ur': json.dumps(self.kps_ur.astype(float).tolist()) if self.kps_ur is not None else None,
                
                'points': json.dumps([p.id if p is not None else None for p in self.points]),
                'outliers': json.dumps(self.outliers.astype(bool).tolist()) if self.outliers is not None else None, 
                
                'kf_ref': self.kf_ref.id if self.kf_ref is not None else None,
                'img': json.dumps(self.img.tolist()) if self.img is not None else None,
                'depth_img': json.dumps(self.depth_img.tolist()) if self.depth_img is not None else None        
                }
        return ret
        
    @staticmethod 
    def from_json(json_str):  
        camera = PinholeCamera.from_json(json_str['camera'])      
        pose = np.array(json.loads(json_str['pose']),dtype=np.float64) if json_str['pose'] is not None else None
        f = Frame(camera=camera,img=None, img_right=None, depth=None,
                   pose = pose, 
                   id=json_str['id'], 
                   timestamp=json_str['timestamp'],
                   img_id=json_str['img_id'])
        
        f.is_keyframe = json_str['is_keyframe']
        
        f.kps = np.array(json.loads(json_str['kps'])) if json_str['kps'] is not None else None
        len_kps = len(f.kps) if f.kps is not None else 1
        
        f.kps_r = np.array(json.loads(json_str['kps_r'])) if json_str['kps_r'] is not None else None
        f.kpsu = np.array(json.loads(json_str['kpsu'])) if json_str['kpsu'] is not None else None
        f.kpsn = np.array(json.loads(json_str['kpsn'])) if json_str['kpsn'] is not None else None
        f.octaves = np.array(json.loads(json_str['octaves'])) if json_str['octaves'] is not None else None
        f.octaves_r = np.array(json.loads(json_str['octaves_r'])) if json_str['octaves_r'] is not None else None
        f.sizes = np.array(json.loads(json_str['sizes'])) if json_str['sizes'] is not None else None
        f.angles = np.array(json.loads(json_str['angles'])) if json_str['angles'] is not None else None
        f.des = np.array(json.loads(json_str['des'])) if json_str['des'] is not None else None
        f.des_r = np.array(json.loads(json_str['des_r'])) if json_str['des_r'] is not None else None
        f.depths = np.array(json.loads(json_str['depths'])) if json_str['depths'] is not None else None
        f.kps_ur = np.array(json.loads(json_str['kps_ur'])) if json_str['kps_ur'] is not None else None
        
        f.points = np.array(json.loads(json_str['points'])) if json_str['points'] is not None else None     
        
        f.outliers = np.array(json.loads(json_str['outliers'])) if json_str['outliers'] is not None else None
        f.kf_ref = json_str['kf_ref'] if json_str['kf_ref'] is not None else None
        f.img = np.array(json.loads(json_str['img'])) if json_str['img'] is not None else None
        f.depth_img = np.array(json.loads(json_str['depth_img'])) if json_str['depth_img'] is not None else None
        
        if f.kps is not None and f.points is not None:
            #print(f'f.kps.shape = {f.kps.shape}, f.points.shape = {f.points.shape}')        
            assert(len(f.kps) == len(f.points))
        
        return f
    
    # post processing after deserialization to replace saved ids with reloaded objects
    def replace_ids_with_objects(self, points, frames, keyframes):
        def get_object_with_id(id, objs):
            if id is None: 
                return None            
            found_objs = [o for o in objs if o is not None and o.id == id]
            #print(f'found_objs = {found_objs}, id = {id}, objs = {[o.id for o in objs]}')
            return found_objs[0] if len(found_objs) > 0 else None
        # get actual points         
        if self.points is not None and len(self.points) > 0:  
            actual_points = np.array([get_object_with_id(id, points) for id in self.points])
            self.points = actual_points
        # get actual kf_ref 
        if self.kf_ref is not None:
            self.kf_ref = get_object_with_id(self.kf_ref, keyframes)
        
                
    @staticmethod
    def set_tracker(tracker):
        Frame.tracker = tracker 
        Frame.feature_manager = tracker.feature_manager 
        Frame.feature_matcher = tracker.matcher
        Frame.descriptor_distance  = tracker.descriptor_distance       
        Frame.descriptor_distances = tracker.descriptor_distances        
        Frame.oriented_features = tracker.feature_manager.oriented_features
        Frame._id = 0           
     
    # KD tree of undistorted keypoints
    @property
    def kd(self):
        if not hasattr(self, '_kd'):
            self._kd = cKDTree(self.kpsu)
        return self._kd

    def delete(self):
        with self._lock_pose:           
            with self._lock_features:          
                del self
                            
    def get_point_match(self, idx):
        with self._lock_features:          
            return self.points[idx] 
        
    def set_point_match(self, p, idx):
        with self._lock_features:  
            self.points[idx] = p 

    def remove_point_match(self, idx):       
        with self._lock_features:  
            self.points[idx] = None 
            
    def replace_point_match(self, p, idx):                
        self.points[idx] = p    # replacing is not critical (it does not create a 'None jump')
        
    def remove_point(self, p):
        with self._lock_features:          
            try: 
                p_idxs = np.where(self.points == p)[0]  # remove all instances 
                self.points[p_idxs] = None        
            except:
                pass 
            
    def remove_frame_views(self, idxs):
        with self._lock_features:    
            if len(idxs) == 0:
                return 
            for idx,p in zip(idxs,self.points[idxs]): 
                if p is not None: 
                    p.remove_frame_view(self,idx)

    def reset_points(self):
        with self._lock_features:          
            self.points = np.array([None]*len(self.kpsu))
            self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)      
            
    def get_points(self):    
        with self._lock_features:                           
            return self.points.copy()    
                    
    def get_matched_points(self):
        with self._lock_features:                   
            matched_idxs = np.flatnonzero(self.points!=None) 
            matched_points = self.points[matched_idxs]            
            return matched_points #, matched_idxs 

    def get_unmatched_points_idxs(self):         
        with self._lock_features:             
            unmatched_idxs = np.flatnonzero(self.points==None)          
            return unmatched_idxs     
                
    def get_matched_inlier_points(self):         
        with self._lock_features:             
            matched_idxs = np.flatnonzero( (self.points!=None) & (self.outliers==False)) 
            matched_points = self.points[matched_idxs]            
            return matched_points, matched_idxs 
    
    def get_matched_good_points(self):       
        with self._lock_features:               
            good_points = [p for p in self.points if p is not None and not p.is_bad]         
            return good_points    
    
    def num_tracked_points(self, minObs = 1):
        with self._lock_features:          
            # num_points = 0
            # for i,p in enumerate(self.points):
            #     if p is not None and not p.is_bad: 
            #         if p.num_observations >= minObs:  
            #             num_points += 1   
            # return num_points 
            return sum(1 for p in self.points if p is not None and p.is_good_with_min_obs(minObs))


    def num_matched_inlier_map_points(self):
        with self._lock_features:          
            # num_matched_points = 0
            # for i,p in enumerate(self.points):
            #     if p is not None and not self.outliers[i]: 
            #         if p.num_observations > 0:
            #             num_matched_points += 1             
            # return num_matched_points     
            return sum(
                1 for i, p in enumerate(self.points)
                if p is not None and not self.outliers[i] and p.num_observations > 0
            )            
        
    # update found count for map points        
    def update_map_points_statistics(self, sensor_type=SensorType.MONOCULAR):
        with self._lock_features:           
            num_matched_points = 0
            for i,p in enumerate(self.points):
                if p is not None and not self.outliers[i]: 
                        p.increase_found() # update point statistics 
                        if p.num_observations > 0:
                            num_matched_points +=1
                elif sensor_type == SensorType.STEREO: 
                    self.points[i] = None
                    
            # for p, is_outlier in zip(self.points, self.outliers):
            #     if p is not None and not is_outlier:
            #         p.increase_found()  # Update point statistics
            #         if p.num_observations > 0:
            #             num_matched_points += 1                    
                    
            return num_matched_points            
           
    # reset outliers detected in last pose optimization       
    def clean_outlier_map_points(self):
        with self._lock_features:          
            num_matched_points = 0
            for i,p in enumerate(self.points):
                if p is not None:
                    if self.outliers[i]: 
                        p.remove_frame_view(self,i)         
                        self.points[i] = None 
                        self.outliers[i] = False
                        p.last_frame_id_seen = self.id     
                    else:
                        if p.num_observations > 0:
                            num_matched_points +=1
            return num_matched_points           
                    
    # reset bad map points and update visibility count          
    def clean_bad_map_points(self):
        with self._lock_features:           
            for i,p in enumerate(self.points):
                if p is None:
                    continue 
                if p.is_bad: 
                    p.remove_frame_view(self,i)         
                    self.points[i] = None 
                    self.outliers[i] = False      
                else:
                    p.last_frame_id_seen = self.id   
                    p.increase_visible()   
                    
    # update statistics for map points        
    def clean_vo_map_points(self):
        with self._lock_features:           
            num_cleaned_points = 0        
            for i,p in enumerate(self.points):
                if p and p.num_observations < 1:
                    self.points[i] = None 
                    self.outliers[i] = False                          
                    num_cleaned_points += 1
            print('#cleaned vo points: ', num_cleaned_points)
                    
    # check for point replacements 
    def check_replaced_map_points(self):
        with self._lock_features:   
            num_replaced_points = 0        
            for i,p in enumerate(self.points):      
                if p is not None: 
                    replacement = p.get_replacement()
                    if replacement is not None: 
                        self.points[i] = replacement       
                        num_replaced_points +=1
            print('#replaced points: ', num_replaced_points)            

    def compute_stereo_from_rgbd(self, kps_data, depth):
        kps_int = np.uint32(kps_data[:,:2])
        depth_values = depth[kps_int[:, 1], kps_int[:, 0]]  # Depth at keypoint locations (v, u)                       
        self.depths = np.where(depth_values > 0, depth_values, -1) 
        safe_depth_values = np.where(depth_values == 0, np.inf, depth_values) # to prevent division by zero 
        self.kps_ur = np.where(depth_values > 0, self.kpsu[:,0] - self.camera.bf / safe_depth_values, -1)   
        #print(f'depth: {self.depths}, kps_ur: {self.kps_ur}')  

    def compute_stereo_matches(self, img, img_right): 
        min_z = self.camera.b
        min_disparity = 0
        max_disparity = self.camera.bf/min_z
        # we enforce matching on the same row here by using the flag row_matching (epipolar constraint)
        row_matching = True
        ratio_test = 0.8
        stereo_matching_result = Frame.feature_matcher.match(img, img_right, self.des, self.des_r, self.kps, self.kps_r, \
                                                                  ratio_test=ratio_test, row_matching=row_matching, max_disparity=max_disparity)
        matched_kps_l = np.array(self.kps[stereo_matching_result.idxs1])
        matched_kps_r = np.array(self.kps_r[stereo_matching_result.idxs2])         
                          
        # check disparity range
        disparities =  np.array(matched_kps_l[:,0] - matched_kps_r[:,0], dtype=np.float64) # assuming keypoints are extracted from rectified images
        good_disparities_mask = np.logical_and(disparities > min_disparity, disparities < max_disparity)
        good_disparities = disparities[good_disparities_mask]
        good_matched_idxs1 = stereo_matching_result.idxs1[good_disparities_mask]
        good_matched_idxs2 = stereo_matching_result.idxs2[good_disparities_mask]
        
        if Parameters.kStereoMatchingShowMatchedPoints: # debug stereo matching
            print(f'[compute_stereo_matches] found intial {len(good_matched_idxs1)} stereo matches')
            stereo_img_matches = draw_feature_matches(img, img_right, self.kps[good_matched_idxs1], self.kps_r[good_matched_idxs2], horizontal=False)
            #cv2.namedWindow('stereo_img_matches', cv2.WINDOW_NORMAL)
            cv2.imshow('stereo_img_matches', stereo_img_matches)
            cv2.waitKey(1)   
            
        # compute fundamental matrix and check inliers, just for the hell of it (debugging) ... this is totally redundant here due to the row_matching
        do_check_with_fundamental_mat = False
        if do_check_with_fundamental_mat:
            ransac_method = None 
            try: 
                ransac_method = cv2.USAC_MSAC 
            except: 
                ransac_method = cv2.RANSAC             
            fmat_err_thld = 3.0         # threshold for fundamental matrix estimation: maximum allowed distance from a point to an epipolar line in pixels (to treat a point pair as an inlier)  
            F, mask_inliers = cv2.findFundamentalMat(self.kps[good_matched_idxs1], self.kps_r[good_matched_idxs2], ransac_method, fmat_err_thld, confidence=0.999)
            mask_inliers = mask_inliers.flatten()==1
            print(f'[compute_stereo_matches] perc good fundamental-matrix matches: {100 * np.sum(mask_inliers) / len(good_matched_idxs1)}')            
            good_disparities = good_disparities[mask_inliers]         
            good_matched_idxs1 = good_matched_idxs1[mask_inliers]
            good_matched_idxs2 = good_matched_idxs2[mask_inliers]   
                                
        # check normalized sum of absolute differences at matched points (at level 0)
        do_check_sads = True 
        if do_check_sads:
            window_size = 5
            # TODO: optimize this conversions (probably we can store them in the class if this has been done before)
            if img.ndim>2:
                img_bw_ = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if img_right.ndim>2:
                img_right_ = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)             
            sads = compute_NSAD_between_matched_keypoints(img_bw_, img_right_, self.kps[good_matched_idxs1], self.kps_r[good_matched_idxs2], window_size)
            #print(f'sads: {sads}')
            sads_median = np.median(sads)     # MAD, approximating dists_median=0 
            sigma_mad = 1.4826 * sads_median
            good_sads_mask = sads < 1.5*sigma_mad
            print(f'[compute_stereo_matches] perc good sads: {100 * np.sum(good_sads_mask) / len(sads)}')
            good_disparities = good_disparities[good_sads_mask]
            good_matched_idxs1 = good_matched_idxs1[good_sads_mask]
            good_matched_idxs2 = good_matched_idxs2[good_sads_mask]           
            
        # check chi2 of reprojections errors, just for the hell of it (debugging)
        do_chi2_check = False
        if do_chi2_check:          
            # triangulate the points    
            pose_l = np.eye(4)
            t_rl = np.array([-self.camera.b, 0, 0])
            pose_rl = poseRt(np.eye(3),t_rl) # assuming stereo rectified images 
            #print(f'pose_l: {pose_l}, pose_lr: {pose_lr}')
            kpsn_r = self.camera.unproject_points(self.kps_r) # assuming rectified stereo images and so kps_r is undistorted
            pts3d, mask_pts3d = triangulate_normalized_points(pose_l, pose_rl, self.kpsn[good_matched_idxs1], kpsn_r[good_matched_idxs2])
            
            pts3d = pts3d[mask_pts3d]
            good_disparities = good_disparities[mask_pts3d]
            good_matched_idxs1 = good_matched_idxs1[mask_pts3d]
            good_matched_idxs2 = good_matched_idxs2[mask_pts3d]

            # check reprojection errors
             
            uvs_l, depths_l = self.camera.project(pts3d)  # left projections   
            good_depths_l = depths_l > 0     
                  
            pts3d_r = pts3d + t_rl   
            uvs_r, depths_r = self.camera.project(pts3d_r) # right projections 
            good_depths_r = depths_r > 0    
            
            good_depths_mask = np.logical_and(good_depths_l, good_depths_r)
            uvs_l = uvs_l[good_depths_mask][:]
            uvs_r = uvs_r[good_depths_mask][:]
            good_disparities = good_disparities[good_depths_mask]
            good_matched_idxs1 = good_matched_idxs1[good_depths_mask]
            good_matched_idxs2 = good_matched_idxs2[good_depths_mask]
                        
            # compute mono reproj errors on left image
            errs_l_vec = uvs_l - self.kps[good_matched_idxs1]
           
            errs_l_sqr = np.sum(errs_l_vec * errs_l_vec, axis=1)  # squared reprojection errors 
            kpsl_levels = self.octaves[good_matched_idxs1]
            invSigmas2_1 = Frame.feature_manager.inv_level_sigmas2[kpsl_levels] 
            chis2_l = errs_l_sqr * invSigmas2_1         # chi square 
            #print(f'chis2_l: {chis2_l}')
            good_chi2_l_mask = chis2_l < Parameters.kChi2Mono
            num_good_chi2_l = good_chi2_l_mask.sum()
            #print(f'perc good chis2_l: {100*num_good_chi2_l/len(good_chi2_l_mask)}')            
            
            errs_r_vec = uvs_r - self.kps_r[good_matched_idxs2]
            errs_r_sqr = np.sum(errs_r_vec * errs_r_vec, axis=1)  # squared reprojection errors 
            kpsr_levels = self.octaves_r[good_matched_idxs2]
            invSigmas2_2 = Frame.feature_manager.inv_level_sigmas2[kpsr_levels] 
            chis2_r = errs_r_sqr * invSigmas2_2         # chi square 
            #print(f'chis2_r: {chis2_r}')
            good_chi2_r_mask = chis2_r < Parameters.kChi2Mono
            num_good_chi2_r = good_chi2_r_mask.sum()
            #print(f'perc good chis2_r: {100*num_good_chi2_r/len(good_chi2_r_mask)}')
            
            good_chis2_mask = np.logical_and(chis2_l < Parameters.kChi2Mono, chis2_r < Parameters.kChi2Mono)
            num_good_chis2 = good_chis2_mask.sum()
            print(f'[compute_stereo_matches] perc good chis2: {100*num_good_chis2/len(good_matched_idxs1)}')
            good_disparities = good_disparities[good_chis2_mask]
            good_matched_idxs1 = good_matched_idxs1[good_chis2_mask]
            good_matched_idxs2 = good_matched_idxs2[good_chis2_mask]            
                              
        # filter descriptor distances with sigma-mad 
        do_check_with_des_distances_sigma_mad = False
        if do_check_with_des_distances_sigma_mad:
            des1 = stereo_matching_result.des1[good_matched_idxs1]
            des2 = stereo_matching_result.des2[good_matched_idxs2]
            sigma_mad, des_dists = descriptor_sigma_mad(des1, des2, descriptor_distances=Frame.descriptor_distances)
            good_des_dists_mask = des_dists < 1.5 * sigma_mad
            num_good_des_dists = good_des_dists_mask.sum()
            print(f'[compute_stereo_matches] perc good des distances: {100*num_good_des_dists/len(good_des_dists_mask)}')
            good_disparities = good_disparities[good_des_dists_mask]
            good_matched_idxs1 = good_matched_idxs1[good_des_dists_mask]
            good_matched_idxs2 = good_matched_idxs2[good_des_dists_mask]                
                
        print(f'[compute_stereo_matches] found final {len(good_matched_idxs1)} stereo matches')                
        self.depths[good_matched_idxs1] = self.camera.bf / good_disparities
        self.kps_ur[good_matched_idxs1] = self.kps_r[good_matched_idxs2][:,0]              

    # unproject keypoints where the depth is available                               
    def unproject_points_3d(self, idxs, transform_in_world=False):
        if self.depths is not None:
            depth_values = self.depths[idxs].reshape(-1, 1) 
            kpsn = add_ones(self.kpsn[idxs])
            pts3d_mask = np.where(depth_values>0, True, False)
            pts3d = np.where(depth_values>0, kpsn*depth_values, np.zeros(3))
            if transform_in_world: 
                pts3d = (self._pose.Rwc @ pts3d.T + self._pose.Ow[:, np.newaxis]).T
            return pts3d, pts3d_mask
        else:
            return None, None
                                                               
    def compute_points_median_depth(self, points3d = None):
        with self._lock_pose:        
            Rcw2 = self._pose.Rcw[2,:3]  # just 2-nd row 
            tcw2 = self._pose.tcw[2]   # just 2-nd row                    
        if points3d is None: 
            with self._lock_features:                
                points3d = np.array([p.pt for p in self.points if p is not None])
        if len(points3d)>0:
            z = np.dot(Rcw2, points3d[:,:3].T) + tcw2 
            z = sorted(z) 
            return z[ ( len(z)-1)//2 ]                
        else:
            Printer.red('frame.compute_points_median_depth() with no points')
            return -1 
        
    # draw tracked features on the image for selected keypoint indexes 
    def draw_feature_trails(self, img, kps_idxs, trail_max_length=9):
        img = img.copy()
        with self._lock_features:
            uvs = np.rint(self.kps[kps_idxs]).astype(np.intp) # image keypoints coordinates  # use distorted coordinates when drawing on distorted original image           
            # for each keypoint idx
            for i, kp_idx in enumerate(kps_idxs):
                #u1, v1 = int(round(self.kps[kp_idx][0])), int(round(self.kps[kp_idx][1]))  # use distorted coordinates when drawing on distorted original image 
                uv = tuple(uvs[i])
                
                #radius = self.sizes[kp_idx] # actual size
                radius = kDrawFeatureRadius[self.octaves[kp_idx]] # fake size for visualization
                
                #color = myjet[self.octaves[i1]]*255
                point = self.points[kp_idx]
                if point is not None and not point.is_bad:
                    p_frame_views = point.frame_views()
                    # there's a corresponding 3D map point
                    color = (0, 255, 0) if len(p_frame_views) > 2 else (255, 0, 0)
                    cv2.circle(img, uv, color=color, radius=radius, thickness=1)  # draw keypoint size as a circle  
                    # draw the trail (for each keypoint, its trail_max_length corresponding points in previous frames)
                    pts = []
                    lfid = None  # last frame id
                    for f, idx in reversed(p_frame_views[-trail_max_length:]):
                        if lfid is not None and lfid-1 != f.id:
                            # stop when there is a jump in the ids of frame observations
                            break
                        pts.append(tuple(map(int,np.round(f.kps[idx]))))
                        lfid = f.id                    
                    if len(pts) > 1:
                        color = myjet[len(pts)] * 255
                        cv2.polylines(img, np.array([pts], dtype=np.int32), False, color, thickness=1, lineType=16)
                else:
                    # no corresponding 3D point
                    cv2.circle(img, uv, color=(0, 0, 0), radius=2) #radius=radius)
            return img    
        
    # draw tracked features on the image
    def draw_all_feature_trails(self, img):
        kps_idxs = range(len(self.kps))
        return self.draw_feature_trails(img, kps_idxs)   


# match frames f1 and f2
# out: a vector of match index pairs [idx1[i],idx2[i]] such that the keypoint f1.kps[idx1[i]] is matched with f2.kps[idx2[i]]
def match_frames(f1: Frame, f2: Frame, ratio_test=None):     
    matching_result = Frame.feature_matcher.match(f1.img, f2.img, f1.des, f2.des, ratio_test)
    return matching_result
    # idxs1, idxs2 = matching_result.idxs1, matching_result.idxs2
    # idxs1 = np.asarray(idxs1)
    # idxs2 = np.asarray(idxs2)   
    # return idxs1, idxs2         
