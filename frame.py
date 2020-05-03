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

import cv2
import numpy as np
#import g2o

from threading import RLock, Thread
from scipy.spatial import cKDTree

from parameters import Parameters  

from camera_pose import CameraPose

from utils_geom import add_ones, poseRt, normalize
from utils import myjet, Printer


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
    def __init__(self, camera, pose=None, id=None, timestamp=None):
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
    # out: [Nx2] array of image points, [Nx1] array of map point depths   
    def project_points(self, points):                   
        pcs = self.transform_points(points)      
        return self.camera.project(pcs)
    # project a list of N MapPoint objects on this frame
    # out: Nx2 image points, [Nx1] array of map point depths 
    def project_map_points(self, map_points):    
        points = np.array([p.pt for p in map_points])
        return self.project_points(points)

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
    # output: [Nx1] array of visibility flags, [Nx2] array of projections, [Nx1] array of depths, [Nx1] array of distances PO
    # check a) points are in image b) good view angle c) good distance range  
    def are_visible(self, map_points):
        points = []
        point_normals = []
        min_dists = []
        max_dists = []
        for i, p in enumerate(map_points):
            points.append(p.pt)
            point_normals.append(p.get_normal())
            min_dists.append(p.min_distance)
            max_dists.append(p.max_distance)
        points = np.array(points)
        point_normals = np.array(point_normals)
        min_dists = np.array(min_dists)
        max_dists = np.array(max_dists)
        
        #with self._lock_pose:  (no need, project_points already locks the pose) 
        uvs, zs = self.project_points(points)    
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


# A Frame mainly collects keypoints, descriptors and their corresponding 3D points 
class Frame(FrameBase):
    tracker         = None      # shared tracker  
    feature_manager = None 
    feature_matcher = None 
    descriptor_distance = None       
    descriptor_distances = None  # norm for vectors     
    is_store_imgs = False         
    def __init__(self, img, camera, pose=None, id=None, timestamp=None, kps_data=None):
        super().__init__(camera, pose, id, timestamp)    
        
        self._lock_features = RLock()  
                
        self.is_keyframe = False  

        # image keypoints information arrays (unpacked from array of cv::KeyPoint())
        self.kps     = None      # keypoint coordinates                  [Nx2]
        self.kpsu    = None      # [u]ndistorted keypoint coordinates    [Nx2]
        self.kpsn    = None      # [n]ormalized keypoint coordinates     [Nx2] (Kinv * [kp,1])    
        self.octaves = None      # keypoint octaves                      [Nx1]
        self.sizes   = None      # keypoint sizes                        [Nx1] 
        self.angles  = None      # keypoint sizes                        [Nx1]         
        self.des     = None      # keypoint descriptors                  [NxD] where D is the descriptor length 

        # map points information arrays 
        self.points   = None      # map points => self.points[idx] (if is not None) is the map point matched with self.kps[idx]
        self.outliers = None      # outliers flags for map points (reset and set by pose_optimization())
        
        self.kf_ref = None        # reference keyframe 

        if img is not None:
            #self.H, self.W = img.shape[0:2]                 
            if Frame.is_store_imgs: 
                self.img = img.copy()  
            else: 
                self.img = None                    
            if kps_data is None:   
                self.kps, self.des = Frame.tracker.detectAndCompute(img)                                                         
                # convert from a list of keypoints to arrays of points, octaves, sizes  
                kps_data = np.array([ [x.pt[0], x.pt[1], x.octave, x.size, x.angle] for x in self.kps ], dtype=np.float32)                            
                self.kps     = kps_data[:,:2]    
                self.octaves = np.uint32(kps_data[:,2]) #print('octaves: ', self.octaves)                      
                self.sizes   = kps_data[:,3]
                self.angles  = kps_data[:,4]       
            else:
                # FIXME: this must be updated according to the new serialization 
                #self.kpsu, self.des = des, np.array(list(range(len(des)))*32, np.uint8).reshape(32, len(des)).T
                pass 
            self.kpsu = self.camera.undistort_points(self.kps) # convert to undistorted keypoint coordinates             
            self.kpsn = self.camera.unproject_points(self.kpsu)
            self.points = np.array( [None]*len(self.kpsu) )  # init map points
            self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)
            
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
            num_points = 0
            for i,p in enumerate(self.points):
                if p is not None and not p.is_bad: 
                    if p.num_observations >= minObs:  
                        num_points += 1   
            return num_points 

    def num_matched_inlier_map_points(self):
        with self._lock_features:          
            num_matched_points = 0
            for i,p in enumerate(self.points):
                if p is not None and not self.outliers[i]: 
                    if p.num_observations > 0:
                        num_matched_points += 1             
            return num_matched_points     
        
    # update found count for map points        
    def update_map_points_statistics(self):
        with self._lock_features:           
            num_matched_points = 0
            for i,p in enumerate(self.points):
                if p is not None and not self.outliers[i]: 
                        p.increase_found() # update point statistics 
                        if p.num_observations > 0:
                            num_matched_points +=1
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
                if p is not None: 
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
                if p is not None:
                    if p.num_observations < 1:
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
                        replaced = p
                        self.points[i] = replacement   
                        del replaced        
                        num_replaced_points +=1
            print('#replaced points: ', num_replaced_points)            
                                                           
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
            for kp_idx in kps_idxs:
                #u1, v1 = int(round(self.kps[kp_idx][0])), int(round(self.kps[kp_idx][1]))  # use distorted coordinates when drawing on distorted original image 
                uv = tuple(uvs[kp_idx])
                
                #radius = self.sizes[kp_idx] # actual size
                radius = kDrawFeatureRadius[self.octaves[kp_idx]] # fake size for visualization
                
                #color = myjet[self.octaves[i1]]*255
                point = self.points[kp_idx]
                if point is not None and not point.is_bad:
                    p_frame_views = point.frame_views()
                    # there's a corresponding 3D map point
                    if len(p_frame_views) > 2: 
                        cv2.circle(img, uv, color=(0, 255, 0), radius=radius, thickness=1)  # draw keypoint size as a circle 
                        #cv2.circle(img, uv, color=color, radius=radius)
                    else:
                        cv2.circle(img, uv, color=(255, 0, 0), radius=radius, thickness=1)  # draw keypoint size as a circle 
                    # draw the trail (for each keypoint, its trail_max_length corresponding points in previous frames)
                    pts = []
                    lfid = None  # last frame id
                    for f, idx in p_frame_views[-trail_max_length:][::-1]:
                        if lfid is not None and lfid-1 != f.id:
                            # stop when there is a jump in the ids of frame observations
                            break
                        pts.append(tuple(map(lambda x: int(round(x)), f.kps[idx])))
                        lfid = f.id                    
                    if len(pts) > 1:
                        cv2.polylines(img, np.array([pts], dtype=np.int32), False, myjet[len(pts)]*255, thickness=1, lineType=16)
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
def match_frames(f1, f2, ratio_test=None):     
    idx1, idx2 = Frame.feature_matcher.match(f1.des, f2.des, ratio_test)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)   
    return idx1, idx2         
