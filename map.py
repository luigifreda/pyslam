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

import time
import numpy as np
import json
import math 

from geom_helpers import poseRt, add_ones
import parameters 
from frame import Frame
from map_point import MapPoint

from helpers import Printer

import optimizer_g2o 
# from optimize_crappy import optimize


class Map(object):
    def __init__(self):
        self.frames = []
        self.keyframes = []
        self.points = []
        self.max_frame_id = 0  # 0 is the first frame id
        self.max_point_id = 0  # 0 is the first point id

        # local map 
        self.local_map = LocalMap()

    def add_point(self, point):
        ret = self.max_point_id
        self.max_point_id += 1
        self.points.append(point)
        return ret

    def remove_point(self, point):
        self.points.remove(point)
        point.delete()

    def add_frame(self, frame):
        ret = self.max_frame_id
        frame.id = ret # override original id    
        self.max_frame_id += 1
        self.frames.append(frame)     
        return ret

    def annotate(self, img):
        if len(self.frames) > 0:
            img_draw = self.frames[-1].draw_feature_trails(img)
            return img_draw
        return img

    # add new points to the map from pairwise matches
    # points4d is [Nx4]
    def add_points(self, points4d, mask_pts4d, f1, f2, idx1, idx2, img1, check_parallax=True):
        assert(points4d.shape[0] == len(idx1))
        new_pts_count = 0
        out_mask_pts4d = np.full(points4d.shape[0], False, dtype=bool)
        if mask_pts4d is None:
            mask_pts4d = np.full(points4d.shape[0], True, dtype=bool)
        
        for i, p in enumerate(points4d):
            if not mask_pts4d[i]:
                #print('p[%d] not good' % i)
                continue

            # check parallax is large enough (this is going to filter out all points when the inter-frame motion is almost zero)
            if check_parallax is True:
                Rwc1 = f1.pose[:3, :3].T
                Rwc2 = f2.pose[:3, :3].T
                r1 = np.dot(Rwc1, add_ones(f1.kpsn[idx1[i]]))
                r2 = np.dot(Rwc2, add_ones(f2.kpsn[idx2[i]]))
                cos_parallax = r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                if cos_parallax > parameters.kCosMinParallax:
                    # print('p[',i,']: ',p,' not enough parallax: ', cos_parallax)
                    continue

            # check points are in front of both cameras
            x1 = np.dot(f1.pose, p)
            x1 = x1/x1[3]
            x2 = np.dot(f2.pose, p)
            x2 = x2/x2[3]
            if x1[2] < 0 or x2[2] < 0:
                #print('p[', i, ']: ', p, ' not visible')
                continue

            # project on camera pixels
            uv1 = np.dot(f1.K, x1[:3])
            uv2 = np.dot(f2.K, x2[:3])
                
            # check reprojection error
            invSigma21 = Frame.detector.inv_level_sigmas2[f1.octaves[idx1[i]]]                
            err1 = (uv1[0:2] / uv1[2]) - f1.kpsu[idx1[i]]        
            err1 = np.linalg.norm(err1)*invSigma21        
            invSigma22 = Frame.detector.inv_level_sigmas2[f2.octaves[idx2[i]]]                 
            err2 = (uv2[0:2] / uv2[2]) - f2.kpsu[idx2[i]]         
            err2 = np.linalg.norm(err2)*invSigma22
            if err1 > parameters.kChi2Mono or err2 > parameters.kChi2Mono: # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                #print('p[%d] big reproj err1 %f, err2 %f' % (i,err1,err2))
                continue

            # add the point to this map 
            try:
                color = img1[int(round(f1.kps[idx1[i], 1])), int(round(f1.kps[idx1[i], 0]))]
            except IndexError:
                color = (255, 0, 0)
            pt = MapPoint(self, p[0:3], color)
            pt.add_observation(f2, idx2[i])
            pt.add_observation(f1, idx1[i])
            new_pts_count += 1
            out_mask_pts4d[i] = True 
        return new_pts_count, out_mask_pts4d

    # get the points of the last N frames and all the frames that see these points 
    def update_local_window_map(self, local_window):
        local_frames = self.frames[-local_window:]  # get the last N frames  
        local_frame_id_set = set([f.id for f in local_frames])                 
        points = []
        point_id_set = set()           
        ref_frames = []   # reference frames, i.e. frames not in "local frames" that see points observed in local frames      

        f_points = [p for f in local_frames for p in f.points if (p is not None) ] 
        for p in f_points: 
            if p.id not in point_id_set and not p.is_bad:  # discard already considered points or bad points 
                points.append(p)
                point_id_set.add(p.id)        
                point_frames = [p_frame for p_frame in p.frames if p_frame.id not in local_frame_id_set] 
                for p_frame in point_frames: 
                    #if p_frame.id not in frame_id_set:                        
                    ref_frames.append(p_frame)
                    local_frame_id_set.add(p_frame.id)                             

        #ref_frames = sorted(ref_frames, key=lambda x:x.id)  
        self.local_map.frames = local_frames
        self.local_map.points = points 
        self.local_map.ref_frames = ref_frames
        self.local_map.f_cur = self.frames[-1]                                                                
        return local_frames, points, ref_frames

    # remove points which have a big reprojection error 
    def cull_map_points(self, points): 
        #print('map points: ', sorted([p.id for p in self.points]))
        #print('points: ', sorted([p.id for p in points]))           
        culled_pt_count = 0
        for p in points:
            # compute reprojection error
            errs = []
            for f, idx in zip(p.frames, p.idxs):
                uv = f.kpsu[idx]
                proj = f.project_map_point(p)
                invSigma2 = Frame.detector.inv_level_sigmas2[f.octaves[idx]]
                errs.append(np.linalg.norm(proj-uv)*invSigma2)
            # cull
            if np.mean(errs) > parameters.kChi2Mono:  # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                culled_pt_count += 1
                #print('removing point: ',p.id, 'from frames: ', [f.id for f in p.frames])
                #self.points.remove(p)
                #p.delete()
                self.remove_point(p)
        print("culled: %d points" % (culled_pt_count))        

    def optimize(self, local_window=parameters.kLargeWindow, verbose=False, rounds=10):
        err = optimizer_g2o.optimization(frames = self.frames, points = self.points, local_window = local_window, verbose = verbose, rounds = rounds)        
        self.cull_map_points(self.points)

        return err

    def locally_optimize(self, local_window=parameters.kLocalWindow, verbose = False, rounds=10):
        frames, points, frames_ref = self.update_local_window_map(local_window)
        print('local optimization window: ', [f.id for f in frames])        
        print('                     refs: ', [f.id for f in frames_ref])
        #print('                   points: ', sorted([p.id for p in points]))        
        #err = optimizer_g2o.optimize(frames, points, None, False, verbose, rounds)  
        err, ratio_bad_observations = optimizer_g2o.localOptimization(frames, points, frames_ref, False, verbose, rounds)
        Printer.green('local optimization - perc bad observations: %.2f %%  ' % (ratio_bad_observations*100) )              
        #self.cull_map_points(points)  # already performed in optimizer_g2o.localOptimization()            
        return err 

    # FIXME: according to new changes
    def serialize(self):
        ret = {}
        ret['points'] = [{'id': p.id, 'pt': p.pt.tolist(
        ), 'color': p.color.tolist()} for p in self.points]
        ret['frames'] = []
        for f in self.frames:
            ret['frames'].append({
                'id': f.id, 'K': f.K.tolist(), 'pose': f.pose.tolist(), 'h': f.h, 'w': f.w,
                'kpus': f.kpus.tolist(), 'des': f.des.tolist(),
                'pts': [p.id if p is not None else -1 for p in f.pts]
                })
        ret['max_frame_id'] = self.max_frame_id
        ret['max_point_id'] = self.max_point_id
        return json.dumps(ret)

    # FIXME: according to new changes
    def deserialize(self, s):
        ret = json.loads(s)
        self.max_frame_id = ret['max_frame_id']
        self.max_point_id = ret['max_point_id']
        self.points = []
        self.frames = []

        pids = {}
        for p in ret['points']:
            pp = MapPoint(self, p['pt'], p['color'], p['id'])
            self.points.append(pp)
            pids[p['id']] = pp

        for f in ret['frames']:
            ff = Frame(self, None, f['K'], f['pose'], f['id'])
            ff.w, ff.h = f['w'], f['h']
            ff.kpsu = np.array(f['kpus'])
            ff.des = np.array(f['des'])
            ff.points = [None] * len(ff.kpsu)
            for i, p in enumerate(f['pts']):
                if p != -1:
                    ff.points[i] = pids[p]
            self.frames.append(ff)

# TODO: implement a proper local mapping 
# a simple implementation of a local map 
class LocalMap(object):
    def __init__(self, f_cur = None):
        self.f_cur = f_cur   # 'reference' frame, around which the local map is computed 
        self.frames = []     # collection of frames 
        self.points = []     # points visible in 'frames'  
        self.ref_frames = [] # collection of frames not in 'frames' that see at least one point in 'points'

    def is_empty(self):
        return len(self.frames)==0 