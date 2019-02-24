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
* along with PYVO. If not, see <http://www.gnu.org/licenses/>.
"""

import time
import numpy as np
import json

from geom_helpers import poseRt, hamming_distance, add_ones
import constants
from frame import Frame
from map_point import MapPoint

import optimizer_g2o 
# from optimize_crappy import optimize


LOCAL_WINDOW = 20
# LOCAL_WINDOW = None


class Map(object):
    def __init__(self):
        self.frames = []
        self.keyframe = []
        self.points = []
        self.max_frame_id = 0  # 0 is the first frame id
        self.max_point_id = 0  # 0 is the first point id

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
        ret['max_frame'] = self.max_frame_id
        ret['max_point'] = self.max_point_id
        return json.dumps(ret)

    # FIXME: according to new changes
    def deserialize(self, s):
        ret = json.loads(s)
        self.max_frame_id = ret['max_frame']
        self.max_point_id = ret['max_point']
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
    def add_points(self, points4d, mask_pts4d, f1, f2, idx1, idx2, img1, check_parallax=False):
        assert(points4d.shape[0] == len(idx1))
        new_pts_count = 0
        out_mask_pts4d = np.full(points4d.shape[0], False, dtype=bool)
        if mask_pts4d is None:
            mask_pts4d = np.full(points4d.shape[0], True, dtype=bool)
        for i, p in enumerate(points4d):
            if not mask_pts4d[i]:
                #print('p[%d] not good' % i)
                continue

            # check parallax is large enough (this is going to create problem when the motion is almost zero)
            if check_parallax is True:
                Rwc1 = np.linalg.inv(f1.pose[:3, :3])
                Rwc2 = np.linalg.inv(f2.pose[:3, :3])
                r1 = np.dot(Rwc1, add_ones(f1.kpsn[idx1[i]]))
                r2 = np.dot(Rwc2, add_ones(f2.kpsn[idx2[i]]))
                cos_parallax = r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                if cos_parallax > constants.kCosMinParallax:
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
            err1 = (uv1[0:2] / uv1[2]) - f1.kpsu[idx1[i]]
            err1 = np.sum(err1**2)            
            err2 = (uv2[0:2] / uv2[2]) - f2.kpsu[idx2[i]]
            err2 = np.sum(err2**2)
            # TODO: put a parameter here and check by using covariance and chi-square error
            if err1 > 2 or err2 > 2:
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
    def getLocalWindowMap(self, local_window):
        frames = self.frames[-local_window:]  # get the last N frames  
        frame_id_set = set([f.id for f in frames])                 
        points = []
        point_id_set = set()           
        frames_ref = []   # reference frames, i.e. those frames not in frames that see points in frames            
        for f in frames:
            f_points = [p for p in f.points if (p is not None)] 
            for p in f_points: 
                if p.id not in point_id_set and not p.is_bad:
                    #if p.id == 6:
                    #    print('point ', p.id, 'in frames ', [f.id for f in p.frames])
                    points.append(p)
                    point_id_set.add(p.id)        
                    point_frames = [p_frame for p_frame in p.frames if p_frame.id not in frame_id_set] 
                    for p_frame in point_frames: 
                        #if p_frame.id not in frame_id_set:                        
                        frames_ref.append(p_frame)
                        frame_id_set.add(p_frame.id)             
        frames_ref = sorted(frames_ref, key=lambda x:x.id)                                                                  
        return frames, points, frames_ref

    # remove points which have a big reprojection error 
    def cullMapPoints(self, points): 
        #print('map points: ', sorted([p.id for p in self.points]))
        #print('points: ', sorted([p.id for p in points]))           
        culled_pt_count = 0
        for p in points:
            # compute reprojection error
            errs = []
            for f, idx in zip(p.frames, p.idxs):
                uv = f.kpsu[idx]
                proj = f.project_map_point(p)
                invSigma2 = Frame.detector.vInvLevelSigma2[f.octaves[idx]]
                errs.append(np.linalg.norm(proj-uv)*invSigma2)
            # cull
            chi2Mono = 5.991 # chi-square 2 DOFs  (Hartley Zisserman pg 119)
            if np.mean(errs) > chi2Mono:
                culled_pt_count += 1
                #print('removing point: ',p.id, 'from frames: ', [f.id for f in p.frames])
                #self.points.remove(p)
                #p.delete()
                self.remove_point(p)
        print("culled: %d points" % (culled_pt_count))        

    def optimize(self, local_window=LOCAL_WINDOW, fix_points=False, verbose=False, rounds=50):
        #err = optimizer_g2o.optimizeNormalized(self.frames, self.points, local_window, fix_points, verbose, rounds)
        err = optimizer_g2o.optimization(self.frames, self.points, local_window, fix_points, verbose, rounds)        
        self.cullMapPoints(self.points)

        return err

    def localOptimize(self, local_window=LOCAL_WINDOW, verbose = False, rounds=10):
        frames, points, frames_ref = self.getLocalWindowMap(local_window)
        print('local optimization window: ', [f.id for f in frames])        
        print('                     refs: ', [f.id for f in frames_ref])
        #print('                   points: ', sorted([p.id for p in points]))        
        #err = optimizer_g2o.optimize(frames, points, None, False, verbose, rounds)  
        err = optimizer_g2o.localOptimization(frames_ref, frames, points, False, verbose, rounds)
        self.cullMapPoints(points)                
        return err 
