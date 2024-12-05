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

import platform 

import config

import time
import math 
import torch.multiprocessing as mp 

import pypangolin as pangolin
import OpenGL.GL as gl
import numpy as np

#import open3d as o3d # apparently, this generates issues under mac

from map import Map

from utils_geom import inv_T, align_trajs_with_svd
from utils_sys import Printer
from utils_mp import MultiprocessingManager
from utils_data import empty_queue

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from slam import Slam  # Only imported when type checking, not at runtime
    

kUiWidth = 180

kDefaultSparsePointSize = 2
kDefaultDensePointSize = 2

kViewportWidth = 1024
kViewportHeight = 550
   
kDrawReferenceCamera = True   

kMinWeightForDrawingCovisibilityEdge=100

kAlignGroundTruthEveryNKeyframes = 10
kAlignGroundTruthEveryNFrames = 30

kRefreshDurationTime = 0.03 # [s]



class Viewer3DMapInput(object): 
    def __init__(self):
        self.cur_frame_id = None
        self.cur_pose = None
        self.cur_pose_timestamp = None
        self.predicted_pose = None 
        self.reference_pose = None 
        self.poses = [] 
        self.pose_timestamps = []
        self.points = [] 
        self.colors = []         
        self.covisibility_graph = []
        self.spanning_tree = []        
        self.loops = []       
        self.gt_trajectory = None 
        self.gt_timestamps = None    
        self.align_gt_with_scale = False 
        
        
class Viewer3DDenseInput(object): 
    def __init__(self):
        self.point_cloud = None
        self.mesh = None
        
              
class Viewer3DVoInput(object): 
    def __init__(self):
        self.poses = [] 
        self.pose_timestamps = []        
        self.traj3d_est = []   # estimated trajectory 
        self.traj3d_gt = []    # ground truth trajectory            
        

class Viewer3D(object):
    def __init__(self, scale=0.1):
        self.scale = scale
        
        self.map_state = None      # type: Viewer3DMapInput
        self.vo_state = None       # type: Viewer3DVoInput
        self.dense_state = None    # type: Viewer3DDenseInput
                
        self.gt_trajectory = None
        self.gt_timestamps = None
        self.align_gt_with_scale = False
        self.estimated_trajectory = None
        self.estimated_trajectory_timestamps = None

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.qmap = self.mp_manager.Queue()
        self.qvo = self.mp_manager.Queue()
        self.qdense = self.mp_manager.Queue()
                
        self._is_running  = mp.Value('i',1)
        self._is_paused = mp.Value('i',0)
        self._is_map_save = mp.Value('i',0)
        self._do_step = mp.Value('i',0)
        self._do_reset = mp.Value('i',0)
        self._is_gt_set = mp.Value('i',0)
        self.vp = mp.Process(target=self.viewer_run,
                          args=(self.qmap, self.qvo, self.qdense,
                                self._is_running,self._is_paused,self._is_map_save, self._do_step, self._do_reset, self._is_gt_set))
        self.vp.daemon = True
        self.vp.start()
        
    def set_gt_trajectory(self, gt_trajectory, gt_timestamps, align_with_scale=False):
        if len(gt_timestamps) > 0:
            self.gt_trajectory = gt_trajectory
            self.gt_timestamps = gt_timestamps
            self.align_gt_with_scale = align_with_scale
            self._is_gt_set.value = 0
            print(f'groundtruth shape: {gt_trajectory.shape}')

    def quit(self):
        print('Viewer3D: quitting...')        
        self._is_running.value = 0
        self.vp.join()    
        #pangolin.Quit()
        print('Viewer3D: done')   
        
    def is_paused(self):
        return (self._is_paused.value == 1)       
    
    def is_map_save(self):
        is_map_save = (self._is_map_save.value == 1) 
        if is_map_save:
            self._is_map_save.value = 0
        return is_map_save   
    
    def do_step(self):
        do_step = (self._do_step.value == 1)
        if do_step:
            self._do_step.value = 0
        return do_step   
    
    def reset(self):
        do_reset = (self._do_reset.value == 1)
        if do_reset:
            self._do_reset.value = 0
        return do_reset       

    def viewer_run(self, qmap, qvo, qdense, is_running, is_paused, is_map_save, do_step, do_reset, is_gt_set):
        self.viewer_init(kViewportWidth, kViewportHeight)
        # init local vars for the the process 
        self.thread_gt_trajectory = None
        self.thread_gt_timestamps = None
        self.thread_align_gt_with_scale = False
        self.thread_gt_aligned = False
        self.thread_last_num_poses_gt_was_aligned = 0
        self.thread_last_frame_id_gt_was_aligned = 0
        while not pangolin.ShouldQuit() and (is_running.value == 1):
            ts = time.time()
            self.viewer_refresh(qmap, qvo, qdense, is_paused, is_map_save, do_step, do_reset, is_gt_set)
            sleep = (time.time() - ts) - kRefreshDurationTime         
            if sleep > 0:
                time.sleep(sleep)     
                
        empty_queue(qmap)   # empty the queue before exiting
        empty_queue(qvo)    # empty the queue before exiting
        empty_queue(qdense) # empty the queue before exiting                                  
        print('Viewer3D: loop exit...')    

    def viewer_init(self, w, h):
        # pangolin.ParseVarsFile('app.cfg')

        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        viewpoint_x =   0 * self.scale
        viewpoint_y = -40 * self.scale
        viewpoint_z = -80 * self.scale
        viewpoint_f = 1000
            
        self.proj = pangolin.ProjectionMatrix(w, h, viewpoint_f, viewpoint_f, w//2, h//2, 0.1, 5000)
        self.look_view = pangolin.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
        self.scam = pangolin.OpenGlRenderState(self.proj, self.look_view)
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, kUiWidth/w, 1.0, -w/h)
        self.dcam.SetHandler(pangolin.Handler3D(self.scam))

        self.panel = pangolin.CreatePanel('ui')
        self.panel.SetBounds(0.0, 1.0, 0.0, kUiWidth/w)

        self.do_follow = True
        self.is_following = True 
        
        self.draw_cameras = True
        self.draw_covisibility = True        
        self.draw_spanning_tree = True           
        self.draw_loops = True   
        self.draw_dense = True
        self.draw_sparse = True
        
        self.draw_wireframe = False             

        #self.button = pangolin.VarBool('ui.Button', value=False, toggle=False)
        
        self.checkboxFollow = pangolin.VarBool('ui.Follow', value=True, toggle=True)
        self.checkboxCams = pangolin.VarBool('ui.Draw Cameras', value=True, toggle=True)
        self.checkboxCovisibility = pangolin.VarBool('ui.Draw Covisibility', value=True, toggle=True)  
        self.checkboxSpanningTree = pangolin.VarBool('ui.Draw Tree', value=True, toggle=True)    
        self.checkboxLoops = pangolin.VarBool('ui.Draw Loops', value=True, toggle=True)           
        self.checkboxGT = pangolin.VarBool('ui.Draw Ground Truth', value=False, toggle=True)    
        self.checkboxPredicted = pangolin.VarBool('ui.Draw Predicted', value=False, toggle=True)
        self.checkboxDrawSparseCloud = pangolin.VarBool('ui.Draw Sparse Map', value=True, toggle=True)        
        self.checkboxDrawDenseCloud = pangolin.VarBool('ui.Draw Dense Map', value=True, toggle=True)                               
        self.checkboxGrid = pangolin.VarBool('ui.Grid', value=True, toggle=True)           
        self.checkboxPause = pangolin.VarBool('ui.Pause', value=False, toggle=True)
        self.buttonSave = pangolin.VarBool('ui.Save', value=False, toggle=False)   
        self.buttonStep = pangolin.VarBool('ui.Step', value=False, toggle=False)
        self.buttonReset = pangolin.VarBool('ui.Reset', value=False, toggle=False)                      
        #self.float_slider = pangolin.VarFloat('ui.Float', value=3, min=0, max=5)
        #self.float_log_slider = pangolin.VarFloat('ui.Log_scale var', value=3, min=1, max=1e4, logscale=True)
        self.sparsePointSizeSlider = pangolin.VarInt('ui.Sparse Point Size', value=kDefaultSparsePointSize, min=1, max=10)
        self.densePointSizeSlider = pangolin.VarInt('ui.Dense Point Size', value=kDefaultDensePointSize, min=1, max=10)
        self.checkboxWireframe = pangolin.VarBool('ui.Mesh Wireframe', value=False, toggle=True)

        self.sparsePointSize = self.sparsePointSizeSlider.Get()
        self.densePointSize = self.densePointSizeSlider.Get()

        self.Twc = pangolin.OpenGlMatrix()
        self.Twc.SetIdentity()
        # print("self.Twc.m",self.Twc.m)


    def viewer_refresh(self, qmap, qvo, qdense, is_paused, is_map_save, do_step, do_reset, is_gt_set):

        # NOTE: take the last elements in the queues
        
        while not qmap.empty():
            self.map_state = qmap.get()

        while not qvo.empty():
            self.vo_state = qvo.get()
            
        while not qdense.empty():
            self.dense_state = qdense.get()

        # if pangolin.Pushed(self.button):
        #    print('You Pushed a button!')

        self.do_follow = self.checkboxFollow.Get()
        self.is_grid = self.checkboxGrid.Get()        
        self.draw_cameras = self.checkboxCams.Get()
        self.draw_covisibility = self.checkboxCovisibility.Get()
        self.draw_spanning_tree = self.checkboxSpanningTree.Get()
        self.draw_loops = self.checkboxLoops.Get()
        self.draw_gt = self.checkboxGT.Get()
        self.draw_predicted = self.checkboxPredicted.Get()
        self.draw_wireframe = self.checkboxWireframe.Get()
        self.draw_dense = self.checkboxDrawDenseCloud.Get()
        self.draw_sparse = self.checkboxDrawSparseCloud.Get()
        
        #if pangolin.Pushed(self.checkboxPause):
        if self.checkboxPause.Get():
            is_paused.value = 1  
        else:
            is_paused.value = 0  
            
        if pangolin.Pushed(self.buttonSave): 
            self.checkboxPause.SetVal(True)
            is_paused.value = 1
            is_map_save.value = 1            
            
        if pangolin.Pushed(self.buttonStep):
            if not is_paused.value:
                self.checkboxPause.SetVal(True)
                is_paused.value = 1            
            do_step.value = 1 
            
        if pangolin.Pushed(self.buttonReset):
            # if not is_paused.value:
            #     self.checkboxPause.SetVal(True)
            #     is_paused.value = 1            
            do_reset.value = 1
                                
        # self.sparsePointSizeSlider.SetVal(int(self.float_slider))
        self.sparsePointSize = self.sparsePointSizeSlider.Get()
        self.densePointSize = self.densePointSizeSlider.Get()
            
        if self.do_follow and self.is_following:
            self.scam.Follow(self.Twc, True)
        elif self.do_follow and not self.is_following:
            self.scam.SetModelViewMatrix(self.look_view)
            self.scam.Follow(self.Twc, True)
            self.is_following = True
        elif not self.do_follow and self.is_following:
            self.is_following = False            

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        
        self.dcam.Activate(self.scam)

        if self.is_grid:
            Viewer3D.drawPlane(scale=self.scale)

        # ==============================
        # draw map 
        if self.map_state is not None:
            
            if not is_gt_set.value and self.map_state.gt_trajectory is not None:
                self.thread_gt_trajectory = np.array(self.map_state.gt_trajectory)
                self.thread_gt_timestamps = np.array(self.map_state.gt_timestamps)
                self.thread_align_gt_with_scale = self.map_state.align_gt_with_scale
                is_gt_set.value = 1
            
            if self.map_state.cur_pose is not None:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                gl.glLineWidth(2)                
                pangolin.DrawCamera(self.map_state.cur_pose, self.scale)
                gl.glLineWidth(1)                
                self.updateTwc(self.map_state.cur_pose)
                
            if self.map_state.predicted_pose is not None and self.draw_predicted:
                # draw predicted pose in red
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawCamera(self.map_state.predicted_pose, self.scale)           
                
            if self.thread_gt_timestamps is not None: 
                if self.draw_gt:                
                    # align the gt to the estimated trajectory every 'kAlignGroundTruthEveryNKeyframes' frames;
                    # the more estimated frames we have the better the alignment! 
                    condition1 = len(self.map_state.poses) > kAlignGroundTruthEveryNKeyframes + self.thread_last_num_poses_gt_was_aligned
                    condition2 = self.map_state.cur_frame_id > kAlignGroundTruthEveryNFrames + self.thread_last_frame_id_gt_was_aligned  # this is useful when we are not generating new kfs
                    if condition1 or condition2:
                        try:
                            estimated_trajectory = np.array([self.map_state.poses[i][0:3,3] for i in range(len(self.map_state.poses))], dtype=np.float32)
                            align_trajs_with_svd(self.map_state.pose_timestamps, estimated_trajectory, self.thread_gt_timestamps, self.thread_gt_trajectory, align_gt=True, compute_align_error=False, find_scale=self.thread_align_gt_with_scale)
                            self.thread_gt_aligned = True
                            self.thread_last_num_poses_gt_was_aligned = len(self.map_state.poses)
                            self.thread_last_frame_id_gt_was_aligned = self.map_state.cur_frame_id
                        except Exception as e:
                            print(f'Viewer3D: viewer_refresh - align_gt_with_svd failed: {e}')
                    if self.thread_gt_aligned:
                        gt_lines = [[*self.thread_gt_trajectory[i], *self.thread_gt_trajectory[i+1]] for i in range(len(self.thread_gt_trajectory)-1)]
                        gl.glLineWidth(1)
                        gl.glColor3f(1.0, 0.0, 0.0)
                        pangolin.DrawLines(gt_lines,2)                      
                    
            if len(self.map_state.poses) >1:
                # draw keyframe poses in green
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.map_state.poses[:], self.scale)

            if self.draw_sparse and len(self.map_state.points)>0:
                # draw keypoints with their color
                gl.glPointSize(self.sparsePointSize)
                #gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(self.map_state.points, self.map_state.colors)    
                
            if self.map_state.reference_pose is not None and kDrawReferenceCamera:
                # draw predicted pose in purple
                gl.glColor3f(0.5, 0.0, 0.5)
                gl.glLineWidth(2)                
                pangolin.DrawCamera(self.map_state.reference_pose, self.scale)      
                gl.glLineWidth(1)          
                
            if len(self.map_state.covisibility_graph)>0:
                if self.draw_covisibility:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawLines(self.map_state.covisibility_graph,3)                                             
                    
            if len(self.map_state.spanning_tree)>0:
                if self.draw_spanning_tree:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 0.0, 1.0)
                    pangolin.DrawLines(self.map_state.spanning_tree,3)              
                    
            if len(self.map_state.loops)>0:
                if self.draw_loops:
                    gl.glLineWidth(2)
                    gl.glColor3f(0.5, 0.0, 0.5)
                    pangolin.DrawLines(self.map_state.loops,3)        
                    gl.glLineWidth(1)                                               


        # ==============================
        # draw dense stuff 
        if self.dense_state is not None:
            if self.draw_dense:
                if self.dense_state.mesh is not None:
                    vertices = self.dense_state.mesh[0]
                    triangles = self.dense_state.mesh[1]
                    colors = self.dense_state.mesh[2]
                    pangolin.DrawMesh(vertices, triangles, colors, self.draw_wireframe)    
                elif self.dense_state.point_cloud is not None: 
                    pc_points = self.dense_state.point_cloud[0]
                    pc_colors = self.dense_state.point_cloud[1]
                    gl.glPointSize(self.densePointSize)
                    pangolin.DrawPoints(pc_points, pc_colors)
            
            
        # ==============================
        # draw vo 
        if self.vo_state is not None:
            if self.vo_state.poses.shape[0] >= 2:
                # draw poses in green
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.vo_state.poses[:-1], self.scale)

            if self.vo_state.poses.shape[0] >= 1:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                current_pose = self.vo_state.poses[-1:]
                pangolin.DrawCameras(current_pose, self.scale)
                self.updateTwc(current_pose[0])

            if self.vo_state.traj3d_est.shape[0] != 0:
                # draw blue estimated trajectory 
                gl.glPointSize(self.sparsePointSize)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawLine(self.vo_state.traj3d_est)

            if self.vo_state.traj3d_gt.shape[0] != 0:
                # draw red ground-truth trajectory 
                gl.glPointSize(self.sparsePointSize)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawLine(self.vo_state.traj3d_gt)                


        pangolin.FinishFrame()


    # draw sparse map
    def draw_map(self, slam: 'Slam'):
        if self.qmap is None:
            return
        map = slam.map                     # type: Map
        map_state = Viewer3DMapInput()    
        
        map_state.cur_frame_id = slam.tracking.f_cur.id if slam.tracking.f_cur is not None else -1
        
        if map.num_frames() > 0: 
            map_state.cur_pose = map.get_frame(-1).Twc.copy()
            map_state.cur_pose_timestamp = map.get_frame(-1).timestamp
            
        if slam.tracking.predicted_pose is not None: 
            map_state.predicted_pose = slam.tracking.predicted_pose.inverse().matrix().copy()
            
        if False: 
            if slam.tracking.kf_ref is not None: 
                map_state.reference_pose = slam.tracking.kf_ref.Twc.copy()            
            
        num_map_keyframes = map.num_keyframes()
        keyframes = map.get_keyframes()
        if num_map_keyframes>0:       
            for kf in keyframes:
                map_state.poses.append(kf.Twc)
                map_state.pose_timestamps.append(kf.timestamp)
        map_state.poses = np.array(map_state.poses, dtype=np.float32)
        map_state.pose_timestamps = np.array(map_state.pose_timestamps, dtype=np.float64)

        num_map_points = map.num_points()
        if num_map_points>0:
            for i,p in enumerate(map.get_points()):                
                map_state.points.append(p.pt)           
                map_state.colors.append(np.flip(p.color))              
        map_state.points = np.array(map_state.points)          
        map_state.colors = np.array(map_state.colors)/256. 
        
        for kf in keyframes:
            for kf_cov in kf.get_covisible_by_weight(kMinWeightForDrawingCovisibilityEdge):
                if kf_cov.kid > kf.kid:
                    map_state.covisibility_graph.append([*kf.Ow, *kf_cov.Ow])
            if kf.parent is not None: 
                map_state.spanning_tree.append([*kf.Ow, *kf.parent.Ow])
            for kf_loop in kf.get_loop_edges():
                if kf_loop.kid > kf.kid:
                    map_state.loops.append([*kf.Ow, *kf_loop.Ow])                
        map_state.covisibility_graph = np.array(map_state.covisibility_graph)   
        map_state.spanning_tree = np.array(map_state.spanning_tree)   
        map_state.loops = np.array(map_state.loops)                     
                            
        if self.gt_trajectory is not None:
            if not self._is_gt_set.value:
                map_state.gt_trajectory = np.array(self.gt_trajectory, dtype=np.float64)
                map_state.gt_timestamps = np.array(self.gt_timestamps, dtype=np.float64)
                map_state.align_gt_with_scale = self.align_gt_with_scale
                             
        self.qmap.put(map_state)

    def draw_dense_map(self, slam: 'Slam'):
        if self.qdense is None:
            return
        dense_map_output = slam.get_dense_map()
        if dense_map_output is not None:
            self.draw_dense_geometry(dense_map_output.point_cloud, dense_map_output.mesh)

    # inputs: 
    #   point_cloud: o3d.geometry.PointCloud or VolumeIntegratorPointCloud (see the file volumetric_integrator.py)
    #   mesh: o3d.geometry.TriangleMesh or VolumeIntegrationMesh (see the file volumetric_integrator.py)
    def draw_dense_geometry(self, point_cloud=None, mesh=None):
        if self.qdense is None:
            return
        dense_state = Viewer3DDenseInput()
        if mesh is not None:
            dense_state.mesh = (np.array(mesh.vertices), np.array(mesh.triangles), np.array(mesh.vertex_colors)) #,np.array(mesh.vertex_normals))
        else:
            if point_cloud is not None:            
                dense_state.point_cloud = (np.array(point_cloud.points), np.array(point_cloud.colors))
            else:
                Printer.orange('WARNING: both point_cloud and mesh are None')

        self.qdense.put(dense_state)


    def draw_vo(self, vo):
        if self.qvo is None:
            return
        vo_state = Viewer3DVoInput()
        vo_state.poses = np.array(vo.poses)
        vo_state.pose_timestamps = np.array(vo.pose_timestamps, dtype=np.float64)
        vo_state.traj3d_est = np.array(vo.traj3d_est).reshape(-1,3)
        vo_state.traj3d_gt = np.array(vo.traj3d_gt).reshape(-1,3)        
        
        self.qvo.put(vo_state)
        

    def updateTwc(self, pose):
        self.Twc.m = pose


    @staticmethod
    def drawPlane(num_divs=200, div_size=10, scale=1.0):
        gl.glLineWidth(0.5)
        # Plane parallel to x-z at origin with normal -y
        div_size = scale*div_size
        minx = -num_divs*div_size
        minz = -num_divs*div_size
        maxx = num_divs*div_size
        maxz = num_divs*div_size
        #gl.glLineWidth(2)
        #gl.glColor3f(0.7,0.7,1.0)
        gl.glColor3f(0.7,0.7,0.7)
        gl.glBegin(gl.GL_LINES)
        for n in range(2*num_divs):
            gl.glVertex3f(minx+div_size*n,0,minz)
            gl.glVertex3f(minx+div_size*n,0,maxz)
            gl.glVertex3f(minx,0,minz+div_size*n)
            gl.glVertex3f(maxx,0,minz+div_size*n)
        gl.glEnd()
        gl.glLineWidth(1)


