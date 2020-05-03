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

import config
import math 
import multiprocessing as mp 
from multiprocessing import Process, Queue, Value
import pangolin
import OpenGL.GL as gl
import numpy as np
from utils_geom import inv_T 


kUiWidth = 180
kDefaultPointSize = 2
kViewportWidth = 1024
#kViewportHeight = 768
kViewportHeight = 550
kDrawCameraPrediction = False   
kDrawReferenceCamera = True   

kMinWeightForDrawingCovisibilityEdge=100


class Viewer3DMapElement(object): 
    def __init__(self):
        self.cur_pose = None 
        self.predicted_pose = None 
        self.reference_pose = None 
        self.poses = [] 
        self.points = [] 
        self.colors = []         
        self.covisibility_graph = []
        self.spanning_tree = []        
        self.loops = []            
        
              
class Viewer3DVoElement(object): 
    def __init__(self):
        self.poses = [] 
        self.traj3d_est = []   # estimated trajectory 
        self.traj3d_gt = []    # ground truth trajectory            
        

class Viewer3D(object):
    def __init__(self):
        self.map_state = None
        self.qmap = Queue()
        self.vo_state = None
        self.qvo = Queue()        
        self._is_running  = Value('i',1)
        self._is_paused = Value('i',1)
        self.vp = Process(target=self.viewer_thread,
                          args=(self.qmap, self.qvo,self._is_running ,self._is_paused,))
        self.vp.daemon = True
        self.vp.start()

    def quit(self):
        self._is_running.value = 0
        self.vp.join()
        #pangolin.Quit()
        print('Viewer stopped')   
        
    def is_paused(self):
        return (self._is_paused.value == 1)       

    def viewer_thread(self, qmap, qvo, is_running, is_paused):
        self.viewer_init(kViewportWidth, kViewportHeight)
        while not pangolin.ShouldQuit() and (is_running.value == 1):
            self.viewer_refresh(qmap, qvo, is_paused)
        print('Quitting viewer...')    

    def viewer_init(self, w, h):
        # pangolin.ParseVarsFile('app.cfg')

        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        viewpoint_x = 0
        viewpoint_y = -40
        viewpoint_z = -80
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

        #self.button = pangolin.VarBool('ui.Button', value=False, toggle=False)
        self.checkboxFollow = pangolin.VarBool('ui.Follow', value=True, toggle=True)
        self.checkboxCams = pangolin.VarBool('ui.Draw Cameras', value=True, toggle=True)
        self.checkboxCovisibility = pangolin.VarBool('ui.Draw Covisibility', value=True, toggle=True)  
        self.checkboxSpanningTree = pangolin.VarBool('ui.Draw Tree', value=True, toggle=True)                
        self.checkboxGrid = pangolin.VarBool('ui.Grid', value=True, toggle=True)           
        self.checkboxPause = pangolin.VarBool('ui.Pause', value=False, toggle=True)             
        #self.float_slider = pangolin.VarFloat('ui.Float', value=3, min=0, max=5)
        #self.float_log_slider = pangolin.VarFloat('ui.Log_scale var', value=3, min=1, max=1e4, logscale=True)
        self.int_slider = pangolin.VarInt('ui.Point Size', value=kDefaultPointSize, min=1, max=10)  

        self.pointSize = self.int_slider.Get()

        self.Twc = pangolin.OpenGlMatrix()
        self.Twc.SetIdentity()
        # print("self.Twc.m",self.Twc.m)


    def viewer_refresh(self, qmap, qvo, is_paused):

        while not qmap.empty():
            self.map_state = qmap.get()

        while not qvo.empty():
            self.vo_state = qvo.get()

        # if pangolin.Pushed(self.button):
        #    print('You Pushed a button!')

        self.do_follow = self.checkboxFollow.Get()
        self.is_grid = self.checkboxGrid.Get()        
        self.draw_cameras = self.checkboxCams.Get()
        self.draw_covisibility = self.checkboxCovisibility.Get()
        self.draw_spanning_tree = self.checkboxSpanningTree.Get()
        
        #if pangolin.Pushed(self.checkboxPause):
        if self.checkboxPause.Get():
            is_paused.value = 0  
        else:
            is_paused.value = 1  
                    
        # self.int_slider.SetVal(int(self.float_slider))
        self.pointSize = self.int_slider.Get()
            
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
            Viewer3D.drawPlane()

        # ==============================
        # draw map 
        if self.map_state is not None:
            if self.map_state.cur_pose is not None:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                gl.glLineWidth(2)                
                pangolin.DrawCamera(self.map_state.cur_pose)
                gl.glLineWidth(1)                
                self.updateTwc(self.map_state.cur_pose)
                
            if self.map_state.predicted_pose is not None and kDrawCameraPrediction:
                # draw predicted pose in red
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawCamera(self.map_state.predicted_pose)           
                
            if len(self.map_state.poses) >1:
                # draw keyframe poses in green
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.map_state.poses[:])

            if len(self.map_state.points)>0:
                # draw keypoints with their color
                gl.glPointSize(self.pointSize)
                #gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(self.map_state.points, self.map_state.colors)    
                
            if self.map_state.reference_pose is not None and kDrawReferenceCamera:
                # draw predicted pose in purple
                gl.glColor3f(0.5, 0.0, 0.5)
                gl.glLineWidth(2)                
                pangolin.DrawCamera(self.map_state.reference_pose)      
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
                if self.draw_spanning_tree:
                    gl.glLineWidth(2)
                    gl.glColor3f(0.5, 0.0, 0.5)
                    pangolin.DrawLines(self.map_state.loops,3)        
                    gl.glLineWidth(1)                                               

        # ==============================
        # draw vo 
        if self.vo_state is not None:
            if self.vo_state.poses.shape[0] >= 2:
                # draw poses in green
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.vo_state.poses[:-1])

            if self.vo_state.poses.shape[0] >= 1:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                current_pose = self.vo_state.poses[-1:]
                pangolin.DrawCameras(current_pose)
                self.updateTwc(current_pose[0])

            if self.vo_state.traj3d_est.shape[0] != 0:
                # draw blue estimated trajectory 
                gl.glPointSize(self.pointSize)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawLine(self.vo_state.traj3d_est)

            if self.vo_state.traj3d_gt.shape[0] != 0:
                # draw red ground-truth trajectory 
                gl.glPointSize(self.pointSize)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawLine(self.vo_state.traj3d_gt)                


        pangolin.FinishFrame()


    def draw_map(self, slam):
        if self.qmap is None:
            return
        map = slam.map 
        map_state = Viewer3DMapElement()        
        
        if map.num_frames() > 0: 
            map_state.cur_pose = map.get_frame(-1).Twc.copy()
            
        if slam.tracking.predicted_pose is not None: 
            map_state.predicted_pose = slam.tracking.predicted_pose.inverse().matrix().copy()
            
        if slam.tracking.kf_ref is not None: 
            reference_pose = slam.tracking.kf_ref.Twc.copy()            
            
        num_map_keyframes = map.num_keyframes()
        keyframes = map.get_keyframes()
        if num_map_keyframes>0:       
            for kf in keyframes:
                map_state.poses.append(kf.Twc)  
        map_state.poses = np.array(map_state.poses)

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
                                             
        self.qmap.put(map_state)


    def draw_vo(self, vo):
        if self.qvo is None:
            return
        vo_state = Viewer3DVoElement()
        vo_state.poses = np.array(vo.poses)
        vo_state.traj3d_est = np.array(vo.traj3d_est).reshape(-1,3)
        vo_state.traj3d_gt = np.array(vo.traj3d_gt).reshape(-1,3)        
        
        self.qvo.put(vo_state)


    def updateTwc(self, pose):
        self.Twc.m = pose


    @staticmethod
    def drawPlane(num_divs=200, div_size=10):
        # Plane parallel to x-z at origin with normal -y
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


