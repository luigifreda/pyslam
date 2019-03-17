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

from multiprocessing import Process, Queue
import pangolin
import OpenGL.GL as gl
import numpy as np


kUiWidth = 180
kDefaultPointSize = 3
kViewportWidth = 1024
#kViewportHeight = 768
kViewportHeight = 600


class Viewer3D(object):
    def __init__(self):
        self.state_map = None
        self.qmap = Queue()
        self.state_vo = None
        self.qvo = Queue()
        self.vp = Process(target=self.viewer_thread,
                          args=(self.qmap, self.qvo,))
        self.vp.daemon = True
        self.vp.start()


    def viewer_thread(self, qmap, qvo):
        self.viewer_init(kViewportWidth, kViewportHeight)
        while not pangolin.ShouldQuit():
            self.viewer_refresh(qmap, qvo)


    def viewer_init(self, w, h):
        # pangolin.ParseVarsFile('app.cfg')

        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 500, 500, 512, 389, 0.1, 1000),
            pangolin.ModelViewLookAt(0, -10, -20, 0, 0, 0, 0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, kUiWidth/w, 1.0, -w/h)
        self.dcam.SetHandler(pangolin.Handler3D(self.scam))

        # hack to avoid small Pangolin, no idea why it's *2
        # self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
        # self.dcam.Activate()

        self.panel = pangolin.CreatePanel('ui')
        self.panel.SetBounds(0.0, 1.0, 0.0, kUiWidth/w)

        self.do_follow = True
        self.draw_cameras = True

        #self.button = pangolin.VarBool('ui.Button', value=False, toggle=False)
        self.checkboxFollow = pangolin.VarBool('ui.Follow', value=True, toggle=True)
        self.checkboxCams = pangolin.VarBool('ui.Draw Cameras', value=True, toggle=True)
        self.checkboxGrid = pangolin.VarBool('ui.Grid', value=True, toggle=True)              
        #self.float_slider = pangolin.VarFloat('ui.Float', value=3, min=0, max=5)
        #self.float_log_slider = pangolin.VarFloat('ui.Log_scale var', value=3, min=1, max=1e4, logscale=True)
        self.int_slider = pangolin.VarInt('ui.Point Size', value=kDefaultPointSize, min=1, max=10)  

        self.pointSize = self.int_slider.Get()

        self.Twc = pangolin.OpenGlMatrix()
        self.Twc.SetIdentity()
        # print("self.Twc.m",self.Twc.m)


    def viewer_refresh(self, qmap, qvo):

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        while not qmap.empty():
            self.state_map = qmap.get()

        while not qvo.empty():
            self.state_vo = qvo.get()

        # if pangolin.Pushed(self.button):
        #    print('You Pushed a button!')

        self.do_follow = self.checkboxFollow.Get()
        self.is_grid = self.checkboxGrid.Get()        
        self.draw_cameras = self.checkboxCams.Get()
        # self.int_slider.SetVal(int(self.float_slider))
        self.pointSize = self.int_slider.Get()

        self.dcam.Activate(self.scam)
        if self.do_follow:
            self.scam.Follow(self.Twc, True)

        if self.is_grid:
            Viewer3D.drawPlane()

        # draw map 
        if self.state_map is not None:
            if self.state_map[0].shape[0] >= 2:
                # draw poses in green
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.state_map[0][:-1])

            if self.state_map[0].shape[0] >= 1:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                currentPose = self.state_map[0][-1:]
                pangolin.DrawCameras(currentPose)
                self.updateTwc(currentPose)

            if self.state_map[1].shape[0] != 0:
                # draw keypoints with their color
                gl.glPointSize(self.pointSize)
                #gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(self.state_map[1], self.state_map[2])

       # draw vo 
        if self.state_vo is not None:
            if self.state_vo[0].shape[0] >= 2:
                # draw poses in green
                if self.draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(self.state_vo[0][:-1])

            if self.state_vo[0].shape[0] >= 1:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                currentPose = self.state_vo[0][-1:]
                pangolin.DrawCameras(currentPose)
                self.updateTwc(currentPose)

            if self.state_vo[1].shape[0] != 0:
                # draw blue estimated trajectory 
                gl.glPointSize(self.pointSize)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawLine(self.state_vo[1])

            if self.state_vo[2].shape[0] != 0:
                # draw red ground-truth trajectory 
                gl.glPointSize(self.pointSize)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawLine(self.state_vo[2])                


        pangolin.FinishFrame()


    def drawMap(self, map):
        if self.qmap is None:
            return

        poses, pts, colors = [], [], []
        for f in map.frames:
            # invert pose for display only
            poses.append(np.linalg.inv(f.pose))

        for p in map.points:
            pts.append(p.pt)
            # colors.append(p.color)
            colors.append(np.flip(p.color, 0))

        self.qmap.put((np.array(poses), np.array(pts), np.array(colors)/256.0))


    def drawVo(self, vo):
        if self.qvo is None:
            return
        self.qvo.put((np.array(vo.poses), np.array(vo.traj3d_est).reshape(-1,3), np.array(vo.traj3d_gt).reshape(-1,3)))


    def updateTwc(self, pose):
        self.Twc.m = pose[0]

    @staticmethod
    def drawPlane(num_divs=200, div_size=1):
        # Plane parallel to x-z at origin with normal -y
        minx = -num_divs*div_size
        minz = -num_divs*div_size
        maxx = num_divs*div_size
        maxz = num_divs*div_size
        #gl.glLineWidth(2)
        gl.glColor3f(0.7,0.7,1.0)
        gl.glBegin(gl.GL_LINES)
        for n in range(2*num_divs):
            gl.glVertex3f(minx+div_size*n,0,minz)
            gl.glVertex3f(minx+div_size*n,0,maxz)
            gl.glVertex3f(minx,0,minz+div_size*n)
            gl.glVertex3f(maxx,0,minz+div_size*n)
        gl.glEnd()


