# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/HelloPangolin

import sys
sys.path.append("../../")
import pyslam.config as config

import OpenGL.GL as gl
import pypangolin as pangolin
import glutils

import numpy as np

def drawPlane(ndivs=100, ndivsize=1):
    # Plane parallel to x-z at origin with normal -y
    minx = -ndivs*ndivsize
    minz = -ndivs*ndivsize
    maxx = ndivs*ndivsize
    maxz = ndivs*ndivsize
    gl.glLineWidth(1)
    gl.glColor3f(0.7,0.7,1.0)
    gl.glBegin(gl.GL_LINES)
    for n in range(2*ndivs):
        gl.glVertex3f(minx+ndivsize*n,0,minz)
        gl.glVertex3f(minx+ndivsize*n,0,maxz)
        gl.glVertex3f(minx,0,minz+ndivsize*n)
        gl.glVertex3f(maxx,0,minz+ndivsize*n)
    gl.glEnd()

def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)

    
    trajectory = [[0, -6, 6]]
    for i in range(300):
        trajectory.append(trajectory[-1] + np.random.random(3)-0.5)
    trajectory = np.array(trajectory)
    print(trajectory.shape)


    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        
        drawPlane()

        # Render OpenGL Cube
        pangolin.glDrawColouredCube(0.1)

        # Draw Point Cloud
        points = np.random.random((10000, 3)) * 3 - 4
        gl.glPointSize(1)
        gl.glColor3f(1.0, 0.0, 0.0)
        glutils.DrawPoints(points)

        # Draw Point Cloud
        points = np.random.random((10000, 3))
        colors = np.zeros((len(points), 3))
        colors[:, 1] = 1 -points[:, 0]
        colors[:, 2] = 1 - points[:, 1]
        colors[:, 0] = 1 - points[:, 2]
        points = points * 3 + 1
        gl.glPointSize(1)
        glutils.DrawPoints(points, colors)

        # Draw lines
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 0.0)
        #glutils.DrawLine(trajectory)   # consecutive
        glutils.DrawTrajectory(trajectory)
        gl.glColor3f(0.0, 1.0, 0.0)
        glutils.DrawLines2(
            trajectory, 
            trajectory + np.random.randn(len(trajectory), 3), 
            point_size=5)   # separate

        # Draw camera
        pose = np.identity(4)
        pose[:3, 3] = np.random.randn(3)
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)
        glutils.DrawCamera(pose, 0.5, 0.75, 0.8)

        # Draw boxes
        poses = [np.identity(4) for i in range(10)]
        for pose in poses:
            pose[:3, 3] = np.random.randn(3) + np.array([5,-3,0])
        sizes = np.random.random((len(poses), 3))
        gl.glLineWidth(1)
        gl.glColor3f(1.0, 0.0, 1.0)
        glutils.DrawBoxes(poses, sizes)


        pangolin.FinishFrame()



if __name__ == '__main__':
    main()