# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/HelloPangolin

import sys

import pyslam.config as config

import OpenGL.GL as gl
import pypangolin as pangolin
import glutils

import numpy as np

def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        
        # Render OpenGL Cube
        pangolin.glDrawColouredCube()

        # Draw Point Cloud
        points = np.random.random((100000, 3)) * 10
        colors = np.zeros((len(points), 3))
        colors[:, 1] = 1 -points[:, 0] / 10.
        colors[:, 2] = 1 - points[:, 1] / 10.
        colors[:, 0] = 1 - points[:, 2] / 10.

        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        # access numpy array directly(without copying data), array should be contiguous.
        glutils.DrawPoints(points, colors)    

        pangolin.FinishFrame()



if __name__ == '__main__':
    main()
