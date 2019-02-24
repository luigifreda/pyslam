# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/HelloPangolin
# https://github.com/stevenlovegrove/Pangolin/blob/master/examples/SimplePlot

import sys
sys.path.append("../../")
import config

import OpenGL.GL as gl 
import pangolin

import numpy as np



def main():
    # Create OpenGL window in single line
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



    # Data logger object
    log = pangolin.DataLog()

    # Optionally add named labels
    labels = ['sin(t)', 'cos(t)', 'sin(t)+cos(t)']
    log.SetLabels(labels)

    # OpenGL 'view' of data. We might have many views of the same data.
    tinc = 0.03
    plotter = pangolin.Plotter(log, 0.0, 6.0*np.pi/tinc, -2.0, 2.0, np.pi/(6*tinc), 0.5)
    plotter.SetBounds(0.05, 0.3, 0.0, 0.4)
    plotter.Track('$i')

    # Add some sample annotations to the plot
    plotter.AddMarker(pangolin.Marker.Vertical, -1000, pangolin.Marker.LessThan, 
        pangolin.Colour.Blue().WithAlpha(0.2))
    plotter.AddMarker(pangolin.Marker.Horizontal, 100, pangolin.Marker.GreaterThan, 
        pangolin.Colour.Red().WithAlpha(0.2))
    plotter.AddMarker(pangolin.Marker.Horizontal,  10, pangolin.Marker.Equal, 
        pangolin.Colour.Green().WithAlpha(0.2))

    pangolin.DisplayBase().AddDisplay(plotter)


    t = 0
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Plot line
        log.Log(np.sin(t), np.cos(t), np.sin(t)+np.cos(t))
        t += tinc


        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        # Render OpenGL 3D Cube
        pangolin.glDrawColouredCube()


        pangolin.FinishFrame()




if __name__ == '__main__':
    main()