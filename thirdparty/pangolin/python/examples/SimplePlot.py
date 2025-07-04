# https://github.com/stevenlovegrove/Pangolin/blob/master/examples/SimplePlot

import sys 
sys.path.append("../../")

import OpenGL.GL as gl 
import pypangolin as pangolin

import numpy as np



def main():
    # Create OpenGL window in single line
    pangolin.CreateWindowAndBind('Main', 640, 480)

    # Data logger object
    log = pangolin.DataLog()

    # Optionally add named labels
    labels = ['sin(t)', 'cos(t)', 'sin(t)+cos(t)']
    log.SetLabels(labels)

    # OpenGL 'view' of data. We might have many views of the same data.
    tinc = 0.03
    plotter = pangolin.Plotter(log, 0.0, 4.0*np.pi/tinc, -2.0, 2.0, np.pi/(4*tinc), 0.5)
    plotter.SetBounds(0.0, 1.0, 0.0, 1.0)
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

        log.Log(np.sin(t), np.cos(t), np.sin(t)+np.cos(t))
        t += tinc

        pangolin.FinishFrame()




if __name__ == '__main__':
    main()