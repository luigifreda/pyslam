# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/SimpleDisplayImage

import sys
sys.path.append("../../")
import config

import numpy as np

import OpenGL.GL as gl
import pangolin


def random_image(w, h):   # HWC
    return (np.ones((h, w, 3), 'uint8') * 
        np.random.randint(256, size=3, dtype='uint8'))


def main():
    # Create OpenGL window in single line
    pangolin.CreateWindowAndBind('Main', 640, 480)

    # 3D Mouse handler requires depth testing to be enabled
    gl.glEnable(gl.GL_DEPTH_TEST)


    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin.ModelViewLookAt(-1, 1, -1, 0, 0, 0, pangolin.AxisDirection.AxisY))

    # Aspect ratio allows us to constrain width and height whilst fitting within specified
    # bounds. A positive aspect ratio makes a view 'shrink to fit' (introducing empty bars),
    # whilst a negative ratio makes the view 'grow to fit' (cropping the view).
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(pangolin.Handler3D(scam))

    # This view will take up no more than a third of the windows width or height, and it
    # will have a fixed aspect ratio to match the image that it will display. When fitting
    # within the specified bounds, push to the top-left (as specified by SetLock).
    dimg = pangolin.Display('image')
    dimg.SetBounds(2./3, 1.0, 0.0, 1./3, 640./480)
    dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)


    w, h = 64, 48
    texture = pangolin.GlTexture(w, h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    # Default hooks for exiting (Esc) and fullscreen (tab).
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.95, 0.95, 0.95, 1.0)

        dcam.Activate(scam)
        gl.glColor3f(1.0, 1.0, 1.0)
        pangolin.glDrawColouredCube()

        # Set some random image data and upload to GPU 
        image = random_image(w, h)
        texture.Upload(image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        # display the image
        dimg.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        texture.RenderToViewport()

        pangolin.FinishFrame()



if __name__ == '__main__':
    main()