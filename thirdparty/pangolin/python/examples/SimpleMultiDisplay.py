# https://github.com/stevenlovegrove/Pangolin/blob/master/examples/SimpleMultiDisplay

import sys 
sys.path.append("../../")

import numpy as np

import OpenGL.GL as gl
import pypangolin as pangolin


def random_image(w, h):
    return (np.ones((h, w, 3), 'uint8') * 
        np.random.randint(256, size=3, dtype='uint8'))


def main():
    # Create OpenGL window in single line
    pangolin.CreateWindowAndBind('Main', 640, 480)

    # 3D Mouse handler requires depth testing to be enabled
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Issue specific OpenGl we might need
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


    # Define Camera Render Object (for view / scene browsing)
    proj = pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000)
    scam = pangolin.OpenGlRenderState(
        proj, pangolin.ModelViewLookAt(1, 0.5, -2, 0, 0, 0, pangolin.AxisY))
    scam2 = pangolin.OpenGlRenderState(
        proj, pangolin.ModelViewLookAt(0, 0, -2, 0, 0, 0, pangolin.AxisY))


    # Add named OpenGL viewport to window and provide 3D Handler
    dcam1 = pangolin.Display('cam1')
    dcam1.SetAspect(640 / 480.)
    dcam1.SetHandler(pangolin.Handler3D(scam))

    dcam2 = pangolin.Display('cam2')
    dcam2.SetAspect(640 / 480.)
    dcam2.SetHandler(pangolin.Handler3D(scam2))

    dcam3 = pangolin.Display('cam3')
    dcam3.SetAspect(640 / 480.)
    dcam3.SetHandler(pangolin.Handler3D(scam))

    dcam4 = pangolin.Display('cam4')
    dcam4.SetAspect(640 / 480.)
    dcam4.SetHandler(pangolin.Handler3D(scam2))

    dimg1 = pangolin.Display('img1')
    dimg1.SetAspect(640 / 480.)

    dimg2 = pangolin.Display('img2')
    dimg2.SetAspect(640 / 480.)


    # LayoutEqual is an EXPERIMENTAL feature - it requires that all sub-displays
    # share the same aspect ratio, placing them in a raster fasion in the
    # viewport so as to maximise display size.
    view = pangolin.Display('multi')
    view.SetBounds(0.0, 1.0, 0.0, 1.0)
    view.SetLayout(pangolin.LayoutEqual)
    view.AddDisplay(dcam1)
    view.AddDisplay(dimg1)
    view.AddDisplay(dcam2)

    view.AddDisplay(dimg2)
    view.AddDisplay(dcam3)
    view.AddDisplay(dcam4)


    w, h = 64, 48
    image_texture = pangolin.GlTexture(
        w, h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    # Default hooks for exiting (Esc) and fullscreen (tab)
    while not pangolin.ShouldQuit():
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)


        gl.glColor3f(1.0, 1.0, 1.0)

        dcam1.Activate(scam)
        pangolin.glDrawColouredCube()

        dcam2.Activate(scam2)
        pangolin.glDrawColouredCube()

        dcam3.Activate(scam)
        pangolin.glDrawColouredCube()

        dcam4.Activate(scam2)
        pangolin.glDrawColouredCube()


        dimg1.Activate()
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        image_texture.Upload(random_image(w, h), gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image_texture.RenderToViewport()

        dimg2.Activate()
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        # image_texture.Upload(random_image(w, h), gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image_texture.RenderToViewport()

        pangolin.FinishFrame()



if __name__ == '__main__':
    main()