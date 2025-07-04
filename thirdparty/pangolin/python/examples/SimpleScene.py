# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/SimpleScene

import sys 
sys.path.append("../../")

import OpenGL.GL as gl 
import pypangolin as pangolin


def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisY))

    tree = pangolin.Renderable()
    tree.Add(pangolin.Axis())

    # Create Interactive View in window
    handler = pangolin.SceneHandler(tree, scam)
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 /480.0)
    dcam.SetHandler(handler)

    def draw(view):
        view.Activate(scam)
        tree.Render()
    dcam.SetDrawFunction(draw)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # or
        # dcam.Activate(scam)
        # tree.Render()

        pangolin.FinishFrame()




if __name__ == '__main__':
    main()