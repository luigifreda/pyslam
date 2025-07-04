# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/SimpleDisplay

import sys 
sys.path.append("../../")

import OpenGL.GL as gl
import pypangolin as pangolin


class SetVarFunctor(object):
    def __init__(self, var=None, value=None):
        super().__init__()
        self.var = var
        self.value = value

    def __call__(self):
        self.var.SetVal(self.value)

def reset():
    #float_slider.SetVal(0.5)
    print('You typed ctrl-r or pushed reset')



def main():
    pangolin.ParseVarsFile('app.cfg')

    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin.ModelViewLookAt(0, 0.5, -3, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler3d = pangolin.Handler3D(scam)

    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 180/640., 1.0, -640.0/480.0)
    # dcam.SetBounds(pangolin.Attach(0.0),     pangolin.Attach(1.0), 
                     # pangolin.Attach.Pix(180), pangolin.Attach(1.0), -640.0/480.0)

    dcam.SetHandler(pangolin.Handler3D(scam))

    panel = pangolin.CreatePanel('ui')
    panel.SetBounds(0.0, 1.0, 0.0, 180/640.)


    button = pangolin.VarBool('ui.Button', value=False, toggle=False)
    checkbox = pangolin.VarBool('ui.Checkbox', value=False, toggle=True)
    float_slider = pangolin.VarFloat('ui.Float', value=3, min=0, max=5)
    float_log_slider = pangolin.VarFloat('ui.Log_scale var', value=3, min=1, max=1e4, logscale=True)
    int_slider = pangolin.VarInt('ui.Int', value=2, min=0, max=5)
    int_slave_slider = pangolin.VarInt('ui.Int_slave', value=2, toggle=False)

    save_window = pangolin.VarBool('ui.Save_Window', value=False, toggle=False)
    save_cube   = pangolin.VarBool('ui.Save_Cube',   value=False, toggle=False)
    record_cube = pangolin.VarBool('ui.Record_Cube', value=False, toggle=False)

    def reset():
        #float_slider.SetVal(0.5)
        print('You typed ctrl-r or pushed reset')

    # Reset = SetVarFunctor(float_slider, 0.5)  
    # reset = pangolin.VarFunc('ui.Reset', reset)                                
    # pangolin.RegisterKeyPressCallback(int(pangolin.PANGO_CTRL) + ord('r'), reset)      # segfault
    # pangolin.RegisterKeyPressCallback(int(pangolin.PANGO_CTRL) + ord('b'), pangolin.SetVarFunctorFloat('ui.Float', 4.5))      # segfault
    # pangolin.RegisterKeyPressCallback(int(pangolin.PANGO_CTRL) + ord('b'), SetVarFunctor(float_slider, 4.5))      # segfault

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if pangolin.Pushed(button):
            print('You Pushed a button!')

        if checkbox.Get():
            int_slider.SetVal(int(float_slider))
        int_slave_slider.SetVal(int_slider)

        if pangolin.Pushed(save_window):
            pangolin.SaveWindowOnRender("window")

        if pangolin.Pushed(save_cube):
            pangolin.SaveWindowOnRender("cube")

        if pangolin.Pushed(record_cube):
            pangolin.DisplayBase().RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi")
        

        dcam.Activate(scam)
        gl.glColor3f(1.0, 1.0, 1.0)
        pangolin.glDrawColouredCube()
        pangolin.FinishFrame()
        

if __name__ == '__main__':
    main()