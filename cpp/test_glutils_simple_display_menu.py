# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/SimpleDisplay

import sys

import pyslam.config as config

import OpenGL.GL as gl
import pypangolin as pangolin

import glutils

import numpy as np


class SetVarFunctor(object):
    def __init__(self, var=None, value=None):
        super().__init__()
        self.var = var
        self.value = value

    def __call__(self):
        self.var.SetVal(self.value)


def reset():
    # float_slider.SetVal(0.5)
    print("You typed ctrl-r or pushed reset")


def colorify_points(points):
    colors = np.zeros((len(points), 3))
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    denom = maxs - mins
    colors[:] = np.divide(points - mins, denom, out=colors, where=denom != 0)
    return colors


def main():
    pangolin.ParseVarsFile("app.cfg")

    window_width = 800
    window_height = 600
    viewpoint_f = 1000
    ui_width = 180

    pangolin.CreateWindowAndBind("Main", window_width, window_height)
    gl.glEnable(gl.GL_DEPTH_TEST)

    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(
            window_width,
            window_height,
            viewpoint_f,
            viewpoint_f,
            window_width // 2,
            window_height // 2,
            0.1,
            1000,
        ),
        pangolin.ModelViewLookAt(0, 0.5, -3, 0, 0, 0, pangolin.AxisDirection.AxisY),
    )
    handler3d = pangolin.Handler3D(scam)

    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, ui_width / window_width, 1.0, -window_width / window_height)
    # dcam.SetBounds(pangolin.Attach(0.0),     pangolin.Attach(1.0),
    # pangolin.Attach.Pix(180), pangolin.Attach(1.0), -640.0/480.0)

    dcam.SetHandler(pangolin.Handler3D(scam))

    panel = pangolin.CreatePanel("ui")
    panel.SetBounds(0.0, 1.0, 0.0, ui_width / window_width)

    button = pangolin.VarBool("ui.Button", value=False, toggle=False)
    checkbox = pangolin.VarBool("ui.Checkbox", value=False, toggle=True)
    float_slider = pangolin.VarFloat("ui.Float", value=3, min=0, max=5)
    float_log_slider = pangolin.VarFloat("ui.Log_scale var", value=3, min=1, max=1e4, logscale=True)
    int_slider = pangolin.VarInt("ui.Int", value=2, min=0, max=5)
    int_slave_slider = pangolin.VarInt("ui.Int_slave", value=2, toggle=False)

    save_window = pangolin.VarBool("ui.Save_Window", value=False, toggle=False)
    save_cube = pangolin.VarBool("ui.Save_Cube", value=False, toggle=False)
    record_cube = pangolin.VarBool("ui.Record_Cube", value=False, toggle=False)

    def reset():
        # float_slider.SetVal(0.5)
        print("You typed ctrl-r or pushed reset")

    # Reset = SetVarFunctor(float_slider, 0.5)
    # reset = pangolin.VarFunc('ui.Reset', reset)
    # pangolin.RegisterKeyPressCallback(int(pangolin.PANGO_CTRL) + ord('r'), reset)      # segfault
    # pangolin.RegisterKeyPressCallback(int(pangolin.PANGO_CTRL) + ord('b'), pangolin.SetVarFunctorFloat('ui.Float', 4.5))      # segfault
    # pangolin.RegisterKeyPressCallback(int(pangolin.PANGO_CTRL) + ord('b'), SetVarFunctor(float_slider, 4.5))      # segfault

    pc_direct = glutils.GlPointCloudDirectF()
    pc = glutils.GlPointCloudF()
    mesh = glutils.GlMeshF()
    mesh_direct = glutils.GlMeshDirectF()

    # Precompute grid for mesh demo
    grid_size = 40
    grid_extent = 4.0
    grid_lin = np.linspace(-grid_extent, grid_extent, grid_size, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(grid_lin, grid_lin, indexing="xy")
    grid_x = np.ascontiguousarray(grid_x.reshape(-1), dtype=np.float32)
    grid_z = np.ascontiguousarray(grid_z.reshape(-1), dtype=np.float32)
    vertex_count = grid_x.shape[0]

    # Build triangle indices for the grid
    tri_list = []
    for r in range(grid_size - 1):
        row_start = r * grid_size
        next_row = (r + 1) * grid_size
        for c in range(grid_size - 1):
            i0 = row_start + c
            i1 = row_start + c + 1
            i2 = next_row + c
            i3 = next_row + c + 1
            tri_list.append([i0, i2, i1])
            tri_list.append([i1, i2, i3])
    tri_indices = np.ascontiguousarray(np.array(tri_list, dtype=np.uint32))
    tri_count = tri_indices.shape[0]

    frame_idx = 0

    update_interval1 = 1
    update_interval2 = 10

    while not pangolin.ShouldQuit():
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if pangolin.Pushed(button):
            print("You Pushed a button!")

        if checkbox.Get():
            int_slider.SetVal(int(float_slider))
        int_slave_slider.SetVal(int_slider)

        if pangolin.Pushed(save_window):
            pangolin.SaveWindowOnRender("window")

        if pangolin.Pushed(save_cube):
            pangolin.SaveWindowOnRender("cube")

        if pangolin.Pushed(record_cube):
            pangolin.DisplayBase().RecordOnRender(
                "ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi"
            )

        dcam.Activate(scam)

        # Draw Point Cloud with GlPointCloudF (dynamic size)
        if True:
            cloud_center = np.array([0.0, 0.0, 5.0], dtype=np.float32)
            size = 3.0
            gl.glPointSize(2)
            if frame_idx % update_interval1 == 0:
                n_points = 2000 + (frame_idx % 8000)
                pc_points = np.ascontiguousarray(
                    (np.random.random((n_points, 3)) - 0.5) * size + cloud_center,
                    dtype=np.float32,
                )
                pc_colors = colorify_points(pc_points)
                pc_colors = np.ascontiguousarray(pc_colors, dtype=np.float32)
                pc.set(pc_points, pc_colors)
            pc.draw()

        # Draw Point Cloud with GlPointCloudDirectF (dynamic size)
        if True:
            cloud_center = np.array([0.0, 0.0, -5.0], dtype=np.float32)
            size = 1.0
            gl.glPointSize(2)
            if frame_idx % update_interval2 == 0:
                n_points = 2000 + (frame_idx % 8000)
                pc_points = np.ascontiguousarray(
                    (np.random.random((n_points, 3)) - 0.5) * size + cloud_center,
                    dtype=np.float32,
                )
                pc_colors = colorify_points(pc_points)
                pc_colors = np.ascontiguousarray(pc_colors, dtype=np.float32)
                pc_direct.update(pc_points, pc_colors)
            pc_direct.draw()

        # Draw Mesh with GlMeshF (dynamic zmap on fixed grid)
        if True:
            mesh_center = np.array([-5.0, 0.0, 0.0], dtype=np.float32)
            size = 0.5
            phase = frame_idx * 0.05
            if frame_idx % update_interval1 == 0:
                y = size * np.sin(grid_x + phase) * np.cos(grid_z - phase)
                vertices = (
                    np.stack([grid_x, y, grid_z], axis=1).astype(np.float32, copy=False)
                    + mesh_center
                )
                colors = colorify_points(vertices)
                colors = np.ascontiguousarray(colors, dtype=np.float32)
                vertices = np.ascontiguousarray(vertices, dtype=np.float32)

                mesh.set_vertices(vertices)
                mesh.set_triangles(tri_indices)
                mesh.set_colors(colors)
            gl.glColor3f(1.0, 1.0, 1.0)
            mesh.draw()

        # Draw Mesh with GlMeshDirectF (dynamic zmap on fixed grid)
        if True:
            mesh_center = np.array([5.0, 0.0, 0.0], dtype=np.float32)
            size = 0.5
            phase = frame_idx * 0.05 + 1.0
            if frame_idx % update_interval2 == 0:
                y = size * np.sin(grid_x + phase) * np.cos(grid_z - phase)
                vertices = (
                    np.stack([grid_x, y, grid_z], axis=1).astype(np.float32, copy=False)
                    + mesh_center
                )
                colors = colorify_points(vertices)
                colors = np.ascontiguousarray(colors, dtype=np.float32)
                vertices = np.ascontiguousarray(vertices, dtype=np.float32)
                mesh_direct.update(vertices, tri_indices, colors)
            mesh_direct.draw()

        gl.glColor3f(1.0, 1.0, 1.0)
        pangolin.glDrawColouredCube()

        glutils.DrawPlane(num_divs=10, div_size=1, scale=1.0)

        pangolin.FinishFrame()

        frame_idx += 1


if __name__ == "__main__":
    main()
