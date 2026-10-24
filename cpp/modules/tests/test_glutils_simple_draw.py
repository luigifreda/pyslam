# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/HelloPangolin

import sys

import pyslam.config as config

import OpenGL.GL as gl
import pypangolin as pangolin
import glutils

import numpy as np


def drawPlane(ndivs=100, ndivsize=1):
    # Plane parallel to x-z at origin with normal -y
    minx = -ndivs * ndivsize
    minz = -ndivs * ndivsize
    maxx = ndivs * ndivsize
    maxz = ndivs * ndivsize
    gl.glLineWidth(1)
    gl.glColor3f(0.7, 0.7, 1.0)
    gl.glBegin(gl.GL_LINES)
    for n in range(2 * ndivs):
        gl.glVertex3f(minx + ndivsize * n, 0, minz)
        gl.glVertex3f(minx + ndivsize * n, 0, maxz)
        gl.glVertex3f(minx, 0, minz + ndivsize * n)
        gl.glVertex3f(maxx, 0, minz + ndivsize * n)
    gl.glEnd()


def colorify_points(points):
    colors = np.zeros((len(points), 3))
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    denom = maxs - mins
    colors[:] = np.divide(points - mins, denom, out=colors, where=denom != 0)
    return colors


def main():
    pangolin.CreateWindowAndBind("Main", 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY),
    )
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
    dcam.SetHandler(handler)

    pc = glutils.GlPointCloudF()
    pc_direct = glutils.GlPointCloudDirectF()
    mesh = glutils.GlMeshF()
    mesh_direct = glutils.GlMeshDirectF()

    trajectory = [[0, -6, 6]]
    for i in range(300):
        trajectory.append(trajectory[-1] + np.random.random(3) - 0.5)
    trajectory = np.array(trajectory)
    print(trajectory.shape)

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

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        # drawPlane()
        glutils.DrawPlane(num_divs=10, div_size=1, scale=1.0)

        # Render OpenGL Cube
        pangolin.glDrawColouredCube()

        # Draw Point Cloud (static size, single color)
        cloud_center = np.array([-5.0, 0.0, 0.0], dtype=np.float32)
        size = 1.0
        points = (np.random.random((10000, 3)) - 0.5) * size + cloud_center
        gl.glPointSize(1)
        gl.glColor3f(1.0, 0.0, 0.0)
        glutils.DrawPoints(points)

        # Draw Point Cloud (static size, multiple colors)
        cloud_center = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        size = 2.0
        points = (np.random.random((10000, 3)) - 0.5) * size + cloud_center
        colors = colorify_points(points)
        gl.glPointSize(1)
        glutils.DrawPoints(points, colors)

        # Draw Point Cloud with GlPointCloudF (dynamic size)
        if True:
            cloud_center = np.array([0.0, 0.0, 5.0], dtype=np.float32)
            size = 3.0
            n_points = 2000 + (frame_idx % 8000)
            if frame_idx % 2 == 0:
                pc_points = np.ascontiguousarray(
                    (np.random.random((n_points, 3)) - 0.5) * size + cloud_center,
                    dtype=np.float32,
                )
                pc_colors = colorify_points(pc_points)
                pc_colors = np.ascontiguousarray(pc_colors, dtype=np.float32)
                pc.set(pc_points, pc_colors)
            gl.glPointSize(2)
            pc.draw()

        # Draw Point Cloud with GlPointCloudDirectF (dynamic size)
        if True:
            gl.glPointSize(2)
            cloud_center = np.array([0.0, 0.0, -5.0], dtype=np.float32)
            size = 3.0
            n_points = 2000 + (frame_idx % 8000)
            if frame_idx % 2 == 0:
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
            mesh_center = np.array([-10.0, 0.0, 0.0], dtype=np.float32)
            size = 0.5
            phase = frame_idx * 0.05
            if frame_idx % 2 == 0:
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
            mesh_center = np.array([10.0, 0.0, 0.0], dtype=np.float32)
            size = 0.5
            phase = frame_idx * 0.05 + 1.0
            if frame_idx % 2 == 0:
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

        # Draw lines
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 0.0)
        # glutils.DrawLine(trajectory)   # consecutive
        glutils.DrawTrajectory(trajectory)
        gl.glColor3f(0.0, 1.0, 0.0)
        glutils.DrawLines2(
            trajectory, trajectory + np.random.randn(len(trajectory), 3), point_size=5
        )  # separate

        # Draw camera
        pose = np.identity(4)
        pose[:3, 3] = np.random.randn(3)
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)
        glutils.DrawCamera(pose, 0.5, 0.75, 0.8)

        # Draw boxes
        poses = [np.identity(4) for i in range(10)]
        for pose in poses:
            pose[:3, 3] = np.random.randn(3) + np.array([5, -3, 0])
        sizes = np.random.random((len(poses), 3))
        gl.glLineWidth(1)
        gl.glColor3f(1.0, 0.0, 1.0)
        glutils.DrawBoxes(poses, sizes)

        pangolin.FinishFrame()
        frame_idx += 1


if __name__ == "__main__":
    main()
