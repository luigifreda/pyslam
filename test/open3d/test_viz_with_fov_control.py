import open3d as o3d
import numpy as np
import time



# Function to create a grid
def create_grid(size=10, divisions=10):
    lines = []
    points = []
    step = size / divisions

    # Generate lines in X-Z plane
    for i in range(divisions + 1):
        coord = -size / 2 + i * step
        # Lines parallel to X-axis
        lines.append([len(points), len(points) + 1])
        points.append([coord, 0, -size / 2])
        points.append([coord, 0, size / 2])

        # Lines parallel to Z-axis
        lines.append([len(points), len(points) + 1])
        points.append([-size / 2, 0, coord])
        points.append([size / 2, 0, coord])

    # Create LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


if __name__ == "__main__":

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Controlled FoV and Zoom", width=800, height=600)

    # Add a grid to the visualizer
    grid = create_grid(size=1000, divisions=1000)
    vis.add_geometry(grid)    
    
    # Add a coordinate frame to the visualizer
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_frame)
        
    # Create initial trajectory
    trajectory_points = np.array([[0.0, 0.0, 0.0]])
    lines = []  # No connections initially    
    colors = []  # Line color (red)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(trajectory_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Add LineSet to visualizer
    vis.add_geometry(line_set)

    # Adjust the FOV using the view control
    view_control = vis.get_view_control()
    view_control.set_constant_z_near(0.1)
    view_control.set_constant_z_far(1000)
    #view_control.set_zoom(0.1)
    # view_control.set_lookat([0, 0, 0])
    # view_control.scale(1000)

    N = 1000

    delta = 0.01

    # Simulate trajectory updates
    for i in range(1, N):
        # Add new point
        new_point = np.array([i * delta, np.sin(i * delta), np.cos(i * delta)])
        trajectory_points = np.vstack((trajectory_points, new_point))
                
        # Update LineSet geometry
        lines = [[j, j + 1] for j in range(len(trajectory_points) - 1)]
        colors = [[1.0, 0.0, 0.0] for _ in lines]

        # Update LineSet geometry
        line_set.points = o3d.utility.Vector3dVector(trajectory_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)  # Keep color red
        
        # Update visualizer
        vis.update_geometry(line_set)
        
        # Adjust FoV and zoom dynamically
        #bbox = line_set.get_axis_aligned_bounding_box()
        #view_control.set_lookat(bbox.get_center())        
        view_control.set_lookat(new_point)
            
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(0.03)  # Simulate a delay

    # Keep the window open
    vis.run()
    vis.destroy_window()