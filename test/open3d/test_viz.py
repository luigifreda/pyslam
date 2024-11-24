import open3d as o3d
import numpy as np
import time

# Function to generate or update point cloud data
def generate_point_cloud_data(range=1.0):
    # Create random points (replace with your actual point cloud data)
    num_points = 1000
    points = np.random.rand(num_points, 3) * range
    colors = np.random.rand(num_points, 3)  # Random colors for the points
    return points, colors

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

def main():
    # Initialize the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Adjust the FOV using the view control
    view_control = vis.get_view_control()
    view_control.set_constant_z_near(0.1)
    view_control.set_constant_z_far(1000)
    #view_control.set_zoom(0.1)
    view_control.set_lookat([0, 0, 0])
    view_control.scale(1000)
        

    # Create the initial point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # Initialize with the first frame's data
    if False:
        points, colors = generate_point_cloud_data(0)
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Add the point cloud to the visualizer
    vis.add_geometry(point_cloud)

    # Add a coordinate frame to the visualizer
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_frame)

    # Add a grid to the visualizer
    grid = create_grid(size=10, divisions=10)
    vis.add_geometry(grid)

    # Continuous update loop
    try:
        frame_id = 0
        while True:
            # Update the existing point cloud's data
            points, colors = generate_point_cloud_data()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            frame_id += 1

            # Update the visualizer
            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()

            # Wait a bit to simulate real-time updates
            time.sleep(0.04)

    except KeyboardInterrupt:
        # Close the visualizer gracefully when interrupted
        vis.destroy_window()
        print("Visualization stopped.")

if __name__ == "__main__":
    main()