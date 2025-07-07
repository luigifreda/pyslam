import sys
sys.path.append("../../")
from pyslam.config import Config

import numpy as np

from pyslam.utilities.utils_depth import point_cloud_to_depth


def depth_to_point_cloud(depth_img, K):
    """
    Generate a point cloud (Nx3) from a depth image and intrinsic matrix K.
    
    Parameters:
      depth_img: (h, w) numpy array of depth values.
      K: (3, 3) intrinsic calibration matrix.
    
    Returns:
      points: (N, 3) array of 3D points in the camera coordinate system.
    """
    h, w = depth_img.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create a grid of (u,v) coordinates
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u_coords = u_coords.astype(np.float32)
    v_coords = v_coords.astype(np.float32)
    
    # Back-project to 3D (points with depth=0 are not valid)
    Z = depth_img
    X = (u_coords - cx) * Z / fx
    Y = (v_coords - cy) * Z / fy

    # Only keep points where depth > 0
    valid = Z > 0
    points = np.stack((X[valid], Y[valid], Z[valid]), axis=1)
    return points


# Test function
def test_point_cloud_to_depth():
    # Define camera intrinsics and image shape
    w, h = 640, 480
    K = np.array([[525.0,   0.0, 319.5],
                  [  0.0, 525.0, 239.5],
                  [  0.0,   0.0,   1.0]])
    
    if False: 
        # Create a synthetic depth image
        # Let's say depth is 1.0 everywhere, with a circular region at 0.5 depth.
        depth_img_original = np.full((h, w), 1.0, dtype=np.float32)
        center = (w // 2, h // 2)
        radius = 50
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        mask = (u_coords - center[0])**2 + (v_coords - center[1])**2 <= radius**2
        depth_img_original[mask] = 0.5
    else: 
        # remove the randomness
        np.random.seed(0)
        # generate a random depth image with values in the range min_depth, max_depth
        min_depth = 0.3
        max_depth = 10.0
        depth_img_original = np.random.uniform(min_depth, max_depth, (h, w))
    

    # Convert depth image to point cloud
    points = depth_to_point_cloud(depth_img_original, K)
    
    # Reconstruct the depth image from the point cloud
    depth_img_reconstructed = point_cloud_to_depth(points, K, w, h)
    
    # Compute difference
    diff = np.abs(depth_img_original - depth_img_reconstructed)
    
    # For test purposes, we can compute a norm or a max difference
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print("Max difference:", max_diff)
    print("Mean difference:", mean_diff)
    
    # A tolerance threshold: adjust as needed (here we allow a small tolerance due to rounding)
    tol = 1e-3
    assert max_diff < tol, f"Reconstructed depth image differs too much (max diff = {max_diff})"

if __name__ == "__main__":
    test_point_cloud_to_depth()
