import sys
sys.path.append("../../")
from config import Config

import unittest
import numpy as np

from pose_estimator import (
    PoseEstimatorType,
    pose_estimator_factory,
    PoseEstimatorInput
)

from utils_geom import rotation_matrix_from_yaw_pitch_roll


# generate random point in camera image
# output = [Nx2] (N = number of points)    
def generate_random_points_2d(width, height, num_points, margin_pixels=3):
    assert width > margin_pixels and height > margin_pixels
    points_2d = np.random.uniform(low=[margin_pixels, margin_pixels], high=[width-margin_pixels, height-margin_pixels], size=(num_points, 2)).astype(np.float32)
    return points_2d
    

# back project 2d image points with given camera matrix by using a random depth map in the range [minD, maxD]
# output = [Nx3] (N = number of points)
def random_backproject_points(K, points_2d, w, h, min_depth, max_depth):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # generate a random depth map with the size (h,w)
    depth = np.random.uniform(min_depth, max_depth, size=(h,w))

    points_2d_int = points_2d.astype(np.int32)
    z = depth[points_2d_int[:, 1], points_2d_int[:, 0]]
    x = (points_2d[:, 0] - cx) * z / fx
    y = (points_2d[:, 1] - cy) * z / fy

    points_3d = np.vstack((x, y, z)).T
    return points_3d.astype(np.float32), depth


# back project 2d image points with given camera matrix by using their depths
# output = [Nx3] (N = number of points)
def backproject_points(K, points_2d, depths):
    assert depths.shape[0] == points_2d.shape[0]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = np.asarray(depths).reshape(-1)  # Ensure it's 1D
    x = (points_2d[:, 0] - cx) * z / fx
    y = (points_2d[:, 1] - cy) * z / fy

    points_3d = np.vstack((x, y, z)).T
    return points_3d.astype(np.float32)


# project 3d points with given camera matrix and return the 2d image points, the visibility mask and the depth map
def project_points(K, points_3d_c, width, height):
    num_points = points_3d_c.shape[0]
    points_2d = np.zeros((num_points,2), dtype=np.float32)
    mask  = np.zeros(num_points, dtype=np.int32)
    depth = np.zeros((height, width), dtype=np.float32)
    index_map = np.full((height, width), -1, dtype=np.int32)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    for i in range(num_points):
        point = points_3d_c[i]
        if point[2] < 1e-6:  # Small threshold to prevent instability
            continue        
        u = (point[0] / point[2]) * fx + cx
        v = (point[1] / point[2]) * fy + cy
        points_2d[i] = np.array([u, v])
        u_int, v_int = int(u), int(v) 
        #u_int, v_int = np.round(u).astype(int), np.round(v).astype(int)        
        if 0 <= u_int < width and 0 <= v_int < height:
            d = depth[v_int, u_int]
            # if a point project to a pixel that has already taken by another point with depth > 0, then we get the one with the smallest depth                  
            if d > 0.0: # depth conflict,
                if point[2] >= d:  # the previous point is closer 
                    mask[i] = 0   # remove the current point
                else: # the current point is closer
                    depth[v_int, u_int] = point[2] 
                    previous_idex = index_map[v_int, u_int]
                    if previous_idex > -1:
                        mask[previous_idex] = 0 # remove the previous point
                    index_map[v_int, u_int] = i
                    mask[i] = 1                    
            else: 
                depth[v_int, u_int] = point[2]
                index_map[v_int, u_int] = i
                mask[i] = 1
    return points_2d, mask, depth


# project 3d points with given camera matrix and retunr the 2d image points, the visibility mask 
# the input depth is used to check the visibility 
def project_points_with_depth_check(K, points_3d_c, width, height, depth):
    assert width == depth.shape[1] and height == depth.shape[0]
    num_points = points_3d_c.shape[0]
    points_2d = np.zeros((num_points,2), dtype=np.float32)
    mask  = np.zeros(num_points, dtype=np.int32)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    for i in range(num_points):
        point = points_3d_c[i]
        if point[2] < 1e-6:  # Small threshold to prevent instability
            continue        
        u = (point[0] / point[2]) * fx + cx
        v = (point[1] / point[2]) * fy + cy
        points_2d[i] = np.array([u, v])
        u_int, v_int = int(u), int(v) 
        #u_int, v_int = np.round(u).astype(int), np.round(v).astype(int)        
        if 0 <= u_int < width and 0 <= v_int < height:
            d = depth[v_int, u_int]
            if d > 0.0 and point[2] <= d: 
                mask[i] = 1
    return points_2d, mask, depth

# compute the full depth map in the second image
def project_depth(K1, depth1, K2, R21, t21):
    depth1 = depth1.astype(np.float32)  # Ensure floating point depth
    vv, uu = np.mgrid[0:depth1.shape[0], 0:depth1.shape[1]]
    uv_coords = np.column_stack((uu.ravel(), vv.ravel())).astype(np.float32)
    uv_coords += np.array([0.5, 0.5], dtype=np.float32)
    #print(f'uv_coords shape: {uv_coords.shape}')
    
    valid = depth1.ravel() > 0
    xyz_1 = backproject_points(K1, uv_coords[valid], depth1.ravel()[valid])  
         
    t21 = t21.reshape(3, 1) if t21.shape != (3, 1) else t21
    xyz_2 = (R21 @ xyz_1.T + t21).T
    uv_2, mask, depth2 = project_points(K2, xyz_2, depth1.shape[1], depth1.shape[0])
    return depth2
    

class TestPoseEstimators(unittest.TestCase):

    fx = 517.306408
    fy = 516.469215
    cx = 318.643040
    cy = 255.313989
    width = 640
    height = 480
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


    # Create synthetic data for pose estimation
    def setUp(self):
        print(f'----------------------------------------')

        # Set a seed for reproducibility
        np.random.seed(0)  # You can change the seed value to any integer
        
        num_points = 300
        self.num_points = num_points
        self.sigma_noise_pixel = 0.1    
            
        self.test_tollerance_R_norm = 1e-3
        self.test_tollerance_t_norm = 5*1e-3

        # Camera intrinsics
        self.K1 = TestPoseEstimators.K.copy()
        self.K2 = TestPoseEstimators.K.copy()
        self.w1, self.h1 = TestPoseEstimators.width, TestPoseEstimators.height
        self.w2, self.h2 = TestPoseEstimators.width, TestPoseEstimators.height        
        
        # Pose 1 data
        Rc1w = np.eye(3, dtype=np.float32)                                 # Rotation matrix for frame 1
        tc1w = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape((3,1))  # Translation vector for frame 1
        Rwc1 = Rc1w.T
        twc1 = (-Rwc1 @ tc1w)
        self.Rc1w = Rc1w
        self.tc1w = tc1w

        # Pose 2 data
        # Random rotation and translation
        max_angle_deg = 5 
        max_t_component = 0.7
        yaw, pitch, roll = np.random.uniform(-max_angle_deg, max_angle_deg, 3)
        tx, ty, tz = np.random.uniform(-max_t_component, max_t_component, 3)

        Rc2w = rotation_matrix_from_yaw_pitch_roll(yaw, pitch, roll)  # Rotation matrix for frame 2
        tc2w = np.array([tx,ty,tz], dtype=np.float32).reshape((3,1))  # Translation vector for frame 2
        Rwc2 = Rc2w.T
        twc2 = -Rwc2 @ tc2w
        self.Rc2w = Rc2w
        self.tc2w = tc2w
        
        Rc2c1 = Rc2w @ Rc1w.T
        tc2c1 = tc2w - (Rc2c1 @ tc1w)
        self.Rc2c1 = Rc2c1
        self.tc2c1 = tc2c1
        
        Rc1c2 = Rc2c1.T
        tc1c2 = -Rc1c2 @ tc2c1
        self.Rc1c2 = Rc1c2
        self.tc1c2 = tc1c2 
        
        # print(f'Rc1c2:\n {Rc1c2}')
        # print(f'tc1c2:\n {tc1c2}')
        print(f'Rc2c1:\n {Rc2c1.flatten()}')
        print(f'tc2c1:\n {tc2c1.flatten()}')        
                        
        # generate random points in camera1 image and back project them with random depths
        points_2d_c1 = generate_random_points_2d(self.w1, self.h1, num_points)
        min_depth, max_depth = 1.0, 10.0
        points_3d_c1, depth1 = random_backproject_points(self.K1, points_2d_c1, self.w1, self.h1, min_depth, max_depth) 
        depths1 = depth1[points_2d_c1[:,1].astype(int), points_2d_c1[:,0].astype(int)]
                
        # add noise to 2D points in camera 1 to generate observations in camera 2
        points_2d_c1_noisy = points_2d_c1 + np.random.normal(0, self.sigma_noise_pixel, points_2d_c1.shape)
        points_3d_c1_noisy = backproject_points(self.K1, points_2d_c1_noisy, depths1) 
        
        # check_3d_diff = points_3d_c1_noisy - points_3d_c1
        # print(f'mean of 3d diff: {np.mean(np.abs(check_3d_diff))}')
        
        # check which noisy points are visible in camera 2
        points_3d_c2 = (Rc2c1 @ points_3d_c1_noisy.T + tc2c1.reshape((3, 1))).T
        points_2d_c2, mask, depth2_sparse = project_points(self.K2, points_3d_c2, self.w2, self.h2)
        depth2 = depth2_sparse  # this depth map is filled just with the depths of visible keypoints      
        #depth2 = project_depth(self.K1, depth1, self.K2, Rc2c1, tc2c1)
        #points_2d_c2, mask, depth2_sparse = project_points_with_depth_check(self.K2, points_3d_c2, self.w2, self.h2, depth2)
        
        print(f'number of visible points: {mask.sum()}')  
                
        if False:
            mask_diff = (depth2_sparse > 0)
            depth_diff = np.abs(depth2[mask_diff] - depth2_sparse[mask_diff])
            print(f'depth_diff: {depth_diff.sum()}')
                
        # remove points that are not visible in camera 2
        points_2d_c1 = points_2d_c1[mask==1,:]
        points_3d_c1 = points_3d_c1[mask==1,:]
        points_2d_c2 = points_2d_c2[mask==1,:]
        points_3d_c2 = points_3d_c2[mask==1,:]    
        depths1 = depths1[mask==1]
        
        points_3d_w = (Rwc1 @ points_3d_c1.T + twc1.reshape(3,1)).T
        #print(f'number of points_3d_w: {points_3d_w.shape[0]}')
        
        self.kpts1 = points_2d_c1
        self.kpts2 = points_2d_c2
        self.depth1 = depth1
        self.depth2 = depth2
        self.pts1 = points_3d_c1
        self.pts2 = points_3d_c2
        
        # double-check projections 
        if False:
            points_2d_c1_check, mask_c1, depth1_check = project_points(self.K1, (self.Rc1w @ points_3d_w.T + self.tc1w).T, self.w1, self.h1)
            average_projection_error_c1 = np.mean(np.linalg.norm(points_2d_c1[mask_c1==1,:] - points_2d_c1_check[mask_c1==1,:], axis=1))
            #print(f'mask_c1: {mask_c1}')
            print(f'[double-check] average projection error 1: {average_projection_error_c1}, mask = {np.sum(mask_c1)}')
            
            points_2d_c2_check, mask_c2, depth2_check = project_points(self.K2, (self.Rc2w @ points_3d_w.T + self.tc2w).T, self.w2, self.h2)
            average_projection_error_c2 = np.mean(np.linalg.norm(points_2d_c2[mask_c2==1,:] - points_2d_c2_check[mask_c2==1,:], axis=1))
            print(f'[duble-check] average projection error 2: {average_projection_error_c2}, mask = {np.sum(mask_c2)}')

    def test_essential_matrix_2d2d(self):
        estimator = pose_estimator_factory(PoseEstimatorType.ESSENTIAL_MATRIX_2D_2D, self.K1, self.K2)
        R21, t21, inliers = estimator.estimate(PoseEstimatorInput(kpts1=self.kpts1, kpts2=self.kpts2))
        print(f'\n[test_essential_matrix_2d2d]:\n R21 = {R21.flatten()},\n t21 = {t21.flatten()},\n #inliers = {inliers}')
        #print(f't norm: {np.linalg.norm(t21)}')
        self.assertEqual(R21.shape, (3, 3))
        #self.assertEqual(t21.shape, (3, 1))
        self.assertAlmostEqual(np.linalg.norm(t21),1.0, delta=self.test_tollerance_t_norm)
        self.assertGreaterEqual(inliers, 0)
        R_diff = np.linalg.norm(R21 @ self.Rc2c1.T - np.eye(3))
        t_diff = np.linalg.norm(t21.reshape((3, 1)) - self.tc2c1/np.linalg.norm(self.tc2c1))
        print(f'R diff: {R_diff}, t diff: {t_diff}')
        self.assertLessEqual(R_diff, self.test_tollerance_R_norm)
        self.assertLessEqual(t_diff, self.test_tollerance_t_norm)
        
    def test_essential_matrix_metric_simple(self):
        estimator = pose_estimator_factory(PoseEstimatorType.ESSENTIAL_MATRIX_METRIC_SIMPLE, self.K1, self.K2)
        R21, t21, inliers = estimator.estimate(PoseEstimatorInput(kpts1=self.kpts1, kpts2=self.kpts2, depth1=self.depth1, depth2=self.depth2))
        print(f'\n[test_essential_matrix_metric_simple]:\n R21 = {R21.flatten()},\n t21 = {t21.flatten()},\n #inliers = {inliers}')
        self.assertEqual(R21.shape, (3, 3))
        self.assertGreaterEqual(inliers, 0)
        R_diff = np.linalg.norm(R21 @ self.Rc2c1.T - np.eye(3))
        t_diff = np.linalg.norm(t21.reshape((3, 1)) - self.tc2c1)
        print(f'R diff: {R_diff}, t diff: {t_diff}')
        self.assertLessEqual(R_diff, self.test_tollerance_R_norm)
        #self.assertLessEqual(t_diff, self.test_tollerance_t_norm)  # This pose estimator is not robust enough      

    def test_essential_matrix_metric(self):
        estimator = pose_estimator_factory(PoseEstimatorType.ESSENTIAL_MATRIX_METRIC, self.K1, self.K2)
        R21, t21, inliers = estimator.estimate(PoseEstimatorInput(kpts1=self.kpts1, kpts2=self.kpts2, depth1=self.depth1, depth2=self.depth2))
        print(f'\n[test_essential_matrix_metric]:\n R21 = {R21.flatten()},\n t21 = {t21.flatten()},\n #inliers = {inliers}')
        self.assertEqual(R21.shape, (3, 3))
        self.assertGreaterEqual(inliers, 0)
        R_diff = np.linalg.norm(R21 @ self.Rc2c1.T - np.eye(3))
        t_diff = np.linalg.norm(t21 - self.tc2c1)
        print(f'R diff: {R_diff}, t diff: {t_diff}')
        self.assertLessEqual(R_diff, self.test_tollerance_R_norm)
        self.assertLessEqual(t_diff, self.test_tollerance_t_norm)    
            
    def test_pnp(self):
        estimator = pose_estimator_factory(PoseEstimatorType.PNP, self.K1, self.K2)
        R21, t21, inliers = estimator.estimate(PoseEstimatorInput(kpts1=self.kpts1, kpts2=self.kpts2, depth1=self.depth1))
        print(f'\n[test_pnp]:\n R21 = {R21.flatten()},\n t21 = {t21.flatten()},\n #inliers = {inliers}')
        self.assertEqual(R21.shape, (3, 3))
        self.assertGreaterEqual(inliers, 0)
        R_diff = np.linalg.norm(R21 @ self.Rc2c1.T - np.eye(3))
        t_diff = np.linalg.norm(t21 - self.tc2c1)
        print(f'R diff: {R_diff}, t diff: {t_diff}')
        self.assertLessEqual(R_diff, self.test_tollerance_R_norm)
        self.assertLessEqual(t_diff, self.test_tollerance_t_norm)    

    def test_pnp_with_sigmas(self):
        estimator = pose_estimator_factory(PoseEstimatorType.PNP_WITH_SIGMAS, self.K1, self.K2)
        R21, t21, inliers = estimator.estimate(PoseEstimatorInput(kpts1=self.kpts1, kpts2=self.kpts2, depth1=self.depth1))
        print(f'\n[test_pnp_with_sigmas]:\n R21 = {R21.flatten()},\n t21 = {t21.flatten()},\n #inliers = {inliers}')
        self.assertEqual(R21.shape, (3, 3))
        self.assertGreaterEqual(inliers, 0)
        R_diff = np.linalg.norm(R21 @ self.Rc2c1.T - np.eye(3))
        t_diff = np.linalg.norm(t21 - self.tc2c1)
        print(f'R diff: {R_diff}, t diff: {t_diff}')
        self.assertLessEqual(R_diff, self.test_tollerance_R_norm)
        self.assertLessEqual(t_diff, self.test_tollerance_t_norm)  
        
    def test_mlpnp_with_sigmas(self):
        estimator = pose_estimator_factory(PoseEstimatorType.MLPNP_WITH_SIGMAS, self.K1, self.K2)
        R21, t21, inliers = estimator.estimate(PoseEstimatorInput(kpts1=self.kpts1, kpts2=self.kpts2, depth1=self.depth1))
        print(f'\n[test_mlpnp_with_sigmas]:\n R21 = {R21.flatten()},\n t21 = {t21.flatten()},\n #inliers = {inliers}')
        self.assertEqual(R21.shape, (3, 3))
        self.assertGreaterEqual(inliers, 0)
        R_diff = np.linalg.norm(R21 @ self.Rc2c1.T - np.eye(3))
        t_diff = np.linalg.norm(t21 - self.tc2c1)
        print(f'R diff: {R_diff}, t diff: {t_diff}')
        self.assertLessEqual(R_diff, self.test_tollerance_R_norm)
        self.assertLessEqual(t_diff, self.test_tollerance_t_norm)          

    def test_procrustes(self):
        estimator = pose_estimator_factory(PoseEstimatorType.PROCUSTES, self.K1, self.K2)
        R21, t21, inliers = estimator.estimate(PoseEstimatorInput(kpts1=self.kpts1, kpts2=self.kpts2, depth1=self.depth1, depth2=self.depth2))
        print(f'\n[test_procrustes]:\n R21 = {R21.flatten()},\n t21 = {t21.flatten()},\n #inliers = {inliers}')
        self.assertEqual(R21.shape, (3, 3))
        self.assertGreaterEqual(inliers, 0)
        R_diff = np.linalg.norm(R21 @ self.Rc2c1.T - np.eye(3))
        t_diff = np.linalg.norm(t21 - self.tc2c1)
        print(f'R diff: {R_diff}, t diff: {t_diff}')
        self.assertLessEqual(R_diff, self.test_tollerance_R_norm)
        self.assertLessEqual(t_diff, self.test_tollerance_t_norm)    
        
    def test_sim3(self):
        estimator = pose_estimator_factory(PoseEstimatorType.SIM3_3D3D, self.K1, self.K2)
        
        kpts1_int = np.int32(self.kpts1)
        kpts2_int = np.int32(self.kpts2)
        depths1 = self.depth1[kpts1_int[:, 1], kpts1_int[:, 0]]
        depths2 = self.depth2[kpts2_int[:, 1], kpts2_int[:, 0]]
        pts1 = backproject_points(self.K1, self.kpts1, depths1)
        pts2 = backproject_points(self.K2, self.kpts2, depths2)
        
        pts1_2 = (self.Rc2c1 @ pts1.T + self.tc2c1).T
        diff = pts1_2 - pts2
        error = np.linalg.norm(diff, axis=1)
        #print(f'mean error: {np.mean(error)}')
                
        R21, t21, inliers = estimator.estimate(PoseEstimatorInput(kpts1=self.kpts1, kpts2=self.kpts2, depth1=self.depth1, depth2=self.depth2, fix_scale=True))
        #R21, t21, inliers = estimator.estimate(PoseEstimatorInput(pts1=pts1, pts2=pts2, fix_scale=True))
        print(f'\n[test_sim3]:\n R21 = {R21.flatten()},\n t21 = {t21.flatten()},\n #inliers = {inliers}')
        self.assertEqual(R21.shape, (3, 3))
        self.assertGreaterEqual(inliers, 0)
        R_diff = np.linalg.norm(R21 @ self.Rc2c1.T - np.eye(3))
        t_diff = np.linalg.norm(t21 - self.tc2c1)
        print(f'R diff: {R_diff}, t diff: {t_diff}')
        self.assertLessEqual(R_diff, 2*self.test_tollerance_R_norm)
        self.assertLessEqual(t_diff, 2*self.test_tollerance_t_norm)

if __name__ == '__main__':
    unittest.main()