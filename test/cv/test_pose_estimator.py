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
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z = depths
    x = (points_2d[:, 0] - cx) * z / fx
    y = (points_2d[:, 1] - cy) * z / fy

    points_3d = np.vstack((x, y, z)).T
    return points_3d.astype(np.float32)


# project 3d points with given camera matrix and retunr the 2d image points, the visibility mask and the depth map
def project_points(K, points_3d_c, width, height):
    num_points = points_3d_c.shape[0]
    points_2d = np.zeros((num_points,2), dtype=np.float32)
    mask  = np.zeros(num_points, dtype=np.int32)
    depth = np.zeros((height, width), dtype=np.float32)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    for i in range(num_points):
        point = points_3d_c[i,:]
        if np.abs(point[2]) < 1e-6:  # Small threshold to prevent instability
            continue        
        u = point[0] / point[2] * fx + cx
        v = point[1] / point[2] * fy + cy
        points_2d[i, :] = np.array([u, v])
        if u >= 0 and u < width and v >= 0 and v < height:
            mask[i] = 1
            # if a point project to a pixel already with depth > 0, then the current pixel is excluded (to avoid occlusions) 
            u_int, v_int = int(u), int(v)   
            #print(f'u_int: {u_int}, v_int: {v_int}')       
            if depth[v_int, u_int] > 0: 
                mask[i] = 0
            else: 
                depth[v_int, u_int] = point[2]
    return points_2d, mask, depth


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

        # Set a seed for reproducibility
        np.random.seed(42)  # You can change the seed value to any integer
        
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
        
        # pose data
        Rc1w = np.eye(3, dtype=np.float32)                                 # Rotation matrix for frame 1
        tc1w = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape((3,1))  # Translation vector for frame 1
        Rwc1 = Rc1w.T
        twc1 = (-Rwc1 @ tc1w)
        self.Rc1w = Rc1w
        self.tc1w = tc1w

        Rc2w = rotation_matrix_from_yaw_pitch_roll(-1.4,-5.6,3.9)        # Rotation matrix for frame 2
        tc2w = np.array([0.3,0.1,0.5], dtype=np.float32).reshape((3,1))  # Translation vector for frame 2
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
        points_2d_c2, mask, depth2 = project_points(self.K2, points_3d_c2, self.w2, self.h2)
        print(f'number of visible points: {mask.sum()}')
                
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
        self.assertLessEqual(t_diff, self.test_tollerance_t_norm)        

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

if __name__ == '__main__':
    unittest.main()