import sys
import numpy as np
import time

import rerun as rr
import cv2
from matplotlib import pyplot as plt

sys.path.append("./lib/")
import sim3solver


fx = 517.306408
fy = 516.469215
cx = 318.643040
cy = 255.313989
width = 640
height = 480
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def horn_absolute_orientation(P, Q):
    """
    Implementation of Horn's method for absolute orientation using unit quaternions.

    Args:
        P (np.ndarray): A (n, 3) array of points in the source frame.
        Q (np.ndarray): A (n, 3) array of corresponding points in the target frame.

    Returns:
        R (np.ndarray): The optimal rotation matrix (3x3).
        t (np.ndarray): The optimal translation vector (3x1).
    """
    assert P.shape == Q.shape, "Point sets must have the same shape."

    # Step 1: Compute the centroids of P and Q
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Step 2: Center the points by subtracting the centroids
    P_prime = P - centroid_P
    Q_prime = Q - centroid_Q

    # Step 3: Compute the cross-covariance matrix H
    H = P_prime.T @ Q_prime

    # Step 4: Compute the SVD of H
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Compute the optimal rotation matrix R
    R = Vt.T @ U.T

    # Handle the case where the determinant of R is negative
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 6: Compute the translation vector t
    t = centroid_Q - R @ centroid_P

    return R, t


def rotation_matrix_from_yaw_pitch_roll(yaw_degs, pitch_degs, roll_degs):
    # Convert angles from degrees to radians
    yaw = np.radians(yaw_degs)
    pitch = np.radians(pitch_degs)
    roll = np.radians(roll_degs)
    # Rotation matrix for Roll (X-axis rotation)
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    # Rotation matrix for Pitch (Y-axis rotation)
    Ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    # Rotation matrix for Yaw (Z-axis rotation)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    # Final rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


# generate random point in camera image
# output = [Nx2] (N = number of points)
def generate_random_points_2d(width, height, num_points):
    points_2d = np.random.uniform(low=[0, 0], high=[width, height], size=(num_points, 2)).astype(
        np.float32
    )
    return points_2d


# back project 2d image points with given camera matrix by using a random depths in the range [minD, maxD]
# output = [Nx3] (N = number of points)
def backproject_points(K, points_2d, minD, maxD):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z = np.random.uniform(minD, maxD, size=points_2d.shape[0])
    x = (points_2d[:, 0] - cx) * z / fx
    y = (points_2d[:, 1] - cy) * z / fy

    points_3d = np.vstack((x, y, z)).T
    return points_3d.astype(np.float32)


def project_points(K, points_3d_c, width, height):
    num_points = points_3d_c.shape[0]
    points_2d = np.zeros((num_points, 2), dtype=np.float32)
    mask = np.zeros((num_points, 1), dtype=np.float32)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    for i in range(num_points):
        point = points_3d_c[i, :]
        x = point[0] / point[2] * fx + cx
        y = point[1] / point[2] * fy + cy
        points_2d[i, :] = np.array([x, y])
        if x >= 0 and x < width and y >= 0 and y < height:
            mask[i] = 1
    return points_2d, mask


def check_solution(solver_input_data, scale12, Rc1c2, tc1c2):
    # check solution

    # transform points from world coordinates to camera coordinates
    points_3d_c1 = (
        solver_input_data.Rcw1 @ np.array(solver_input_data.points_3d_w1).T
        + solver_input_data.tcw1.reshape(3, 1)
    ).T
    points_3d_c2 = (
        solver_input_data.Rcw2 @ np.array(solver_input_data.points_3d_w2).T
        + solver_input_data.tcw2.reshape(3, 1)
    ).T

    aligned_points_c1 = (scale12 * Rc1c2 @ np.array(points_3d_c2).T + tc1c2.reshape(3, 1)).T
    average_alignment_error = np.mean(np.linalg.norm(aligned_points_c1 - points_3d_c1, axis=1))
    print(f"[check_solution] Average alignment error: {average_alignment_error}")
    return average_alignment_error


def check_solution2(solver_input_data2, scale12, Rc1c2, tc1c2):
    # check solution

    # transform points from world coordinates to camera coordinates
    points_3d_c1 = solver_input_data2.points_3d_c1
    points_3d_c2 = solver_input_data2.points_3d_c2

    aligned_points_c1 = (scale12 * Rc1c2 @ np.array(points_3d_c2).T + tc1c2.reshape(3, 1)).T
    average_alignment_error = np.mean(np.linalg.norm(aligned_points_c1 - points_3d_c1, axis=1))
    print(f"[check_solution2] Average alignment error: {average_alignment_error}")
    return average_alignment_error


def create_mock_input():
    # Creating mock data for Sim3SolverInput

    num_points = 100
    # grount-truth rotation and translation from world a to world b (this represents the pose estimation drift in Sim(3))
    gt_drift_R = rotation_matrix_from_yaw_pitch_roll(10.4, -5.6, 3.9)  # input in degrees
    gt_drift_t = np.array([0.1, -0.6, 0.2], dtype=np.float32)
    gt_drift_s = 0.9

    solver_input_data = sim3solver.Sim3SolverInput()
    solver_input_data.fix_scale = False

    # keyframes data
    solver_input_data.K1 = K  # Camera matrix for KF1
    solver_input_data.Rcw1 = np.eye(3, dtype=np.float32)  # Rotation matrix for KF1
    solver_input_data.tcw1 = np.array(
        [0.0, 0.0, 0.0], dtype=np.float32
    )  # Translation vector for KF1

    Rwc1 = solver_input_data.Rcw1.T
    twc1 = (-Rwc1 @ solver_input_data.tcw1.T).T

    solver_input_data.K2 = K  # Camera matrix for KF2 (first, it coincides with KF1)
    solver_input_data.Rcw2 = rotation_matrix_from_yaw_pitch_roll(0, 0, 0)  # Rotation matrix for KF2
    solver_input_data.tcw2 = np.array(
        [-0.1, 0.2, -0.7], dtype=np.float32
    )  # Translation vector for KF2

    Rwc2 = solver_input_data.Rcw2.T
    twc2 = (-Rwc2 @ solver_input_data.tcw2.T).T

    Rc2c1 = solver_input_data.Rcw2 @ solver_input_data.Rcw1.T
    tc2c1 = solver_input_data.tcw2 - (Rc2c1 @ solver_input_data.tcw1.T).T

    Rc1c2 = Rc2c1.T
    tc1c2 = -Rc1c2 @ tc2c1

    print(f"Rc1c2: {Rc1c2}")
    print(f"tc1c2: {tc1c2}")

    # Set a seed for reproducibility
    np.random.seed(42)  # You can change the seed value to any integer

    # generate random points in camera1 image and back project them with random depths
    points_2d_c1 = generate_random_points_2d(width, height, num_points)
    # print(f'points 2D shape: {points_2d_c1.shape}')
    # print(f'points 2D: {points_2d_c1}')
    points_3d_c1 = backproject_points(solver_input_data.K1, points_2d_c1, 1.0, 10.0)

    # check which points are visible in camera 2
    points_3d_c2 = (Rc2c1 @ points_3d_c1.T + tc2c1.reshape((3, 1))).T
    points_2d_c2, mask = project_points(solver_input_data.K2, points_3d_c2, width, height)

    # remove points that are not visible in camera 2
    mask = mask.ravel()
    points_2d_c1 = points_2d_c1[mask == 1, :]
    points_3d_c1 = points_3d_c1[mask == 1, :]
    points_2d_c2 = points_2d_c2[mask == 1, :]
    points_3d_c2 = points_3d_c2[mask == 1, :]

    points_3d_w1 = (Rwc1 @ points_3d_c1.T + twc1.reshape(3, 1)).T
    points_3d_w2 = points_3d_w1.copy()
    print(f"visible 3D points shape: {points_3d_w1.shape}")
    # print(f'points 3D: {points_3d_w1}')

    # check projections
    if False:
        points_2d_c1_check, mask_c1 = project_points(
            solver_input_data.K1,
            (solver_input_data.Rcw1 @ points_3d_w1.T + solver_input_data.tcw1.reshape(3, 1)).T,
            width,
            height,
        )
        average_projection_error_c1 = np.mean(
            np.linalg.norm(points_2d_c1 - points_2d_c1_check, axis=1)
        )
        print(
            f"[double-check] average projection error 1: {average_projection_error_c1}, mask = {np.sum(mask_c1)}"
        )

        points_2d_c2_check, mask_c2 = project_points(
            solver_input_data.K2,
            (solver_input_data.Rcw2 @ points_3d_w2.T + solver_input_data.tcw2.reshape(3, 1)).T,
            width,
            height,
        )
        average_projection_error_c2 = np.mean(
            np.linalg.norm(points_2d_c2 - points_2d_c2_check, axis=1)
        )
        print(
            f"[duble-check] average projection error 2: {average_projection_error_c2}, mask = {np.sum(mask_c2)}"
        )

    # print(f'original Rc1c2: {Rc1c2}')
    # print(f'original tc1c2: {tc1c2}')

    # now we have perfectly matched points
    # let's simulate a drift on camera 2
    # Tc1c2_d = Tc1c2 * gt_drift_T
    Rc1c2_d = Rc1c2 @ gt_drift_R
    tc1c2_d = Rc1c2 @ gt_drift_t + tc1c2
    points_3d_c2_d = gt_drift_s * points_3d_c2  # scaling is applied in camera frame

    # print(f'drifted Rc1c2: {Rc1c2_d}')
    # print(f'drifted tc1c2: {tc1c2_d}')

    Rwc2_d = Rwc1 @ Rc1c2_d
    twc2_d = Rwc1 @ tc1c2_d + twc1
    Rcw2_d = Rwc2_d.T
    tcw2_d = -Rcw2_d @ twc2_d

    # camera 2 does not know anything about its drift
    # solver_input_data.Rcw2 = Rcw2_d
    # solver_input_data.tcw2 = tcw2_d

    points_3d_w2_d = (Rwc2_d @ points_3d_c2_d.T + twc2_d.reshape(3, 1)).T

    # test = solver_input_data.Rcw1 @ Rcw2_d

    # matched 3D points
    solver_input_data.points_3d_w1 = points_3d_w1
    solver_input_data.points_3d_w2 = points_3d_w2_d
    print(f"3D points 1 shape: {points_3d_w1.shape}")
    print(f"3D points 2 shape: {points_3d_w2_d.shape}")

    # Mock sigma squared data
    solver_input_data.sigmas2_1 = [1.0 for _ in range(num_points)]
    solver_input_data.sigmas2_2 = [1.0 for _ in range(num_points)]

    solver_input_data.fix_scale = False  # Scale is not fixed

    # Correction includes the inverse of the drift
    inv_drift_R = gt_drift_R.T
    inv_drift_t = -inv_drift_R @ gt_drift_t / gt_drift_s
    Rcorr = Rc1c2 @ inv_drift_R
    tcorr = Rc1c2 @ inv_drift_t + tc1c2
    scorr = 1 / gt_drift_s
    print(f"correction R: {Rcorr}")
    print(f"correction t: {tcorr}")
    print(f"correction s: {scorr}")

    check_solution(solver_input_data, scorr, Rcorr, tcorr)

    return solver_input_data


def test_sim3solver(solver_input_data):

    print("----------------------------------------------")

    if False:
        rr.init("sim3solver_test", spawn=True)
        rr.log("points_3d_w1", rr.Points3D(solver_input_data.points_3d_w1, colors=[255, 0, 0]))
        rr.log("points_3d_w2", rr.Points3D(solver_input_data.points_3d_w2, colors=[0, 255, 0]))

    if False:
        # test python solution by using points in world coordinates
        R, t = horn_absolute_orientation(
            np.array(solver_input_data.points_3d_w1), np.array(solver_input_data.points_3d_w2)
        )
        print(f"python solution: R: {R}, t: {t}")

    # Create Sim3Solver object with the input data
    solver = sim3solver.Sim3Solver(solver_input_data)

    # Set RANSAC parameters (using defaults here)
    solver.set_ransac_parameters(0.99, 20, 300)

    # Prepare variables for iterative solving
    vbInliers = [False] * len(solver_input_data.points_3d_w1)
    nInliers = 0
    bConverged = False

    # Test the first iteration (e.g., 10 iterations)
    transformation, bNoMore, vbInliers, nInliers, bConverged = solver.iterate(5)

    if False:
        print("Estimated transformation after 10 iterations:")
        print(transformation)

    # Check if the solver returned inliers and transformation
    print("Number of inliers:", nInliers)
    print(f"bConverged: {bConverged}")
    # print("Vector of inliers:", vbInliers)

    # Test getting the best transformation
    Tc1c2 = solver.get_estimated_transformation()
    # print("Estimated transformation matrix Tc1c2:")
    # print(Tc1c2)

    # Test getting estimated rotation, translation, and scale
    Rc1c2 = solver.get_estimated_rotation()
    tc1c2 = solver.get_estimated_translation()
    scale12 = solver.get_estimated_scale()

    error3d = solver.compute_3d_registration_error()

    print("Estimated rotation matrix Rc1c2:")
    print(Rc1c2)
    print("Estimated translation vector tc1c2:")
    print(tc1c2)
    print("Estimated scale scale12:")
    print(scale12)
    print("error3d: ", error3d)

    # check solution
    check_solution(solver_input_data, scale12, Rc1c2, tc1c2)


def test_sim3solver2(solver_input_data):

    print("----------------------------------------------")

    # Here we use a the second input data format
    solver_input_data2 = sim3solver.Sim3SolverInput2()
    solver_input_data2.points_3d_c1 = (
        solver_input_data.Rcw1 @ np.array(solver_input_data.points_3d_w1).T
        + solver_input_data.tcw1.reshape(3, 1)
    ).T
    solver_input_data2.points_3d_c2 = (
        solver_input_data.Rcw2 @ np.array(solver_input_data.points_3d_w2).T
        + solver_input_data.tcw2.reshape(3, 1)
    ).T
    solver_input_data2.sigmas2_1 = solver_input_data.sigmas2_1
    solver_input_data2.sigmas2_2 = solver_input_data.sigmas2_2
    solver_input_data2.fix_scale = solver_input_data.fix_scale

    # Create Sim3Solver object with the input data
    solver = sim3solver.Sim3Solver(solver_input_data2)

    # Set RANSAC parameters (using defaults here)
    solver.set_ransac_parameters(0.99, 20, 300)

    # Prepare variables for iterative solving
    vbInliers = [False] * len(solver_input_data2.points_3d_c1)
    nInliers = 0
    bConverged = False

    # Test the first iteration (e.g., 10 iterations)
    transformation, bNoMore, vbInliers, nInliers, bConverged = solver.iterate(5)

    if False:
        print("Estimated transformation after 10 iterations:")
        print(transformation)

    # Check if the solver returned inliers and transformation
    print("Number of inliers:", nInliers)
    print(f"bConverged: {bConverged}")
    # print("Vector of inliers:", vbInliers)

    # Test getting the best transformation
    Tc1c2 = solver.get_estimated_transformation()
    # print("Estimated transformation matrix Tc1c2:")
    # print(Tc1c2)

    # Test getting estimated rotation, translation, and scale
    Rc1c2 = solver.get_estimated_rotation()
    tc1c2 = solver.get_estimated_translation()
    scale12 = solver.get_estimated_scale()

    error3d = solver.compute_3d_registration_error()

    print("Estimated rotation matrix Rc1c2:")
    print(Rc1c2)
    print("Estimated translation vector tc1c2:")
    print(tc1c2)
    print("Estimated scale scale12:")
    print(scale12)
    print("error3d: ", error3d)

    # check solution
    check_solution2(solver_input_data2, scale12, Rc1c2, tc1c2)


if __name__ == "__main__":
    # Create a Sim3SolverInput object with mock data
    solver_input_data = create_mock_input()

    test_sim3solver(solver_input_data)
    test_sim3solver2(solver_input_data)
