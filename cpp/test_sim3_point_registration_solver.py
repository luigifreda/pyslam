import sys
import numpy as np
import time

import rerun as rr
import cv2
from matplotlib import pyplot as plt

sys.path.append("./lib/")
import sim3solver


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
def generate_random_points_3d(min, max, num_points):
    points_3d = np.random.uniform(
        low=[min, min, min], high=[max, max, max], size=(num_points, 3)
    ).astype(np.float32)
    return points_3d


def check_solution(solver_input_data, scale12, R12, t12):
    # check solution

    # transform points from world coordinates to camera coordinates
    points_3d_w1 = np.array(solver_input_data.points_3d_w1)
    points_3d_w2 = np.array(solver_input_data.points_3d_w2)

    aligned_points_2 = (scale12 * R12 @ points_3d_w2.T + t12.reshape(3, 1)).T
    average_alignment_error = np.mean(np.linalg.norm(aligned_points_2 - points_3d_w1, axis=1))
    print(f"[check_solution] Average alignment error: {average_alignment_error}")
    return average_alignment_error


def create_mock_input(num_points=100, min3d=-10, max3d=10, noise_std=0.001):
    # Creating mock data for Sim3PointRegistrationSolverInput

    # grount-truth rotation and translation from world a to world b (this represents the pose estimation drift in Sim(3))
    gt_drift_R = rotation_matrix_from_yaw_pitch_roll(10.4, -5.6, 3.9)  # input in degrees
    gt_drift_t = np.array([0.1, -0.6, 0.2], dtype=np.float32)
    gt_drift_s = 0.9

    solver_input_data = sim3solver.Sim3PointRegistrationSolverInput()
    solver_input_data.fix_scale = False

    # Set a seed for reproducibility
    np.random.seed(42)  # You can change the seed value to any integer

    # generate random points in camera1 image and back project them with random depths
    points_3d_w1 = generate_random_points_3d(min3d, max3d, num_points)
    points_3d_w2 = (gt_drift_s * gt_drift_R @ points_3d_w1.T + gt_drift_t.reshape(3, 1)).T

    # Add Gaussian noise to 3D points
    noise = np.random.normal(0, noise_std, points_3d_w2.shape)
    points_3d_w2_n = points_3d_w2 + noise

    # matched 3D points
    solver_input_data.points_3d_w1 = points_3d_w1
    solver_input_data.points_3d_w2 = points_3d_w2_n
    print(f"3D points 1 shape: {points_3d_w1.shape}")
    print(f"3D points 2 shape: {points_3d_w2_n.shape}")

    solver_input_data.fix_scale = False  # Scale is not fixed

    # Correction includes the inverse of the drift
    inv_drift_R = gt_drift_R.T
    inv_drift_t = -inv_drift_R @ gt_drift_t / gt_drift_s
    Rcorr = inv_drift_R
    tcorr = inv_drift_t
    scorr = 1 / gt_drift_s
    print(f"correction R: {Rcorr}")
    print(f"correction t: {tcorr}")
    print(f"correction s: {scorr}")

    check_solution(solver_input_data, scorr, Rcorr, tcorr)

    return solver_input_data


def test_sim3solver():

    num_points = 100

    # Create a Sim3SolverInput object with mock data
    solver_input_data = create_mock_input(num_points)

    print("----------------------------------------------")

    # Create Sim3PointRegistrationSolver object with the input data
    solver = sim3solver.Sim3PointRegistrationSolver(solver_input_data)

    # Set RANSAC parameters (using defaults here)
    solver.set_ransac_parameters(0.99, 20, 300)

    # Prepare variables for iterative solving
    vbInliers = [False] * len(solver_input_data.points_3d_w1)
    nInliers = 0
    bConverged = False

    # Test the first iteration (e.g., 10 iterations)
    time_start = time.time()
    transformation, bNoMore, vbInliers, nInliers, bConverged = solver.iterate(5)
    time_end = time.time()
    print(f"Elapsed time: {time_end - time_start} seconds")

    if False:
        print("Estimated transformation after 10 iterations:")
        print(transformation)

    # Check if the solver returned inliers and transformation
    print("Number of inliers:", nInliers)
    print(f"bConverged: {bConverged}")
    # print("Vector of inliers:", vbInliers)

    # Test getting the best transformation
    T12 = solver.get_estimated_transformation()
    # print("Estimated transformation matrix Tc1c2:")
    # print(Tc1c2)

    # Test getting estimated rotation, translation, and scale
    R12 = solver.get_estimated_rotation()
    t12 = solver.get_estimated_translation()
    scale12 = solver.get_estimated_scale()

    error3d = solver.compute_3d_registration_error()

    print("Estimated rotation matrix R12:")
    print(R12)
    print("Estimated translation vector t12:")
    print(t12)
    print("Estimated scale scale12:")
    print(scale12)
    print("error3d: ", error3d)

    # check solution
    check_solution(solver_input_data, scale12, R12, t12)


if __name__ == "__main__":
    test_sim3solver()
