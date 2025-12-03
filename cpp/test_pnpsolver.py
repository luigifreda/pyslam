import numpy as np
import cv2
import sys

import matplotlib.pyplot as plt

sys.path.append("./lib/")
import pnpsolver as pnp


fx = 517.306408
fy = 516.469215
cx = 318.643040
cy = 255.313989
width = 640
height = 480
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


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


def generate_synthetic_data(num_points=100, noise_std=1.0):
    """
    Generates synthetic 3D points, projects them to 2D using a known pose,
    and adds Gaussian noise to the 2D points.
    """
    # np.random.seed(42)  # For reproducibility

    # grount-truth rotation and translation
    angular_range_factor = 0.3
    rad_to_deg = 180.0 / np.pi
    yaw = angular_range_factor * np.random.uniform(-np.pi, np.pi) * rad_to_deg
    pitch = angular_range_factor * np.random.uniform(-np.pi / 2, np.pi / 2) * rad_to_deg
    roll = angular_range_factor * np.random.uniform(-np.pi / 2, np.pi / 2) * rad_to_deg
    gt_R = rotation_matrix_from_yaw_pitch_roll(yaw, pitch, roll)  # input in degrees

    tx = np.random.uniform(-5, 5)
    ty = np.random.uniform(-5, 5)
    tz = np.random.uniform(-5, 5)
    gt_t = np.array([tx, ty, tz], dtype=np.float32)

    # Define a known rotation (identity) and translation
    R_true = gt_R
    t_true = gt_t

    # Generate random 3D points in front of the camera

    points_3d_w = np.random.uniform(-5, 5, (num_points, 3))

    # Project 3D points to 2D using the known pose
    points_3d_c = (R_true @ points_3d_w.T).T + t_true  # Transform to camera coordinates

    points_2d_c, mask = project_points(K, points_3d_c, width, height)

    # remove points that are not visible in camera 2
    mask = mask.ravel()
    points_2d_c = points_2d_c[mask == 1, :]
    points_3d_c = points_3d_c[mask == 1, :]
    points_3d_w = points_3d_w[mask == 1, :]

    print(f"Number of visible points: {points_2d_c.shape[0]}")

    # Add Gaussian noise to 2D points
    noise = np.random.normal(0, noise_std, points_2d_c.shape)
    points_2d_noisy = points_2d_c + noise

    # Define sigmas squared (variance of the noise)
    sigmas2 = np.full((num_points,), noise_std**2)

    return points_2d_noisy, points_3d_w, sigmas2, fx, fy, cx, cy, R_true, t_true


def solve_pnp(solver_input, R_true, t_true, points_3d, points_2d, use_MLPnP=False):
    solver_type = "MLPnPsolver" if use_MLPnP else "PnPsolver"
    solver = pnp.MLPnPsolver(solver_input) if use_MLPnP else pnp.PnPsolver(solver_input)

    # Run the PnP solver
    ok, transformation, no_more, inliers, n_inliers = solver.iterate(5)

    inliers = np.array(inliers).astype(bool)

    # print(f'transformation: {transformation}, inliers: {inliers}, n_inliers: {n_inliers}')

    print(f"---- {solver_type} results ----")

    if n_inliers == 0:
        print(f"{solver_type}: No inliers found. Exiting.")
        return (float("inf"), float("inf"))

    # Extract rotation and translation from the transformation matrix
    R_est = transformation[:3, :3]
    t_est = transformation[:3, 3]

    # Calculate errors
    rotation_error = np.arccos((np.trace(R_est.T @ R_true) - 1) / 2) * (180.0 / np.pi)  # in degrees
    translation_error = np.linalg.norm(t_est - t_true)

    # Print results
    print(f"Type: {solver_type}")
    print(f"Number of correspondences: {len(points_3d)}")
    print(f"Estimated Rotation:\n{R_est}")
    print(f"True Rotation:\n{R_true}")
    print(f"Rotation Error (degrees): {rotation_error:.4f}")
    print(f"Estimated Translation: {t_est}")
    print(f"True Translation: {t_true}")
    print(f"Translation Error: {translation_error:.4f}")
    print(f"Number of Inliers: {n_inliers} / {len(points_3d)}")

    # Optionally, visualize inliers
    visualize = False
    if visualize:

        inlier_indices = np.where(inliers == True)
        outlier_indices = np.where(inliers == False)

        plt.figure(figsize=(8, 6))
        plt.scatter(points_2d[:, 0], points_2d[:, 1], c="r", label="Outliers", marker="x")
        plt.scatter(
            points_2d[inlier_indices, 0],
            points_2d[inlier_indices, 1],
            c="g",
            label="Inliers",
            marker="o",
        )
        plt.title(f"{solver_type} Solver Inliers and Outliers")
        plt.xlabel("u (pixels)")
        plt.ylabel("v (pixels)")
        plt.legend()
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates

    return rotation_error, translation_error


def main():
    num_points = 500
    noise_std = 1.0

    num_tests = 100

    errors_mlpnp = []
    errors_pnp = []

    for i in range(num_tests):
        print("=====================================================")
        print(f"Test {i+1}/{num_tests}")

        # Generate synthetic data
        points_2d, points_3d, sigmas2, fx, fy, cx, cy, R_true, t_true = generate_synthetic_data(
            num_points=num_points, noise_std=noise_std
        )

        # Create PnPsolverInput instance
        solver_input = pnp.PnPsolverInput()
        solver_input.points_2d = points_2d.tolist()
        solver_input.points_3d = points_3d.tolist()
        solver_input.sigmas2 = sigmas2.tolist()
        solver_input.fx = fx
        solver_input.fy = fy
        solver_input.cx = cx
        solver_input.cy = cy

        print()
        rotation_error, translation_error = solve_pnp(
            solver_input, R_true, t_true, points_3d, points_2d, use_MLPnP=False
        )
        errors_mlpnp.append([rotation_error, translation_error])
        print()
        rotation_error, translation_error = solve_pnp(
            solver_input, R_true, t_true, points_3d, points_2d, use_MLPnP=True
        )
        errors_pnp.append([rotation_error, translation_error])

    print()
    print(f"=== Summary ===")
    num_failures_mlpnp = len([error for error in errors_mlpnp if error[0] == float("inf")])
    num_failures_pnp = len([error for error in errors_pnp if error[0] == float("inf")])

    print(f"Number of failures MLPnP: {num_failures_mlpnp}")
    print(f"Number of failures PnP: {num_failures_pnp}")

    mean_t_err_mlpnp = np.mean([error[1] for error in errors_mlpnp if error[1] != float("inf")])
    mean_t_err_pnp = np.mean([error[1] for error in errors_pnp if error[1] != float("inf")])

    print(f"Mean translation error MLPnP: {mean_t_err_mlpnp:.4f} m")
    print(f"Mean translation error PnP: {mean_t_err_pnp:.4f} m")

    mean_r_err_mlpnp = np.mean([error[0] for error in errors_mlpnp if error[0] != float("inf")])
    mean_r_err_pnp = np.mean([error[0] for error in errors_pnp if error[0] != float("inf")])

    print(f"Mean rotation error MLPnP: {mean_r_err_mlpnp:.4f} deg")
    print(f"Mean rotation error PnP: {mean_r_err_pnp:.4f} deg")

    # bar plots
    fig, ax = plt.subplots()
    ax.bar(np.arange(1, num_tests + 1), [error[0] for error in errors_mlpnp], label="MLPnP")
    ax.bar(np.arange(1, num_tests + 1), [error[0] for error in errors_pnp], label="PnP")
    ax.set_title("Rotation Error")
    ax.set_xlabel("Test")
    ax.set_ylabel("Rotation Error (degrees)")
    ax.legend()

    fig, ax = plt.subplots()
    ax.bar(np.arange(1, num_tests + 1), [error[1] for error in errors_mlpnp], label="MLPnP")
    ax.bar(np.arange(1, num_tests + 1), [error[1] for error in errors_pnp], label="PnP")
    ax.set_title("Translation Error")
    ax.set_xlabel("Test")
    ax.set_ylabel("Translation Error (m)")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
