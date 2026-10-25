import sys
import cv2
import numpy as np


from pyslam.utilities.depth import (
    depth2pointcloud,
    img_from_depth,
    filter_shadow_points,
    PointCloud,
)


from pyslam.viz.viewer3D import Viewer3D, VizPointCloud, VizMesh, VizCameraImage
from pyslam.utilities.img_management import ImageTable
from pyslam.utilities.file_management import select_image_files
from pyslam.utilities.geometry import poseRt, inv_poseRt
import time

import glob, os, torch
from depth_anything_3.api import DepthAnything3


depth_anything_v2_path = "../../thirdparty/depth_anything_v3"
data_path = "../data/"

# 1)
# image_paths = ["kitti06-13-color.png", "kitti06-14-color.png", "kitti06-17-color.png"]
# image_paths = [data_path + image_path for image_path in image_paths]

# 2)
# example_path = data_path + "SOH"
# image_paths = sorted(glob.glob(os.path.join(example_path, "*.png")))


# 3) TUM desk long_office_household (no distortion here)
images_path = (
    "/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg3_long_office_household/rgb"
)
start_frame_name = "1341847980.722988.png"
n_frame = 10
delta_frame = 30
image_paths = select_image_files(
    images_path, start_frame_name, n_frame=n_frame, delta_frame=delta_frame
)
image_paths = [os.path.join(images_path, x) for x in image_paths]


# create a scaled image of uint8 from a image of floats
def img_from_depth_local(img_flt, img_max=None, img_min=None, eps=1e-9):
    assert img_flt.dtype in [np.float32, np.float64, np.float16, np.double, np.single]
    img_max = np.max(img_flt) if img_max is None else img_max
    img_min = np.min(img_flt) if img_min is None else img_min
    mean = np.mean(img_flt)
    print(f"img_max: {img_max}, img_min: {img_min}, mean: {mean}")
    if img_max is not None or img_min is not None:
        img_flt = np.clip(img_flt, img_min, img_max)
    img_range = img_max - img_min
    if img_range < eps:
        img_range = 1
    img = (img_flt - img_min) / img_range * 255
    return img.astype(np.uint8)


if __name__ == "__main__":

    images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"image path {image_path} does not exist")
            raise ValueError(f"image path {image_path} does not exist")
        images.append(cv2.imread(image_path, cv2.IMREAD_COLOR))

    device = torch.device("cuda")

    # model_type = "depth-anything/DA3NESTED-GIANT-LARGE"
    model_type = "depth-anything/DA3-LARGE"
    # model_type = "depth-anything/DA3METRIC-LARGE"  # metric depth estimation, no poses available

    model = DepthAnything3.from_pretrained(model_type)

    model = model.to(device=device)

    depth_prediction = model.inference(
        images,
    )
    # prediction.processed_images : [N, H, W, 3] array (check dtype and range)
    print(f"processed_images shape: {depth_prediction.processed_images.shape}")
    print(f"processed_images dtype: {depth_prediction.processed_images.dtype}")
    print(
        f"processed_images range: [{depth_prediction.processed_images.min():.3f}, {depth_prediction.processed_images.max():.3f}]"
    )
    # prediction.depth            : [N, H, W]    float32 array
    print(f"depth shape: {depth_prediction.depth.shape}")
    if depth_prediction.conf is not None:
        # prediction.conf             : [N, H, W]    float32 array
        print(f"conf shape: {depth_prediction.conf.shape}")
    else:
        print("conf is None")
    if depth_prediction.extrinsics is not None:
        # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
        print(f"extrinsics shape: {depth_prediction.extrinsics.shape}")
    else:
        print("extrinsics is None")
    if depth_prediction.intrinsics is not None:
        # prediction.intrinsics       : [N, 3, 3]    float32 array
        print(f"intrinsics shape: {depth_prediction.intrinsics.shape}")
    else:
        print("intrinsics is None")

    raw_images = []  # raw input images
    processed_images = []  # images interally processed by the model
    depth_predictions = []  # depth predictions from the model
    rescaled_depth_predictions = []  # depth predictions rescaled to the original image dimensions
    depth_images = []  # depth images
    point_clouds = []  # point clouds
    extrinsics = []  # extrinsics
    intrinsics = []  # intrinsics

    poses_are_available = (
        depth_prediction.extrinsics is not None and depth_prediction.intrinsics is not None
    )

    for i, image in enumerate(images):
        raw_images.append(image.copy())
        processed_images.append(depth_prediction.processed_images[i])

        # Filter depth
        filter_depth = True  # do you want to filter the depth?
        if filter_depth:
            depth_filtered = filter_shadow_points(depth_prediction.depth[i], delta_depth=None)
        else:
            depth_filtered = depth_prediction.depth[i]

        depth_predictions.append(depth_filtered)

        if depth_prediction.extrinsics is not None:
            extrinsics.append(depth_prediction.extrinsics[i])
        else:
            extrinsics.append(None)

        if depth_prediction.intrinsics is not None:
            intrinsics.append(depth_prediction.intrinsics[i])
        else:
            intrinsics.append(None)

        # Rescale depth prediction to match original image dimensions if needed
        if depth_predictions[i].shape[:2] != image.shape[:2]:
            # Use INTER_NEAREST for depth to preserve discrete depth values
            # Use INTER_LINEAR if smooth depth maps are preferred
            rescaled_depth_predictions.append(
                cv2.resize(
                    depth_predictions[i],
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
        else:
            rescaled_depth_predictions.append(depth_predictions[i])

        depth_images.append(depth_predictions[i])

        if poses_are_available and intrinsics[i] is not None:
            # Convert image to RGB format for depth2pointcloud
            if len(raw_images[i].shape) == 2 or raw_images[i].shape[2] == 1:
                # Grayscale image (1 channel)
                raw_images[i] = cv2.cvtColor(raw_images[i], cv2.COLOR_GRAY2RGB)
            else:
                raw_images[i] = cv2.cvtColor(raw_images[i], cv2.COLOR_BGR2RGB)

            fx = intrinsics[i][0, 0]
            fy = intrinsics[i][1, 1]
            cx = intrinsics[i][0, 2]
            cy = intrinsics[i][1, 2]

            # Verify intrinsics match depth map dimensions
            depth_h, depth_w = depth_predictions[i].shape[:2]
            processed_h, processed_w = processed_images[i].shape[:2]
            print(
                f"Image {i}: depth shape={depth_predictions[i].shape}, processed shape={processed_images[i].shape}"
            )
            print(f"Image {i}: intrinsics fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

            point_cloud = depth2pointcloud(
                depth_predictions[i],
                processed_images[i] * 255.0,
                fx,
                fy,
                cx,
                cy,
            )
            # transform point cloud to world coordinates
            # Depth Anything 3 extrinsics are in w2c (world-to-camera) format
            # We need to convert to c2w (camera-to-world) to transform points
            Rcw = extrinsics[i][:3, :3]
            tcw = extrinsics[i][:3, 3]
            Twc = inv_poseRt(Rcw, tcw)
            Rwc = Twc[:3, :3]
            twc = Twc[:3, 3].reshape(3, 1)
            print(
                f"Image {i}: Camera position (world): [{twc[0,0]:.3f}, {twc[1,0]:.3f}, {twc[2,0]:.3f}]"
            )
            print(f"point_cloud.points shape: {point_cloud.points.shape}")
            point_cloud.points = (Rwc @ point_cloud.points.T + twc).T
        else:
            point_cloud = None
        point_clouds.append(point_cloud)

    for i, image in enumerate(raw_images):
        # cv2.imshow expects BGR format, but raw_images may be RGB after conversion
        # Convert back to BGR for display if needed
        if poses_are_available and intrinsics[i] is not None:
            # Image was converted to RGB, convert back to BGR for display
            display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # Image is still in BGR format from cv2.imread
            display_image = image
        cv2.imshow(f"raw image_{i}", display_image)
        cv2.imshow(f"processed image_{i}", processed_images[i])
        cv2.imshow(f"depth_{i}", depth_images[i])
        # cv2.imshow(f"rescaled depth_{i}", rescaled_depth_predictions[i])

    if poses_are_available:
        viewer3D = Viewer3D()
        time.sleep(1)

        # merge all point clouds
        global_points = np.concatenate(
            [point_clouds[i].points for i in range(len(point_clouds))], axis=0
        )
        global_colors = np.concatenate(
            [point_clouds[i].colors for i in range(len(point_clouds))], axis=0
        )

        viz_point_cloud = VizPointCloud(
            points=global_points,
            colors=global_colors,
            normalize_colors=True,
            reverse_colors=True,
        )
        viz_camera_images = []
        for i, img in enumerate(raw_images):
            # Depth Anything 3 extrinsics are already in w2c (world-to-camera) format
            # which is what VizCameraImage expects (Twc)
            Rcw = extrinsics[i][:3, :3]
            tcw = extrinsics[i][:3, 3]
            Twc = inv_poseRt(Rcw, tcw)
            viz_camera_images.append(VizCameraImage(image=img, Twc=Twc, scale=0.1))
        viewer3D.draw_dense_geometry(point_cloud=viz_point_cloud, camera_images=viz_camera_images)

        while viewer3D.is_running():
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q") or key == 27:
                break

        viewer3D.quit()

    else:
        print("poses are not available")
        cv2.waitKey()
