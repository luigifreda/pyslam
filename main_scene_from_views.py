#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.

Example script demonstrating how to use the SceneFromViews factory and pipelines
for 3D scene reconstruction from multiple views.

Usage examples:
    # Basic usage with Depth Anything V3
    python main_scene_from_views.py --image_dir ../data/SOH --method DEPTH_ANYTHING_V3

    # Using MASt3R with custom parameters
    python main_scene_from_views.py --image_dir ../data/SOH --method MAST3R --inference_size 512 --min_conf_thr 1.5

    # Using VGGT with point cloud output
    python main_scene_from_views.py --image_dir ../data/SOH --method VGGT --as_pointcloud

    # Process only first 5 images
    python main_scene_from_views.py --image_dir ../data/SOH --max_images 5 --method MVDUST3R

    # Skip 3D visualization, only show 2D images
    python main_scene_from_views.py --image_dir ../data/SOH --no_3d
"""

import os
import sys
import cv2
import numpy as np
import argparse
import glob
import time

from pyslam.utilities.logging import Printer
from pyslam.scene_from_views import (
    SceneFromViewsType,
    scene_from_views_factory,
    SceneFromViewsResult,
)
from pyslam.viz.viewer3D import Viewer3D, VizPointCloud, VizMesh, VizCameraImage
from pyslam.utilities.img_management import ImageTable, img_from_floats
from pyslam.utilities.file_management import select_image_files, load_images_from_directory


kThisFileFolder = os.path.dirname(os.path.abspath(__file__))
kDataFolder = os.path.join(kThisFileFolder, "data/test_data")

# TUM desk long_office_household (no distortion here)
images_path = (
    "/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg3_long_office_household/rgb"
)
start_frame_name = "1341847980.722988.png"
n_frame = 5
delta_frame = 100


def visualize_results(
    result: SceneFromViewsResult,
    show_images=True,
    show_3d=True,
    reverse_3d_colors=True,
    original_images=None,
    method=None,
):
    """
    Visualize reconstruction results.

    Args:
        result: SceneFromViewsResult containing reconstruction data
        show_images: Whether to show 2D image visualizations
        show_3d: Whether to show 3D visualization
        reverse_3d_colors: Whether to reverse colors for 3D visualization
        original_images: Optional list of original input images to display
    """
    if show_images:
        # Show original input images if provided
        if original_images is not None and len(original_images) > 0:
            orig_table = ImageTable(num_columns=min(4, len(original_images)), resize_scale=0.8)
            added = 0
            for img in original_images:
                if orig_table.add(img):
                    added += 1
            if added > 0:
                orig_table.render()
                cv2.imshow("Original Input Images", orig_table.image() if added > 0 else None)

        # Show processed images
        if len(result.processed_images) > 0:
            img_table = ImageTable(
                num_columns=min(4, len(result.processed_images)), resize_scale=0.8
            )
            added = 0
            for img in result.processed_images:
                if img_table.add(img):
                    added += 1
            if added > 0:
                img_table.render()
                cv2.imshow("Processed Images", img_table.image())
                cv2.waitKey(1)

        # Show depth maps if available
        if result.depth_predictions is not None and len(result.depth_predictions) > 0:
            depth_table = ImageTable(
                num_columns=min(4, len(result.depth_predictions)), resize_scale=0.8
            )
            added = 0
            for depth in result.depth_predictions:
                depth_img = img_from_floats(depth)
                if depth_table.add(depth_img):
                    added += 1
            if added > 0:
                depth_table.render()
                cv2.imshow("Depth Maps", depth_table.image())
                cv2.waitKey(1)

        # Show confidence maps if available
        if result.confidences is not None and len(result.confidences) > 0:
            conf_table = ImageTable(num_columns=min(4, len(result.confidences)), resize_scale=0.8)
            added = 0
            for conf in result.confidences:
                conf_img = img_from_floats(conf)
                if conf_table.add(conf_img):
                    added += 1
            if added > 0:
                conf_table.render()
                cv2.imshow("Confidence Maps", conf_table.image())
                cv2.waitKey(1)

        # Pump the OpenCV event loop to make sure the above windows become visible
        cv2.waitKey(1)

    if show_3d:
        viewer3D = Viewer3D()
        time.sleep(1)

        # Visualize point cloud or mesh
        viz_point_cloud = None
        viz_mesh = None

        if result.global_point_cloud is not None:
            viz_point_cloud = VizPointCloud(
                points=result.global_point_cloud.vertices,
                colors=result.global_point_cloud.colors,
                normalize_colors=True,
                reverse_colors=reverse_3d_colors,
            )
            Printer.green(f"Point cloud: {len(result.global_point_cloud.vertices)} points")

        if result.global_mesh is not None:
            viz_mesh = VizMesh(
                vertices=result.global_mesh.vertices,
                triangles=result.global_mesh.faces,
                vertex_colors=result.global_mesh.visual.vertex_colors,
                normalize_colors=True,
            )
            Printer.green(
                f"Mesh: {len(result.global_mesh.vertices)} vertices, {len(result.global_mesh.faces)} faces"
            )

        # Visualize camera poses if available
        viz_camera_images = []
        cam_scale = 0.1
        if method == SceneFromViewsType.DUST3R or method == SceneFromViewsType.FAST3R:
            cam_scale = 0.03
        if result.camera_poses is not None and len(result.camera_poses) > 0:
            for i, (pose, img) in enumerate(zip(result.camera_poses, result.processed_images)):
                if pose is not None:
                    if reverse_3d_colors:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img_rgb = img
                    viz_camera_images.append(
                        VizCameraImage(image=img_rgb, Twc=pose, scale=cam_scale)
                    )
            Printer.green(f"Camera poses: {len(viz_camera_images)} cameras")

        viewer3D.draw_dense_geometry(
            point_cloud=viz_point_cloud,
            mesh=viz_mesh,
            camera_images=viz_camera_images if len(viz_camera_images) > 0 else None,
        )

        # Wait for user to quit
        while viewer3D.is_running():
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q") or key == 27:
                break

        viewer3D.quit()


def main():
    parser = argparse.ArgumentParser(
        description="3D Scene Reconstruction from Multiple Views",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # see the file pyslam/scene_from_views/scene_from_views_types.py for the available methods
    # NOTE: VGGT requires a decent amount of GPU memory
    parser.add_argument(
        "--method",
        type=str,
        default="VGGT_ROBUST",
        choices=[
            "DUST3R",
            "MAST3R",
            "MVDUST3R",
            "VGGT",
            "VGGT_ROBUST",
            "DEPTH_ANYTHING_V3",
            "FAST3R",
        ],
        help="Reconstruction method to use",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=kDataFolder + "/tum_office",
        help="Directory containing input images",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern for image files",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (None for all)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or None for auto)",
    )

    parser.add_argument(
        "--skip_optimizer",
        action="store_true",
        help="Skip optimizer and use initial poses from Dust3r inference (for debugging)",
    )

    parser.add_argument(
        "--as_pointcloud",
        action="store_true",
        default=True,
        help="Return point cloud instead of mesh",
    )

    parser.add_argument(
        "--no_3d",
        action="store_true",
        help="Skip 3D visualization",
    )

    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Skip 2D image visualization",
    )

    # Model-specific arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type (e.g., 'depth-anything/DA3-LARGE' for Depth Anything V3)",
    )

    parser.add_argument(
        "--inference_size",
        type=int,
        default=None,
        help="Inference image size (model-specific)",
    )

    parser.add_argument(
        "--min_conf_thr",
        type=float,
        default=None,
        help="Minimum confidence threshold (model-specific)",
    )

    args = parser.parse_args()

    # Convert method string to enum
    try:
        method = SceneFromViewsType.from_string(args.method)
    except ValueError as e:
        Printer.red(f"Invalid method: {args.method}")
        Printer.red(f"Available methods: {[t.name for t in SceneFromViewsType]}")
        sys.exit(1)

    Printer.green(f"Using scene reconstruction method: {method.name}")

    # Load images
    Printer.blue(f"Loading images from: {args.image_dir}")
    if not os.path.isdir(args.image_dir):
        Printer.red(f"Directory does not exist: {args.image_dir}")
        sys.exit(1)

    images = load_images_from_directory(args.image_dir, args.pattern, args.max_images)

    if True:  # DEBUG: override images with selected image files
        image_filenames = select_image_files(
            images_path, start_frame_name, n_frame=n_frame, delta_frame=delta_frame
        )
        print(f"selected image files: {image_filenames}")
        img_paths = [os.path.join(images_path, x) for x in image_filenames]
        print(f"selected image paths: {img_paths}")
        images = [cv2.imread(x, cv2.IMREAD_COLOR) for x in img_paths]

    if len(images) == 0:
        Printer.red("No images found!")
        sys.exit(1)

    Printer.green(f"Loaded {len(images)} images")

    # --------------------------------
    # Prepare factory arguments
    factory_kwargs = {}
    if args.device is not None:
        factory_kwargs["device"] = args.device

    # Add model-specific arguments
    if args.model_type is not None:
        factory_kwargs["model_type"] = args.model_type

    if args.inference_size is not None:
        factory_kwargs["inference_size"] = args.inference_size

    if args.min_conf_thr is not None:
        factory_kwargs["min_conf_thr"] = args.min_conf_thr

    # --------------------------------
    # Create reconstructor using factory
    Printer.blue("Creating scene reconstructor...")
    try:
        reconstructor = scene_from_views_factory(
            scene_from_views_type=method,
            **factory_kwargs,
        )
        Printer.green("Scene reconstructor created successfully")
    except Exception as e:
        Printer.red(f"Failed to create reconstructor: {e}")
        Printer.red("Make sure all required dependencies are installed")
        sys.exit(1)

    # --------------------------------
    # Run scene reconstruction
    Printer.blue("Running scene reconstruction...")
    start_time = time.time()

    try:
        # Option to skip optimizer and use initial poses (for debugging)
        skip_optimizer = getattr(args, "skip_optimizer", False)

        result = reconstructor.reconstruct(
            images=images,
            as_pointcloud=args.as_pointcloud,
            skip_optimizer=skip_optimizer,  # Skip optimization and use initial poses
        )
        elapsed_time = time.time() - start_time
        Printer.green(f"Scene reconstruction completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        Printer.red(f"Scene reconstruction failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # --------------------------------
    # Print scene reconstruction summary
    Printer.blue("=" * 60)
    Printer.blue("Scene reconstruction summary:")
    Printer.blue("=" * 60)
    Printer.blue(f"Method: {method.name}")
    Printer.blue(f"Number of images: {len(images)}")
    Printer.blue(f"Processed images: {len(result.processed_images)}")

    if result.global_point_cloud is not None:
        Printer.blue(f"Global point cloud: {len(result.global_point_cloud.vertices)} points")

    if result.global_mesh is not None:
        Printer.blue(
            f"Global mesh: {len(result.global_mesh.vertices)} vertices, {len(result.global_mesh.faces)} faces"
        )

    if result.camera_poses is not None:
        Printer.blue(
            f"Camera poses: {len([p for p in result.camera_poses if p is not None])} valid poses"
        )

    if result.depth_predictions is not None:
        Printer.blue(f"Depth maps: {len(result.depth_predictions)}")

    if result.confidences is not None:
        Printer.blue(f"Confidence maps: {len(result.confidences)}")

    Printer.blue("=" * 60)

    # --------------------------------
    # Visualize scene reconstruction results
    if not args.no_images or not args.no_3d:
        Printer.blue("Visualizing scene reconstruction results...")
        visualize_results(
            result,
            show_images=not args.no_images,
            show_3d=not args.no_3d,
            # original_images=images,
            method=method,
        )

    Printer.green("Scene reconstruction completed!")


if __name__ == "__main__":
    main()
