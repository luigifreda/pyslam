# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from tkinter import image_names 
sys.path.append("../../")
import pyslam.config as config
config.cfg.set_lib('vggt') 

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

import trimesh
import matplotlib
from scipy.spatial.transform import Rotation
import cv2
import os

from visual_util import segment_sky
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from pyslam.utilities.utils_files import select_image_files 
from pyslam.utilities.utils_sys import download_file_from_url
from pyslam.utilities.utils_dust3r import convert_mv_output_to_geometry
from pyslam.utilities.utils_img import img_from_floats, ImageTable
from pyslam.utilities.utils_torch import to_numpy

from pyslam.viz.viewer3D import Viewer3D, VizPointCloud, VizMesh, VizCameraImage

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kVggtFolder = kRootFolder + '/thirdparty/vggt'
kResultsFolder = kRootFolder + '/results/vggt'


# Euroc
# images_path = '/home/luigi/Work/datasets/rgbd_datasets/euroc/V101/mav0/cam0/data'
# start_frame_name = '1403715273362142976.png'
# gl_reverse_rgb = True

# TUM room (PAY ATTENTION there is distortion here!)
# images_path = '/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg1_room/rgb'
# start_frame_name = '1305031910.765238.png'
# gl_reverse_rgb = True

# TUM desk long_office_household (no distortion here)
images_path = '/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg3_long_office_household/rgb'
start_frame_name = '1341847980.722988.png'
gl_reverse_rgb = True



def predictions_to_3D_data(
    predictions,
    conf_thres=50.0,
    mask_black_bg=False,
    mask_white_bg=False,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
    as_pointcloud=True,
):
    """
    Extracts 3D point cloud or mesh and associated data from VGGT predictions.

    Returns:
        tuple: (global_pc, global_mesh, images, pts3d, masks, cams2world, focals, confs)
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # Get predicted points and confidences
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        pred_world_points = predictions["world_points"]
        pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    images = predictions["images"]  # (S, H, W, 3)
    if images.ndim == 4 and images.shape[1] == 3:
        # Convert from (S, 3, H, W) to (S, H, W, 3)
        images = np.transpose(images, (0, 2, 3, 1))
        
    # Convert BGR to RGB
    images = images[..., ::-1]

    # Ensure image dtype is uint8 in [0, 255] for consistency
    # if images.dtype in [np.float32, np.float64] and images.max() <= 1.0:
    #     images = (images * 255).astype(np.uint8)
    # elif images.dtype != np.uint8:
    #     images = images.astype(np.uint8)        
        
    camera_matrices = predictions["extrinsic"]  # (S, 3, 4)
    S, H, W = pred_world_points_conf.shape

    # Optional sky masking
    if mask_sky and target_dir is not None:
        import onnxruntime
        image_dir = os.path.join(target_dir, "images")
        mask_dir = os.path.join(target_dir, "sky_masks")
        os.makedirs(mask_dir, exist_ok=True)
        image_list = sorted(os.listdir(image_dir))
        sky_mask_list = []

        if not os.path.exists("skyseg.onnx"):
            download_file_from_url(
                "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
            )
        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")

        for i, image_name in enumerate(image_list[:S]):
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)

            if os.path.exists(mask_path):
                sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                sky_mask = segment_sky(image_path, skyseg_session, mask_path)

            if sky_mask.shape != (H, W):
                sky_mask = cv2.resize(sky_mask, (W, H))
            sky_mask_list.append(sky_mask)

        sky_mask_array = np.array(sky_mask_list)
        sky_mask_binary = (sky_mask_array > 0).astype(np.float32)
        pred_world_points_conf *= sky_mask_binary

    # Per-frame confidence maps and masks
    confs = [pred_world_points_conf[i] for i in range(S)]
    masks = [c > np.percentile(c, conf_thres) for c in confs]

    # Optional background masking
    if mask_black_bg:
        black_mask = [img.sum(axis=-1) >= 16 for img in images]
        masks = [m & b for m, b in zip(masks, black_mask)]

    if mask_white_bg:
        white_mask = [~((img[..., 0] > 240) & (img[..., 1] > 240) & (img[..., 2] > 240)) for img in images]
        masks = [m & w for m, w in zip(masks, white_mask)]

    # Normalize mask shapes to (H, W)
    normalized_masks = []
    for i in range(S):
        img_h, img_w = images[i].shape[:2]
        m = masks[i]
        if m.ndim == 1:
            if m.size != img_h * img_w:
                raise ValueError(f"Cannot reshape flat mask of size {m.size} to match image {img_h}x{img_w}")
            m = m.reshape((img_h, img_w))
        elif m.shape != (img_h, img_w):
            raise ValueError(f"Mask shape {m.shape} does not match image shape {(img_h, img_w)}")
        normalized_masks.append(m)

    # Convert to point cloud or mesh
    global_pc, global_mesh = convert_mv_output_to_geometry(
        imgs=images,
        pts3d=pred_world_points,
        mask=normalized_masks,
        as_pointcloud=as_pointcloud
    )

    # Camera extrinsics (S, 4, 4)
    cams2world = np.zeros((S, 4, 4))
    cams2world[:, :3, :4] = camera_matrices
    cams2world[:, 3, 3] = 1.0

    # Dummy intrinsics if not available
    focals = predictions.get("focals", np.ones((S, 2)))
    print(f'Focals: {focals}')

    return global_pc, global_mesh, images, pred_world_points, normalized_masks, cams2world, focals, confs



def load_and_preprocess_images_cv2(image_path_list, mode="crop", target_size=518):
    """
    Load and preprocess images using OpenCV to match the PIL-based preprocessing behavior.

    Args:
        image_path_list (list): List of image file paths.
        mode (str): "crop" or "pad".
        target_size (int): Target size for resizing/padding.

    Returns:
        torch.Tensor: Batched tensor of shape (N, 3, H, W), float32 in [0, 1].
    """
    if not image_path_list:
        raise ValueError("At least one image is required.")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'.")

    processed_images = []
    shapes = []

    for image_path in image_path_list:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise IOError(f"Failed to read image: {image_path}")

        # Handle alpha channel
        if img.shape[-1] == 4:
            b, g, r, a = cv2.split(img)
            alpha = a.astype(np.float32) / 255.0
            rgb = cv2.merge((r, g, b)).astype(np.float32) / 255.0
            white = np.ones_like(rgb)
            img = alpha[..., None] * rgb + (1 - alpha[..., None]) * white
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w = img.shape[:2]

        # Resize logic
        if mode == "pad":
            if w >= h:
                new_w = target_size
                new_h = round(h * (new_w / w) / 14) * 14
            else:
                new_h = target_size
                new_w = round(w * (new_h / h) / 14) * 14
        else:  # crop mode
            new_w = target_size
            new_h = round(h * (new_w / w) / 14) * 14

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Crop mode: center crop height to target_size if needed
        if mode == "crop" and new_h > target_size:
            start_y = (new_h - target_size) // 2
            img = img[start_y : start_y + target_size, :, :]

        # Pad mode: pad to square target_size x target_size
        if mode == "pad":
            pad_h = target_size - img.shape[0]
            pad_w = target_size - img.shape[1]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            img = cv2.copyMakeBorder(
                img,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(1.0, 1.0, 1.0),  # white
            )

        shapes.append(img.shape[:2])
        processed_images.append(img)

    # Ensure consistent shapes: pad if necessary
    if len(set(shapes)) > 1:
        print(f"Warning: inconsistent image sizes {set(shapes)} — padding to largest shape.")
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded_images = []

        for img in processed_images:
            h, w = img.shape[:2]
            pad_top = (max_h - h) // 2
            pad_bottom = max_h - h - pad_top
            pad_left = (max_w - w) // 2
            pad_right = max_w - w - pad_left
            img = cv2.copyMakeBorder(
                img,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(1.0, 1.0, 1.0),
            )
            padded_images.append(img)

        processed_images = padded_images

    # Final stack: (N, 3, H, W)
    images = np.stack([img.transpose(2, 0, 1) for img in processed_images])  # HWC → CHW
    images = torch.from_numpy(images).float()

    return images


def run_model_and_build3d(
    image_paths,
    target_dir,
    model,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    if not image_paths:
        raise RuntimeError("No image paths provided.")

    print("Starting 3D reconstruction from images in", image_paths)
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    try:
        print("Running model...")
        images = load_and_preprocess_images_cv2(image_paths, mode="pad")
        #images = load_and_preprocess_images(image_names)
        print(f"Preprocessed images shape: {images.shape}")

        images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Running inference...")
        predictions = run_model(images, model)
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")

    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    global_pc, global_mesh, rgb_imgs, pts3d, mask, cams2world, focals, confs = predictions_to_3D_data(
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
        as_pointcloud=True,
    )

    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    print(f"Reconstruction Success ({len(image_paths)} frames). Waiting for visualization.")

    return global_pc, global_mesh, rgb_imgs, pts3d, mask, cams2world, focals, confs


# -------------------------------------------------------------------------
# Core model inference
# -------------------------------------------------------------------------
def run_model(img_paths, model) -> dict:
    """
    Run the VGGT model on images in the 'img_paths' list and return predictions.
    """
    print(f"Processing images from {img_paths}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = sorted(img_paths)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    #images = load_and_preprocess_images(image_names).to(device)
    images = load_and_preprocess_images_cv2(image_names, mode="pad", target_size=518).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print(f'Intrinsic: {intrinsic}')

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions

# -------------------------------------------------------------------------
def get_reconstructed_scene(
    img_paths,
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform 3D reconstruction from a directory of images.

    Args:
        img_paths (list): List of image file paths.
        target_dir (str): Directory to store intermediate outputs.
        conf_thres (float): Confidence threshold for filtering.
        frame_filter (str): Which frames to process ("All" or "idx: filename").
        mask_black_bg (bool): Whether to mask out black background.
        mask_white_bg (bool): Whether to mask out white background.
        show_cam (bool): Whether to render camera models.
        mask_sky (bool): Whether to apply sky segmentation masks.
        prediction_mode (str): Prediction mode name.

    Returns:
        tuple: (global_pc, global_mesh, rgb_imgs, pts3d, masks, cams2world, focals, confs)
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'Starting 3D reconstruction from images in {img_paths}')

    print("Running model...")
    try:
        with torch.no_grad():
            predictions = run_model(img_paths, model)
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")

    # Save predictions for later inspection
    os.makedirs(target_dir, exist_ok=True)
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Ensure default frame filter
    frame_filter = frame_filter or "All"

    # Convert predictions to 3D structures
    global_pc, global_mesh, rgb_imgs, pts3d, mask, cams2world, focals, confs = predictions_to_3D_data(
        predictions=predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
        as_pointcloud=True,
    )

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"3D reconstruction completed in {end_time - start_time:.2f}s.")
    print(f"Processed {len(img_paths)} images from '{img_paths}'.")

    return global_pc, global_mesh, rgb_imgs, pts3d, mask, cams2world, focals, confs



if __name__ == "__main__":
    
    image_filenames = select_image_files(images_path, start_frame_name, n_frame=2, delta_frame=50)
    print(f'selected image files: {image_filenames}')
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(kResultsFolder, f"input_images_{timestamp}")
            
    img_paths = [os.path.join(images_path, x) for x in image_filenames]
    imgs = [cv2.imread(x) for x in img_paths]   
    
    global_pc, global_mesh, rgb_imgs, pts3d, mask, cams2world, focals, confs = \
        get_reconstructed_scene(img_paths, target_dir, conf_thres=3.0, frame_filter="All", mask_black_bg=False, mask_white_bg=False, show_cam=True, mask_sky=False, prediction_mode="Pointmap Regression")
    
    for i,img in enumerate(rgb_imgs):
            rgb_imgs[i] = to_numpy(img)
            confs[i] = img_from_floats(to_numpy(confs[i]))
            print(f'conf {i}: {confs[i].shape}, {confs[i].dtype}')
        
    print(f'done extracting 3d model from inference')      
    
    viewer3D = Viewer3D()
    time.sleep(1)
    
    viz_point_cloud = VizPointCloud(points=global_pc.vertices, colors=global_pc.colors, normalize_colors=True, reverse_colors=True) if global_pc is not None else None
    viz_mesh = VizMesh(vertices=global_mesh.vertices, triangles=global_mesh.faces, vertex_colors=global_mesh.visual.vertex_colors, normalize_colors=True) if global_mesh is not None else None
    viz_camera_images = []
    for i, img in enumerate(rgb_imgs):
        img_char = (img*255).astype(np.uint8)    
        if gl_reverse_rgb:
            img_char = cv2.cvtColor(img_char, cv2.COLOR_RGB2BGR)
        # ensure it's contiguous for OpenGL
        img_char = np.ascontiguousarray(img_char)
        is_contiguous = img_char.flags['C_CONTIGUOUS']
        h_ratio=img_char.shape[0] / img_char.shape[1]
        print(f'image {i}, shape {img_char.shape}, h_ratio: {h_ratio}, type: {img_char.dtype}, min {np.min(img_char)}, max {np.max(img_char)}, is contiguous: {is_contiguous}')
        viz_camera_images.append(VizCameraImage(image=img_char, Twc=cams2world[i], h_ratio=h_ratio, scale=0.1))
    viewer3D.draw_dense_geometry(point_cloud=viz_point_cloud, mesh=viz_mesh, camera_images=viz_camera_images)

    show_image_tables = True
    table_resize_scale=0.8    
    if show_image_tables:
        img_table_originals = ImageTable(num_columns=4, resize_scale=table_resize_scale)
        for i, img in enumerate(imgs):
            img_table_originals.add(img)
        img_table_originals.render()
        cv2.imshow('Original Images', img_table_originals.image())
        
        img_table = ImageTable(num_columns=4, resize_scale=table_resize_scale)
        for i, img in enumerate(rgb_imgs):
            img_table.add(img)
        img_table.render()
        cv2.imshow('VGTT Images', img_table.image())
        
        # img_inverted_table = ImageTable(num_columns=4, resize_scale=table_resize_scale)
        # for i, img in enumerate(inverted_images):
        #     img_inverted_table.add(img)
        # img_inverted_table.render()
        # cv2.imshow('Inverted Images', img_inverted_table.image())
        
        img_table_conf = ImageTable(num_columns=4, resize_scale=table_resize_scale)
        for i, conf in enumerate(confs):
            img_table_conf.add(conf)
        img_table_conf.render()
        cv2.imshow('Confidence Images', img_table_conf.image())


    while viewer3D.is_running():
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            break    
        
    viewer3D.quit()