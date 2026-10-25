import sys

from pyslam.config import Config

config = Config()
config.set_lib("mast3r")

import os
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import pts3d_to_trimesh, cat_meshes

import cv2
import numpy as np
import torch

from pyslam.utilities.drawing import draw_feature_matches, combine_images_horizontally
from pyslam.utilities.img_management import img_from_floats
from pyslam.utilities.dust3r import (
    Dust3rImagePreprocessor,
    convert_mv_output_to_geometry,
    estimate_focal_knowing_depth,
    calibrate_camera_pnpransac,
)
from pyslam.utilities.depth import point_cloud_to_depth
from pyslam.utilities.logging import Printer
from pyslam.utilities.img_management import ImgWriter
from pyslam.utilities.geom_lie import so3_log_angle

from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType
from pyslam.io.ground_truth import groundtruth_factory

from pyslam.viz.localization_plot_drawer import LocalizationPlotDrawer

from pyslam.viz.viewer3D import (
    Viewer3D,
    VizPointCloud,
    VizMesh,
    VizCameraImage,
    Viewer3DMapInput,
    MapStateData,
)

import time


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kMast3rFolder = kRootFolder + "/thirdparty/mast3r"
kResultsFolder = kRootFolder + "/results/mast3r"

model_name = kMast3rFolder + "/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
device = "cuda"


class Mast3rOutput:
    def __init__(self):
        self.cams2world = None
        self.focals = None
        self.confs = None
        self.rgb_imgs = None
        self.pts3d = None
        self.global_pc = None
        self.global_mesh = None
        self.matches_im0 = None
        self.matches_im1 = None


class OdometryState:
    def __init__(self):
        self.init_pose_id = None
        self.prev_pose_id = None
        self.cur_pose = None
        self.cur_pose_timestamp = None
        self.cur_pose_id = None
        self.ref_pose = None
        self.ref_pose_timestamp = None
        self.ref_pose_id = None
        self.poses = {}  # pose_id -> (pose, timestamp)


def mast3r_processing(imgs_preproc, model, device, min_conf_thr):

    # get inference output
    output = inference([tuple(imgs_preproc)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]

    # extract descriptors
    desc1, desc2 = pred1["desc"].squeeze(0).detach(), pred2["desc"].squeeze(0).detach()

    # extract rgb images
    rgb_imgs = [output["view1"]["img"]] + [output["view2"]["img"]]
    for i in range(len(rgb_imgs)):
        rgb_imgs[i] = (rgb_imgs[i] + 1) / 2
        rgb_imgs[i] = rgb_imgs[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
        rgb_imgs[i] = cv2.cvtColor(rgb_imgs[i], cv2.COLOR_RGB2BGR)

    # extract 3D points
    pts3d = [output["pred1"]["pts3d"][0]] + [output["pred2"]["pts3d_in_other_view"][0]]

    # extract predicted confidence
    conf = [output["pred1"]["conf"][0]] + [output["pred2"]["conf"][0]]
    conf_vec = torch.stack([x.reshape(-1) for x in conf])  # get a monodimensional vector
    conf_sorted = conf_vec.reshape(-1).sort()[0]
    conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
    print(f"confidence threshold: {conf_thres}")
    mask = [x >= conf_thres for x in conf]

    # estimate focals of first image
    h, w = rgb_imgs[0].shape[0:2]  # [H, W]
    conf_first = conf[0].reshape(-1)  # [bs, H * W]
    conf_first_sorted = conf_first.sort()[0]  # [bs, h * w]
    # conf_first_thres = conf_first_sorted[int(conf_first_sorted.shape[0] * 0.03)]  # here we use a different threshold
    conf_first_thres = conf_first_sorted[
        int(conf_first_sorted.shape[0] * float(min_conf_thr) * 0.01)
    ]
    valid_first = conf_first_sorted >= conf_first_thres  # & valids[0].reshape(bs, -1)
    valid_first = valid_first.reshape(h, w)

    focals = (
        estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda()).cpu().item()
    )

    intrinsics = torch.eye(
        3,
    )
    intrinsics[0, 0] = focals
    intrinsics[1, 1] = focals
    intrinsics[0, 2] = w / 2
    intrinsics[1, 2] = h / 2
    intrinsics = intrinsics.cuda()

    # estimate camera poses
    y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda()  # [H, W, 2]
    c2ws = []
    for pr_pt, valid in zip(pts3d, mask):
        c2ws_i = calibrate_camera_pnpransac(
            pr_pt.cuda().flatten(0, 1)[None],
            pixel_coords.flatten(0, 1)[None],
            valid.cuda().flatten(0, 1)[None],
            intrinsics[None],
        )
        c2ws.append(c2ws_i[0])
    cams2world = torch.stack(c2ws, dim=0).cpu()  # [N, 4, 4]

    # convert extracted data to numpy
    cams2world = to_numpy(cams2world)
    focals = to_numpy(focals)
    mask = [to_numpy(x) for x in mask]
    confs = [to_numpy(x) for x in conf]
    rgb_imgs = [to_numpy(x) for x in rgb_imgs]
    pts3d = to_numpy(pts3d)

    # extract the point cloud or mesh
    as_pointcloud = True
    global_pc, global_mesh = convert_mv_output_to_geometry(rgb_imgs, pts3d, mask, as_pointcloud)

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2, subsample_or_initxy1=8, device=device, dist="dot", block_size=2**13
    )

    mast3r_output = Mast3rOutput()
    mast3r_output.cams2world = cams2world
    mast3r_output.focals = focals
    mast3r_output.confs = confs
    mast3r_output.rgb_imgs = [cv2.cvtColor(img * 255.0, cv2.COLOR_BGR2RGB) for img in rgb_imgs]
    mast3r_output.pts3d = pts3d
    mast3r_output.global_pc = global_pc
    mast3r_output.global_mesh = global_mesh
    mast3r_output.matches_im0 = matches_im0
    mast3r_output.matches_im1 = matches_im1
    return mast3r_output


def viz_matches(mast3r_output: Mast3rOutput, n_viz_percent=50):

    rgb0 = mast3r_output.rgb_imgs[0].copy()
    rgb1 = mast3r_output.rgb_imgs[1].copy()
    matches_im0 = mast3r_output.matches_im0
    matches_im1 = mast3r_output.matches_im1

    # ignore small border around the edge
    # H0, W0 = view1['true_shape'][0]
    H0, W0 = rgb0.shape[0:2]
    valid_matches_im0 = (
        (matches_im0[:, 0] >= 3)
        & (matches_im0[:, 0] < int(W0) - 3)
        & (matches_im0[:, 1] >= 3)
        & (matches_im0[:, 1] < int(H0) - 3)
    )

    # H1, W1 = view2['true_shape'][0]
    H1, W1 = rgb1.shape[0:2]
    valid_matches_im1 = (
        (matches_im1[:, 0] >= 3)
        & (matches_im1[:, 0] < int(W1) - 3)
        & (matches_im1[:, 1] >= 3)
        & (matches_im1[:, 1] < int(H1) - 3)
    )

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # visualize a few matches
    # n_viz_percent is the percentage of shown matches
    num_matches = matches_im0.shape[0]
    n_viz = int(100 / n_viz_percent)
    match_idx_to_viz = np.arange(0, num_matches - 1, n_viz)  # extract 1 sample every n_viz samples
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    out_img = draw_feature_matches(rgb0, rgb1, viz_matches_im0, viz_matches_im1, horizontal=True)
    return out_img


def update_and_draw_map(
    cur_timestamp,
    cur_frame_id,
    ref_frame_id,
    mast3r_output: Mast3rOutput,
    odometry_state: OdometryState,
    viewer3D: Viewer3D,
    viz_mast3r_map_state: Viewer3DMapInput,
):

    if ref_frame_id != odometry_state.init_pose_id and odometry_state.ref_pose_id != ref_frame_id:
        print(f"Updating reference frame id: {ref_frame_id}")
        ref_pose, ref_timestamp = odometry_state.poses[ref_frame_id]
        odometry_state.ref_pose = ref_pose
        odometry_state.ref_pose_timestamp = ref_timestamp
        odometry_state.ref_pose_id = ref_frame_id

    odometry_state.prev_pose_id = odometry_state.cur_pose_id
    odometry_state.cur_pose = (
        odometry_state.ref_pose @ mast3r_output.cams2world[1]
        if odometry_state.ref_pose is not None
        else mast3r_output.cams2world[1]
    )
    odometry_state.cur_pose_timestamp = cur_timestamp
    odometry_state.cur_pose_id = cur_frame_id
    odometry_state.poses[cur_frame_id] = (odometry_state.cur_pose, cur_timestamp)

    viz_mast3r_map_state.cur_pose = odometry_state.cur_pose
    viz_mast3r_map_state.cur_pose_timestamp = cur_timestamp
    viz_mast3r_map_state.cur_frame_id = cur_frame_id

    # Initialize map_data only if it doesn't exist
    if viz_mast3r_map_state.map_data is None:
        viz_mast3r_map_state.map_data = MapStateData()

        # Initialize spanning tree with first two poses
        print(f"Initializing viz map_data with first two poses")
        if (
            odometry_state.prev_pose_id is not None
            and odometry_state.prev_pose_id in odometry_state.poses
        ):
            prev_Ow = odometry_state.poses[odometry_state.prev_pose_id][0][:3, 3]
            cur_Ow = odometry_state.cur_pose[:3, 3]
            viz_mast3r_map_state.map_data.spanning_tree.append([*prev_Ow, *cur_Ow])
        else:
            # First frame - use reference and current pose
            ref_Ow = mast3r_output.cams2world[0][:3, 3]
            cur_Ow = mast3r_output.cams2world[1][:3, 3]
            viz_mast3r_map_state.map_data.spanning_tree.append([*ref_Ow, *cur_Ow])
    else:
        # Append to spanning tree
        if (
            odometry_state.prev_pose_id is not None
            and odometry_state.prev_pose_id in odometry_state.poses
        ):
            last_Ow = odometry_state.poses[odometry_state.prev_pose_id][0][:3, 3]
            cur_Ow = odometry_state.cur_pose[:3, 3]
            viz_mast3r_map_state.map_data.spanning_tree.append([*last_Ow, *cur_Ow])

    viz_mast3r_map_state.map_data.poses.append(odometry_state.cur_pose)
    viz_mast3r_map_state.map_data.pose_timestamps.append(cur_timestamp)

    print(f"Frame {cur_frame_id}: #poses {len(viz_mast3r_map_state.map_data.poses)}")

    if viewer3D.gt_trajectory is None:
        Printer.yellow(f"You did not set groundtruth in map_state")

    viewer3D.draw_map(viz_mast3r_map_state)

    global_pc = mast3r_output.global_pc
    global_mesh = mast3r_output.global_mesh
    rgb_imgs = mast3r_output.rgb_imgs

    print(f"global_pc.vertices shape: {global_pc.vertices.shape}")

    # transform vertices into reference frame
    if ref_frame_id != odometry_state.init_pose_id:
        ref_pose = odometry_state.ref_pose
        if global_pc is not None and global_pc.vertices is not None:
            global_pc.vertices = (
                ref_pose[:3, :3] @ global_pc.vertices.T + ref_pose[:3, 3].reshape(3, 1)
            ).T
        if global_mesh is not None and global_mesh.vertices is not None:
            global_mesh.vertices = (
                ref_pose[:3, :3] @ global_mesh.vertices.T + ref_pose[:3, 3].reshape(3, 1)
            ).T

    viz_point_cloud = (
        VizPointCloud(
            points=global_pc.vertices,
            colors=global_pc.colors,
            normalize_colors=True,
            reverse_colors=False,
        )
        if global_pc is not None
        else None
    )
    viz_mesh = (
        VizMesh(
            vertices=global_mesh.vertices,
            triangles=global_mesh.faces,
            vertex_colors=global_mesh.visual.vertex_colors,
            normalize_colors=True,
        )
        if global_mesh is not None
        else None
    )
    viz_camera_images = []
    # img_char = cv2.cvtColor(rgb_imgs[1], cv2.COLOR_RGB2BGR)
    # viz_camera_images.append(VizCameraImage(image=img_char, Twc=mast3r_output.cams2world[1], id=cur_frame_id, scale=0.1))
    viewer3D.draw_dense_geometry(
        point_cloud=viz_point_cloud, mesh=viz_mesh, camera_images=viz_camera_images
    )


def update_and_plot(
    cur_timestamp, cur_frame_id, plot_drawer: LocalizationPlotDrawer, mast3r_output: Mast3rOutput
):
    data_dict = {}
    data_dict["num_matched_kps"] = len(mast3r_output.matches_im0)
    plot_drawer.draw(cur_frame_id, data_dict)


if __name__ == "__main__":

    reference_img_id = 0
    min_conf_thr = 50  # percentage of the max confidence value
    inference_size = 512
    matches_viz_percent = 20  # percentage of shown matches
    delta_translation_for_new_ref = 0.2  # meters
    delta_angle_for_new_ref = 20  # degrees

    dataset = dataset_factory(config)
    is_monocular = dataset.sensor_type == SensorType.MONOCULAR

    groundtruth = groundtruth_factory(config.dataset_settings)

    img_writer = ImgWriter(font_scale=0.7)
    viewer3D = Viewer3D()
    time.sleep(1)

    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    dust3r_preprocessor = Dust3rImagePreprocessor(inference_size=inference_size)

    # get reference image
    reference_img = dataset.getImageColor(reference_img_id)
    reference_img_preproc = dust3r_preprocessor.preprocess_image(reference_img)

    if False:
        reference_img_show = reference_img.copy()
        img_writer.write(reference_img_show, f"reference image id: {reference_img_id}", (30, 30))
        cv2.imshow("reference_img", reference_img_show)

    if groundtruth is not None:
        print(f"groundtruth found, setting GT trajectory")
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        viewer3D.set_gt_trajectory(gt_traj3d, gt_timestamps, align_with_scale=is_monocular)

    viz_mast3r_map_state = Viewer3DMapInput()
    plot_drawer = LocalizationPlotDrawer(viewer3D)

    odometry_state = OdometryState()
    odometry_state.init_pose_id = reference_img_id

    img_id = 0  # 180, 340, 400   # you can start from a desired frame id if needed
    while viewer3D.is_running():

        timestamp, img = None, None

        if dataset.is_ok:
            timestamp = dataset.getTimestamp()  # get current timestamp
            img = dataset.getImageColor(img_id)

        if img is not None:
            print("----------------------------------------")
            print(f"processing img {img_id}, reference img {reference_img_id}")

            img_preproc = dust3r_preprocessor.preprocess_image(img)

            imgs_preproc = [reference_img_preproc, img_preproc]
            mast3r_output = mast3r_processing(
                imgs_preproc, model, device, min_conf_thr=min_conf_thr
            )

            # viz current image
            img_writer.write(img, f"id: {img_id}", (30, 30))
            cv2.imshow("img", img)

            # viz matches
            matches_img = viz_matches(mast3r_output, matches_viz_percent)
            cv2.imshow("matches", matches_img)

            # viz map
            update_and_draw_map(
                timestamp,
                img_id,
                reference_img_id,
                mast3r_output,
                odometry_state,
                viewer3D,
                viz_mast3r_map_state,
            )

            # viz plots
            update_and_plot(timestamp, img_id, plot_drawer, mast3r_output)

            # update reference image strategy
            delta_t_to_ref = np.linalg.norm(mast3r_output.cams2world[1][:3, 3])
            delta_angle_to_ref = so3_log_angle(mast3r_output.cams2world[1][:3, :3])
            print(f"delta_t: {delta_t_to_ref}, delta_angle: {delta_angle_to_ref}")
            if (
                delta_t_to_ref > delta_translation_for_new_ref
                or delta_angle_to_ref > delta_angle_for_new_ref
            ):
                reference_img_id = img_id
                reference_img = img
                reference_img_preproc = img_preproc

            cv2.waitKey(1) & 0xFF
        else:
            cv2.waitKey(100) & 0xFF

        img_id += 1

    if plot_drawer is not None:
        plot_drawer.quit()
    if viewer3D is not None:
        viewer3D.quit()
