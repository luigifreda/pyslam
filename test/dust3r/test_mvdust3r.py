#!/usr/bin/env python3

import sys

import pyslam.config as config

config.cfg.set_lib("mvdust3r")

import cv2
import argparse
import copy
import os
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms as tvf
import trimesh
from scipy.spatial.transform import Rotation
import time

from pyslam.viz.viewer3D import Viewer3D, VizPointCloud, VizMesh, VizCameraImage

from pyslam.utilities.file_management import select_image_files
from pyslam.utilities.dust3r import convert_mv_output_to_geometry, Dust3rImagePreprocessor
from pyslam.utilities.img_management import img_from_floats, ImageTable

inf = np.inf

import sys

# from dust3r.dummy_io import *
os.environ["meta_internal"] = "False"

import matplotlib.pyplot as pl
from mvdust3r.dust3r.inference import inference, inference_mv
from mvdust3r.dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from mvdust3r.dust3r.model import AsymmetricCroCo3DStereoMultiView
from mvdust3r.dust3r.utils.device import to_numpy

from mvdust3r.dust3r.utils.image import load_images, rgb
from mvdust3r.dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, OPENGL, pts3d_to_trimesh

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kMvdust3rFolder = kRootFolder + "/thirdparty/mvdust3r"
kResultsFolder = kRootFolder + "/results/mvdust3r"


# Euroc
# images_path = '/home/luigi/Work/datasets/rgbd_datasets/euroc/V101/mav0/cam0/data'
# start_frame_name = '1403715273362142976.png'
# gl_reverse_rgb = False

# TUM room (PAY ATTENTION there is distortion here!)
# images_path = '/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg1_room/rgb'
# start_frame_name = '1305031910.765238.png'
# gl_reverse_rgb = False

# TUM desk long_office_household (no distortion here)
images_path = (
    "/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg3_long_office_household/rgb"
)
start_frame_name = "1341847980.722988.png"
gl_reverse_rgb = True


# # Note: to visualize glb file: https://glb.ee/
# def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
#                                  cam_color=None, as_pointcloud=False,
#                                  transparent_cams=False, silent=False):
#     assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
#     pts3d = to_numpy(pts3d)
#     imgs = to_numpy(imgs)
#     focals = to_numpy(focals)
#     cams2world = to_numpy(cams2world)

#     scene = trimesh.Scene()

#     # full pointcloud
#     if as_pointcloud:
#         pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
#         col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
#         pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
#         scene.add_geometry(pct)
#     else:
#         meshes = []
#         for i in range(len(imgs)):
#             meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
#         mesh = trimesh.Trimesh(**cat_meshes(meshes))
#         scene.add_geometry(mesh)

#     # add each camera
#     for i, pose_c2w in enumerate(cams2world):
#         if isinstance(cam_color, list):
#             camera_edge_color = cam_color[i]
#         else:
#             camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
#         add_scene_cam(scene, pose_c2w, camera_edge_color,
#                       None if transparent_cams else imgs[i], focals[i],
#                       imsize=imgs[i].shape[1::-1], screen_width=cam_size)

#     rot = np.eye(4)
#     rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
#     scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
#     outfile = os.path.join(outdir, 'scene.glb')
#     if not silent:
#         print('(exporting 3D scene to', outfile, ')')
#     scene.export(file_obj=outfile)
#     return outfile


# def get_3D_model_from_scene(outdir, silent, output, min_conf_thr=3, as_pointcloud=False, transparent_cams=False, cam_size=0.05, only_model=False):
#     """
#     extract 3D_model (glb file) from a reconstructed scene
#     """

#     with torch.no_grad():

#         _, h, w = output['pred1']['rgb'].shape[0:3] # [1, H, W, 3]
#         rgbimg = [output['pred1']['rgb'][0]] + [x['rgb'][0] for x in output['pred2s']]
#         for i in range(len(rgbimg)):
#             rgbimg[i] = (rgbimg[i] + 1) / 2
#         pts3d = [output['pred1']['pts3d'][0]] + [x['pts3d_in_other_view'][0] for x in output['pred2s']]
#         conf = torch.stack([output['pred1']['conf'][0]] + [x['conf'][0] for x in output['pred2s']], 0) # [N, H, W]
#         conf_sorted = conf.reshape(-1).sort()[0]
#         conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
#         msk = conf >= conf_thres

#         # calculate focus:

#         conf_first = conf[0].reshape(-1) # [bs, H * W]
#         conf_sorted = conf_first.sort()[0] # [bs, h * w]
#         conf_thres = conf_sorted[int(conf_first.shape[0] * 0.03)]
#         valid_first = (conf_first >= conf_thres) # & valids[0].reshape(bs, -1)
#         valid_first = valid_first.reshape(h, w)

#         focals = estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda()).cpu().item()

#         intrinsics = torch.eye(3,)
#         intrinsics[0, 0] = focals
#         intrinsics[1, 1] = focals
#         intrinsics[0, 2] = w / 2
#         intrinsics[1, 2] = h / 2
#         intrinsics = intrinsics.cuda()

#         focals = torch.Tensor([focals]).reshape(1,).repeat(len(rgbimg))

#         y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
#         pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda() # [H, W, 2]

#         c2ws = []
#         for (pr_pt, valid) in zip(pts3d, msk):
#             c2ws_i = calibrate_camera_pnpransac(pr_pt.cuda().flatten(0,1)[None], pixel_coords.flatten(0,1)[None], valid.cuda().flatten(0,1)[None], intrinsics[None])
#             c2ws.append(c2ws_i[0])

#         cams2world = torch.stack(c2ws, dim=0).cpu() # [N, 4, 4]
#         focals = to_numpy(focals)

#         pts3d = to_numpy(pts3d)
#         msk = to_numpy(msk)

#     glb_file = _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)
#     conf = to_numpy([x[0] for x in conf.split(1, dim=0)])
#     rgbimg = to_numpy(rgbimg)
#     if only_model:
#         return glb_file
#     return glb_file, rgbimg, conf


# def get_reconstructed_scene(outdir, model, device, imgs, min_conf_thr, as_pointcloud, transparent_cams, cam_size, silent):
#     """
#     from a list of images, run dust3r inference, global aligner.
#     then run get_3D_model_from_scene
#     """
#     #imgs = load_images(filelist, size=image_size, verbose=not silent, n_frame = n_frame)
#     if len(imgs) == 1:
#         imgs = [imgs[0], copy.deepcopy(imgs[0])]
#         imgs[1]['idx'] = 1
#     for img in imgs:
#         img['true_shape'] = torch.from_numpy(img['true_shape']).long()

#     if len(imgs) < 12:
#         if len(imgs) > 3:
#             imgs[1], imgs[3] = deepcopy(imgs[3]), deepcopy(imgs[1])
#         if len(imgs) > 6:
#             imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])
#     else:
#         change_id = len(imgs) // 4 + 1
#         imgs[1], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[1])
#         change_id = (len(imgs) * 2) // 4 + 1
#         imgs[2], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[2])
#         change_id = (len(imgs) * 3) // 4 + 1
#         imgs[3], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[3])

#     output = inference_mv(imgs, model, device, verbose=not silent)

#     # print(output['pred1']['rgb'].shape, imgs[0]['img'].shape, 'aha')
#     output['pred1']['rgb'] = imgs[0]['img'].permute(0,2,3,1)
#     for x, img in zip(output['pred2s'], imgs[1:]):
#         x['rgb'] = img['img'].permute(0,2,3,1)

#     outfile, rgbimg, confs = get_3D_model_from_scene(outdir, silent, output, min_conf_thr, as_pointcloud, transparent_cams, cam_size)

#     # also return rgb, depth and confidence imgs
#     # depth is normalized with the max value for all images
#     # we apply the jet colormap on the confidence maps
#     # rgbimg = scene.imgs
#     # depths = to_numpy(scene.get_depthmaps())
#     # confs = to_numpy([c for c in scene.im_conf])
#     cmap = pl.get_cmap('jet')
#     # depths_max = max([d.max() for d in depths])
#     # depths = [d/depths_max for d in depths]
#     confs_max = max([d.max() for d in confs])
#     confs = [cmap(d/confs_max) for d in confs]

#     imgs = []
#     for i in range(len(rgbimg)):
#         imgs.append(rgbimg[i])
#         # imgs.append(rgb(depths[i]))
#         imgs.append(rgb(confs[i]))

#     return output, outfile, imgs


def infer_mv_scene(model, device, imgs, silent):
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
    for img in imgs:
        img["true_shape"] = torch.from_numpy(img["true_shape"]).long()

    if len(imgs) < 12:
        if len(imgs) > 3:
            imgs[1], imgs[3] = deepcopy(imgs[3]), deepcopy(imgs[1])
        if len(imgs) > 6:
            imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])
    else:
        change_id = len(imgs) // 4 + 1
        imgs[1], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[1])
        change_id = (len(imgs) * 2) // 4 + 1
        imgs[2], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[2])
        change_id = (len(imgs) * 3) // 4 + 1
        imgs[3], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[3])

    output = inference_mv(imgs, model, device, verbose=not silent)

    # print(output['pred1']['rgb'].shape, imgs[0]['img'].shape, 'aha')
    output["pred1"]["rgb"] = imgs[0]["img"].permute(0, 2, 3, 1)
    for x, img in zip(output["pred2s"], imgs[1:]):
        x["rgb"] = img["img"].permute(0, 2, 3, 1)

    return output


# def convert_mv_output_to_geometry(imgs, pts3d, mask, as_pointcloud): #, focals, cams2world):
#     assert len(pts3d) == len(mask) <= len(imgs) # <= len(cams2world) == len(focals)
#     pts3d = to_numpy(pts3d)
#     imgs = to_numpy(imgs)
#     #focals = to_numpy(focals)
#     #cams2world = to_numpy(cams2world)

#     #scene = trimesh.Scene()
#     global_pc = None
#     global_mesh = None

#     # full pointcloud
#     if as_pointcloud:
#         pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
#         col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
#         global_pc = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
#         #scene.add_geometry(global_pc)
#     else:
#         meshes = []
#         for i in range(len(imgs)):
#             meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
#         global_mesh = trimesh.Trimesh(**cat_meshes(meshes))
#         #scene.add_geometry(global_mesh)

#     # add each camera
#     # for i, pose_c2w in enumerate(cams2world):
#     #     if isinstance(cam_color, list):
#     #         camera_edge_color = cam_color[i]
#     #     else:
#     #         camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
#     #     add_scene_cam(scene, pose_c2w, camera_edge_color,
#     #                   None if transparent_cams else imgs[i], focals[i],
#     #                   imsize=imgs[i].shape[1::-1], screen_width=cam_size)

#     #rot = np.eye(4)
#     #rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
#     #scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

#     return global_pc, global_mesh


def get_3D_dense_map(
    output, min_conf_thr=3, as_pointcloud=True
):  # , transparent_cams=False, cam_size=0.05, only_model=False):
    """
    extract 3D_model from a reconstructed scene
    """
    with torch.no_grad():

        _, h, w = output["pred1"]["rgb"].shape[0:3]  # [1, H, W, 3]
        rgb_imgs = [output["pred1"]["rgb"][0]] + [x["rgb"][0] for x in output["pred2s"]]
        for i in range(len(rgb_imgs)):
            rgb_imgs[i] = (rgb_imgs[i] + 1) / 2
        pts3d = [output["pred1"]["pts3d"][0]] + [
            x["pts3d_in_other_view"][0] for x in output["pred2s"]
        ]
        conf = torch.stack(
            [output["pred1"]["conf"][0]] + [x["conf"][0] for x in output["pred2s"]], 0
        )  # [N, H, W]
        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
        msk = conf >= conf_thres

        # calculate focus:

        conf_first = conf[0].reshape(-1)  # [bs, H * W]
        conf_sorted = conf_first.sort()[0]  # [bs, h * w]
        conf_thres = conf_sorted[int(conf_first.shape[0] * 0.03)]
        valid_first = conf_first >= conf_thres  # & valids[0].reshape(bs, -1)
        valid_first = valid_first.reshape(h, w)

        focals = (
            estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda())
            .cpu()
            .item()
        )

        intrinsics = torch.eye(
            3,
        )
        intrinsics[0, 0] = focals
        intrinsics[1, 1] = focals
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        intrinsics = intrinsics.cuda()

        focals = (
            torch.Tensor([focals])
            .reshape(
                1,
            )
            .repeat(len(rgb_imgs))
        )

        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda()  # [H, W, 2]

        c2ws = []
        for pr_pt, valid in zip(pts3d, msk):
            c2ws_i = calibrate_camera_pnpransac(
                pr_pt.cuda().flatten(0, 1)[None],
                pixel_coords.flatten(0, 1)[None],
                valid.cuda().flatten(0, 1)[None],
                intrinsics[None],
            )
            c2ws.append(c2ws_i[0])

        cams2world = torch.stack(c2ws, dim=0).cpu()  # [N, 4, 4]

        cams2world = to_numpy(cams2world)
        focals = to_numpy(focals)

        pts3d = to_numpy(pts3d)
        msk = to_numpy(msk)

    # glb_file = _convert_scene_output_to_glb(outdir, rgb_imgs, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)
    global_pc, global_mesh = convert_mv_output_to_geometry(rgb_imgs, pts3d, msk, as_pointcloud)
    confs = to_numpy([x[0] for x in conf.split(1, dim=0)])

    # try:
    #     print(f'rgb_imgs shape: {rgb_imgs.shape}')
    # except:
    #     print(f'rgb_imgs len: {len(rgb_imgs)}')

    # try:
    #     print(f'pts3d shape: {pts3d.shape}')
    # except:
    #     print(f'pts3d len: {len(pts3d)}')

    # print(f'msk shape: {msk.shape}')

    # NOTE:
    # rgb_imgs[i] is the i-th image
    # pts3d[i] is the 3D points of the i-th image
    # msk[i] is the mask of the i-th image
    # focals[i] is the focal length of the i-th image
    return global_pc, global_mesh, rgb_imgs, pts3d, msk, cams2world, focals, confs


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="image size (note, we do not train and test on other resolutions yet, this should not be changed)",
    )
    parser_weights = parser.add_mutually_exclusive_group(required=False)
    parser_weights.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default=kMvdust3rFolder + "/checkpoints/MVD.pth",
    )
    parser_weights.add_argument(
        "--model_name", type=str, help="name of the model weights", choices=["MVD", "MVDp"]
    )
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument("--silent", action="store_true", default=False, help="silence logs")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    weights_path = args.weights
    if args.model_name is None:
        if "MVDp" in args.weights:
            args.model_name = "MVDp"
        elif "MVD" in args.weights:
            args.model_name = "MVD"
        else:
            raise ValueError("model name not found in weights path")

    if not os.path.exists(weights_path):
        raise ValueError(f"weights file {weights_path} not found")

    if args.model_name == "MVD":
        model = AsymmetricCroCo3DStereoMultiView(
            pos_embed="RoPE100",
            img_size=(224, 224),
            head_type="linear",
            output_mode="pts3d",
            depth_mode=("exp", -inf, inf),
            conf_mode=("exp", 1, 1e9),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            GS=True,
            sh_degree=0,
            pts_head_config={"skip": True},
        )
        model.to(args.device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(weights_path).to(
            args.device
        )
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)
    elif args.model_name == "MVDp":
        model = AsymmetricCroCo3DStereoMultiView(
            pos_embed="RoPE100",
            img_size=(224, 224),
            head_type="linear",
            output_mode="pts3d",
            depth_mode=("exp", -inf, inf),
            conf_mode=("exp", 1, 1e9),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            GS=True,
            sh_degree=0,
            pts_head_config={"skip": True},
            m_ref_flag=True,
            n_ref=4,
        )
        model.to(args.device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(weights_path).to(
            args.device
        )
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)

    else:
        raise ValueError(f"{args.model_name} is not supported")

    image_filenames = select_image_files(images_path, start_frame_name, n_frame=10, delta_frame=30)
    print(f"selected image files: {image_filenames}")

    img_paths = [os.path.join(images_path, x) for x in image_filenames]
    imgs = [cv2.imread(x) for x in img_paths]
    if False:
        for img in imgs:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dust3r_preprocessor = Dust3rImagePreprocessor(
        inference_size=args.image_size, verbose=not args.silent
    )

    imgs_preproc = dust3r_preprocessor.preprocess_images(imgs)
    print(f"done preprocessing images")

    # adjust the confidence threshold
    min_conf_thr = 5.0  # confidence threshold (%): value=3.0, minimum=0.0, maximum=20

    as_pointcloud = True  # get point cloud or mesh

    # adjust the camera size in the output pointcloud
    # cam_size = 0.05 # camera size: value=0.05, minimum=0.001, maximum=0.5
    # transparent_cams = False

    if not os.path.exists(kResultsFolder):
        os.makedirs(kResultsFolder)

    # (outdir, model, device, imgs, min_conf_thr, as_pointcloud, transparent_cams, cam_size, silent))
    # output, outfile, imgs_output = get_reconstructed_scene(outdir=kResultsFolder, model=model, device=args.device, \
    #         imgs=imgs_preproc, min_conf_thr=min_conf_thr, as_pointcloud=as_pointcloud, \
    #         transparent_cams=transparent_cams, cam_size=cam_size, silent=args.silent)

    output = infer_mv_scene(model=model, device=args.device, imgs=imgs_preproc, silent=args.silent)
    print(f"done mv inference")

    global_pc, global_mesh, rgb_imgs, pts3d, msk, cams2world, focals, confs = get_3D_dense_map(
        output, min_conf_thr=min_conf_thr, as_pointcloud=as_pointcloud
    )
    for i, img in enumerate(rgb_imgs):
        rgb_imgs[i] = to_numpy(img)
        confs[i] = img_from_floats(to_numpy(confs[i]))

    print(f"done extracting 3d model from inference")

    viewer3D = Viewer3D()
    time.sleep(1)

    viz_point_cloud = (
        VizPointCloud(
            points=global_pc.vertices,
            colors=global_pc.colors,
            normalize_colors=True,
            reverse_colors=True,
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
    for i, img in enumerate(rgb_imgs):
        img_char = (img * 255).astype(np.uint8)
        # is_contiguous = img_char.flags['C_CONTIGUOUS']
        # print(f'image {i}, min {np.min(img_char)}, max {np.max(img_char)}, is contiguous: {is_contiguous}')
        if gl_reverse_rgb:
            img_char = cv2.cvtColor(img_char, cv2.COLOR_RGB2BGR)
        viz_camera_images.append(VizCameraImage(image=img_char, Twc=cams2world[i], scale=0.1))
    viewer3D.draw_dense_geometry(
        point_cloud=viz_point_cloud, mesh=viz_mesh, camera_images=viz_camera_images
    )

    show_image_tables = True
    table_resize_scale = 0.8
    if show_image_tables:
        img_table_originals = ImageTable(num_columns=4, resize_scale=table_resize_scale)
        for i, img in enumerate(imgs):
            img_table_originals.add(img)
        img_table_originals.render()
        cv2.imshow("Original Images", img_table_originals.image())

        img_table = ImageTable(num_columns=4, resize_scale=table_resize_scale)
        for i, img in enumerate(rgb_imgs):
            img_table.add(img)
        img_table.render()
        cv2.imshow("Dust3r Images", img_table.image())

        img_table_conf = ImageTable(num_columns=4, resize_scale=table_resize_scale)
        for i, conf in enumerate(confs):
            img_table_conf.add(conf)
        img_table_conf.render()
        cv2.imshow("Confidence Images", img_table_conf.image())

    while viewer3D.is_running():
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q") or key == 27:
            break

    viewer3D.quit()
