import sys 
sys.path.append("../../")
import pyslam.config as config
config.cfg.set_lib('mast3r') 

import os
import torch
import tempfile
import argparse

from contextlib import nullcontext

#from mast3r.demo import get_args_parser, main_demo
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5


import math
import builtins
import datetime
import gradio
import os
import torch
import numpy as np
import cv2
import functools
import trimesh
import copy
import tempfile
import shutil
from scipy.spatial.transform import Rotation

from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
#from dust3r.demo import get_args_parser as dust3r_get_args_parser

from pyslam.utilities.utils_files import select_image_files 
from pyslam.utilities.utils_dust3r import Dust3rImagePreprocessor, convert_mv_output_to_geometry
from pyslam.utilities.utils_img import img_from_floats

from pyslam.viz.viewer3D import Viewer3D, VizPointCloud, VizMesh, VizCameraImage
from pyslam.utilities.utils_img import ImageTable
import time

import matplotlib.pyplot as pl
pl.ion()


torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kMast3rFolder = kRootFolder + '/thirdparty/mast3r'
kResultsFolder = kRootFolder + '/results/mast3r'


# Euroc
# images_path = '/home/luigi/Work/datasets/rgbd_datasets/euroc/V101/mav0/cam0/data'
# start_frame_name = '1403715273362142976.png'
# gl_reverse_rgb = False

# TUM room (PAY ATTENTION there is distortion here!)
# images_path = '/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg1_room/rgb'
# start_frame_name = '1305031910.765238.png'
# gl_reverse_rgb = False

# TUM desk long_office_household (no distortion here)
images_path = '/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg3_long_office_household/rgb'
start_frame_name = '1341847980.722988.png'
gl_reverse_rgb = True



class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


# def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
#                             clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
#     """
#     extract 3D_model (glb file) from a reconstructed scene
#     """
#     if scene_state is None:
#         return None
#     outfile = scene_state.outfile_name
#     if outfile is None:
#         return None

#     # get optimized values from scene
#     scene = scene_state.sparse_ga
#     rgbimg = scene.imgs
#     focals = scene.get_focals().cpu()
#     cams2world = scene.get_im_poses().cpu()

#     # 3D pointcloud from depthmap, poses and intrinsics
#     if TSDF_thresh > 0:
#         tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
#         pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
#     else:
#         pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
#     msk = to_numpy([c > min_conf_thr for c in confs])
#     return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
#                                         transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def get_3D_dense_map(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract point cloud or mesh (3D_model) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgb_imgs = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    mask = to_numpy([c > min_conf_thr for c in confs])
    
    # return _convert_scene_output_to_glb(outfile, rgb_imgs, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
    #                                     transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)
    
    global_pc, global_mesh = convert_mv_output_to_geometry(rgb_imgs, pts3d, mask, as_pointcloud)  
    
    for i,p in enumerate(pts3d):
        print(f'pts3d[{i}].shape: {pts3d[i].shape}, mask[{i}].shape: {mask[i].shape}, confs[{i}].shape: {confs[i].shape}')
    
    confs = [to_numpy(x) for x in confs]
        
    # NOTE: 
    # rgb_imgs[i] is the i-th image
    # pts3d[i] is the 3D points of the i-th image
    # msk[i] is the mask of the i-th image
    # focals[i] is the focal length of the i-th image
    return global_pc, global_mesh, rgb_imgs, pts3d, mask, cams2world, focals, confs


# def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
#                             filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
#                             as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
#                             win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
#     """
#     from a list of images, run mast3r inference, sparse global aligner.
#     then run get_3D_model_from_scene
#     """
#     imgs = load_images(filelist, size=image_size, verbose=not silent)
#     if len(imgs) == 1:
#         imgs = [imgs[0], copy.deepcopy(imgs[0])]
#         imgs[1]['idx'] = 1
#         filelist = [filelist[0], filelist[0] + '_2']

#     scene_graph_params = [scenegraph_type]
#     if scenegraph_type in ["swin", "logwin"]:
#         scene_graph_params.append(str(winsize))
#     elif scenegraph_type == "oneref":
#         scene_graph_params.append(str(refid))
#     if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
#         scene_graph_params.append('noncyclic')
#     scene_graph = '-'.join(scene_graph_params)
#     pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
#     if optim_level == 'coarse':
#         niter2 = 0
        
#     # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
#     if current_scene_state is not None and \
#         not current_scene_state.should_delete and \
#             current_scene_state.cache_dir is not None:
#         cache_dir = current_scene_state.cache_dir
#     elif gradio_delete_cache:
#         cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
#     else:
#         cache_dir = os.path.join(outdir, 'cache')
#     os.makedirs(cache_dir, exist_ok=True)
#     scene = sparse_global_alignment(filelist, pairs, cache_dir,
#                                     model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
#                                     opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
#                                     matching_conf_thr=matching_conf_thr, **kw)
#     if current_scene_state is not None and \
#         not current_scene_state.should_delete and \
#             current_scene_state.outfile_name is not None:
#         outfile_name = current_scene_state.outfile_name
#     else:
#         outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

#     scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    
#     outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
#                                       clean_depth, transparent_cams, cam_size, TSDF_thresh)
#     return scene_state, outfile


def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, current_scene_state,
                            imgs, filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    #imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    
    print(f'starting make_pairs...')    
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
        
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f'starting sparse_global_alignment...')        
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    
    print(f'starting get_3D_dense_map...')        
    return get_3D_dense_map(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)


# def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
#     num_files = len(inputfiles) if inputfiles is not None else 1
#     show_win_controls = scenegraph_type in ["swin", "logwin"]
#     show_winsize = scenegraph_type in ["swin", "logwin"]
#     show_cyclic = scenegraph_type in ["swin", "logwin"]
#     max_winsize, min_winsize = 1, 1
#     if scenegraph_type == "swin":
#         if win_cyclic:
#             max_winsize = max(1, math.ceil((num_files - 1) / 2))
#         else:
#             max_winsize = num_files - 1
#     elif scenegraph_type == "logwin":
#         if win_cyclic:
#             half_size = math.ceil((num_files - 1) / 2)
#             max_winsize = max(1, math.ceil(math.log(half_size, 2)))
#         else:
#             max_winsize = max(1, math.ceil(math.log(num_files, 2)))
#     winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
#                             minimum=min_winsize, maximum=max_winsize, step=1, visible=show_winsize)
#     win_cyclic = gradio.Checkbox(value=win_cyclic, label="Cyclic sequence", visible=show_cyclic)
#     win_col = gradio.Column(visible=show_win_controls)
#     refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
#                           maximum=num_files - 1, step=1, visible=scenegraph_type == 'oneref')
#     return win_col, winsize, win_cyclic, refid



class Mast3rConfig:
    def __init__(self):
        # Coarse learning rate (Range: 0.01 - 0.2, Step: 0.01)
        self.lr1 = 0.07
        
        # Number of iterations for coarse alignment (Range: 0 - 10,000)
        self.niter1 = 500
        
        # Fine learning rate (Range: 0.005 - 0.05, Step: 0.001)
        self.lr2 = 0.014
        
        # Number of iterations for refinement (Range: 0 - 100,000)
        self.niter2 = 200
        
        # Optimization level (Options: 'coarse', 'refine', 'refine+depth')
        self.optim_level = 'refine+depth'
        
        # Matching confidence threshold (Range: 0.0 - 30.0, Step: 0.1)
        self.matching_conf_thr = 5.0
        
        # Shared intrinsics flag
        self.shared_intrinsics = False
        
        # Scenegraph type (Options: 'complete', 'swin', 'logwin', 'oneref')
        self.scenegraph_type = 'complete'
        
        # Scene Graph: Window Size (Range: 1 - 1, Step: 1)
        self.winsize = 1
        
        # Cyclic sequence flag
        self.win_cyclic = False
        
        # Scene Graph: Id (Range: 0 - 0, Step: 1)
        self.refid = 0
        
        # Minimum confidence threshold (Range: 0.0 - 10.0, Step: 0.1)
        self.min_conf_thr = 1.5
        
        # Camera size in output point cloud (Range: 0.001 - 1.0, Step: 0.001)
        self.cam_size = 0.2
        
        # TSDF threshold (Range: 0.0 - 1.0, Step: 0.01)
        self.TSDF_thresh = 0.0
        
        # As pointcloud flag
        self.as_pointcloud = True
        
        # Mask sky flag
        self.mask_sky = False
        
        # Clean-up depthmaps flag
        self.clean_depth = True
        
        # Transparent cameras flag
        self.transparent_cams = False

    def get_dict(self):
        return self.__dict__


# def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False,
#               share=False, gradio_delete_cache=False):
#     if not silent:
#         print('Outputing stuff in', tmpdirname)

#     recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, gradio_delete_cache, model, device,
#                                   silent, image_size)
#     model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

#     def get_context(delete_cache):
#         css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
#         title = "MASt3R Demo"
#         if delete_cache:
#             return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
#         else:
#             return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions

#     with get_context(gradio_delete_cache) as demo:
#         # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
#         scene = gradio.State(None)
#         gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
#         with gradio.Column():
#             inputfiles = gradio.File(file_count="multiple")
#             with gradio.Row():
#                 with gradio.Column():
#                     with gradio.Row():
#                         lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
#                         niter1 = gradio.Number(value=500, precision=0, minimum=0, maximum=10_000,
#                                                label="num_iterations", info="For coarse alignment!")
#                         lr2 = gradio.Slider(label="Fine LR", value=0.014, minimum=0.005, maximum=0.05, step=0.001)
#                         niter2 = gradio.Number(value=200, precision=0, minimum=0, maximum=100_000,
#                                                label="num_iterations", info="For refinement!")
#                         optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
#                                                       value='refine+depth', label="OptLevel",
#                                                       info="Optimization level")
#                     with gradio.Row():
#                         matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=5.,
#                                                           minimum=0., maximum=30., step=0.1,
#                                                           info="Before Fallback to Regr3D!")
#                         shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
#                                                             info="Only optimize one set of intrinsics for all views")
#                         scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
#                                                            ("swin: sliding window", "swin"),
#                                                            ("logwin: sliding window with long range", "logwin"),
#                                                            ("oneref: match one image with all", "oneref")],
#                                                           value='complete', label="Scenegraph",
#                                                           info="Define how to make pairs",
#                                                           interactive=True)
#                         with gradio.Column(visible=False) as win_col:
#                             winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
#                                                     minimum=1, maximum=1, step=1)
#                             win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
#                         refid = gradio.Slider(label="Scene Graph: Id", value=0,
#                                               minimum=0, maximum=0, step=1, visible=False)
#             run_btn = gradio.Button("Run")

#             with gradio.Row():
#                 # adjust the confidence threshold
#                 min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
#                 # adjust the camera size in the output pointcloud
#                 cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
#                 TSDF_thresh = gradio.Slider(label="TSDF Threshold", value=0., minimum=0., maximum=1., step=0.01)
#             with gradio.Row():
#                 as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
#                 # two post process implemented
#                 mask_sky = gradio.Checkbox(value=False, label="Mask sky")
#                 clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
#                 transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

#             outmodel = gradio.Model3D()

#             # events
#             scenegraph_type.change(set_scenegraph_options,
#                                    inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
#                                    outputs=[win_col, winsize, win_cyclic, refid])
#             inputfiles.change(set_scenegraph_options,
#                               inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
#                               outputs=[win_col, winsize, win_cyclic, refid])
#             win_cyclic.change(set_scenegraph_options,
#                               inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
#                               outputs=[win_col, winsize, win_cyclic, refid])
#             run_btn.click(fn=recon_fun,
#                           inputs=[scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
#                                   as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
#                                   scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics],
#                           outputs=[scene, outmodel])
#             min_conf_thr.release(fn=model_from_scene_fun,
#                                  inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
#                                          clean_depth, transparent_cams, cam_size, TSDF_thresh],
#                                  outputs=outmodel)
#             cam_size.change(fn=model_from_scene_fun,
#                             inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
#                                     clean_depth, transparent_cams, cam_size, TSDF_thresh],
#                             outputs=outmodel)
#             TSDF_thresh.change(fn=model_from_scene_fun,
#                                inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
#                                        clean_depth, transparent_cams, cam_size, TSDF_thresh],
#                                outputs=outmodel)
#             as_pointcloud.change(fn=model_from_scene_fun,
#                                  inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
#                                          clean_depth, transparent_cams, cam_size, TSDF_thresh],
#                                  outputs=outmodel)
#             mask_sky.change(fn=model_from_scene_fun,
#                             inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
#                                     clean_depth, transparent_cams, cam_size, TSDF_thresh],
#                             outputs=outmodel)
#             clean_depth.change(fn=model_from_scene_fun,
#                                inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
#                                        clean_depth, transparent_cams, cam_size, TSDF_thresh],
#                                outputs=outmodel)
#             transparent_cams.change(model_from_scene_fun,
#                                     inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
#                                             clean_depth, transparent_cams, cam_size, TSDF_thresh],
#                                     outputs=outmodel)
#     demo.launch(share=share, server_name=server_name, server_port=server_port)
    


def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now()
        formatted_date_time = now.strftime(time_format)

        builtin_print(f'[{formatted_date_time}] ', end='')  # print with time stamp
        builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp
    
def dust3r_get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size for inference")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),default=None)
    parser_weights = parser.add_mutually_exclusive_group(required=False)
    #parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=kMast3rFolder + "/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                         "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                         "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False, help="silence logs")
    return parser

def mast3r_get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser


if __name__ == '__main__':
    parser = mast3r_get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    # if args.server_name is not None:
    #     server_name = args.server_name
    # else:
    #     server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)
    
    image_filenames = select_image_files(images_path, start_frame_name, n_frame=10, delta_frame=30)
    print(f'selected image files: {image_filenames}')
        
    img_paths = [os.path.join(images_path, x) for x in image_filenames]
    imgs = [cv2.imread(x) for x in img_paths]
    if False:
        for img in imgs:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    dust3r_preprocessor = Dust3rImagePreprocessor(inference_size=args.image_size, verbose=not args.silent)
    imgs_preproc = dust3r_preprocessor.preprocess_images(imgs)
    print(f'done preprocessing images')    

    def get_context(tmp_dir):
        return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None \
            else nullcontext(tmp_dir)
    with get_context(args.tmp_dir) as tmpdirname:
        cache_path = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)
        print(f'cache_path: {cache_path}')
        #main_demo(cache_path, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent,
        #          share=args.share, gradio_delete_cache=args.gradio_delete_cache)
            
        mast3r_config = Mast3rConfig()
        
        print(f'mast3r_config: {mast3r_config.get_dict()}')
    
        print(f'starting get_reconstructed_scene...')
        global_pc, global_mesh, rgb_imgs, pts3d, mask, cams2world, focals, confs = \
            get_reconstructed_scene(outdir=cache_path, 
                                gradio_delete_cache=args.gradio_delete_cache, 
                                model=model, 
                                device=args.device, 
                                silent=args.silent, 
                                current_scene_state=None,
                                imgs=imgs_preproc, 
                                filelist=image_filenames, **mast3r_config.get_dict())


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
        #is_contiguous = img_char.flags['C_CONTIGUOUS']
        #print(f'image {i}, min {np.min(img_char)}, max {np.max(img_char)}, is contiguous: {is_contiguous}')
        if gl_reverse_rgb:
            img_char = cv2.cvtColor(img_char, cv2.COLOR_RGB2BGR)
        h_ratio = img_char.shape[0] / args.image_size
        viz_camera_images.append(VizCameraImage(image=img_char, Twc=cams2world[i], h_ratio=h_ratio, scale=0.1))
    viewer3D.draw_dense_geometry(point_cloud=viz_point_cloud, mesh=viz_mesh, camera_images=viz_camera_images)
    
    
    # inverted_images = invert_dust3r_preprocess_images([(im*255).astype(np.uint8) for im in rgb_imgs], 
    #                                            [im.shape[0:2] for im in imgs], 
    #                                            args.image_size)

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
        cv2.imshow('Dust3r Images', img_table.image())
        
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
        