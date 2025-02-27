import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('mast3r') 

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
import trimesh
import torchvision.transforms.functional
from matplotlib import pyplot as pl

from utils_draw import draw_feature_matches
from utils_img import img_from_floats
from utils_dust3r import  Dust3rImagePreprocessor, convert_mv_output_to_geometry, estimate_focal_knowing_depth, calibrate_camera_pnpransac
from utils_depth import point_cloud_to_depth
from utils_sys import Printer

from viewer3D import Viewer3D, VizPointCloud, VizMesh, VizCameraImage
from utils_img import ImageTable
import time


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kMast3rFolder = kRootFolder + '/thirdparty/mast3r'
kResultsFolder = kRootFolder + '/results/mast3r'

model_name = kMast3rFolder + "/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
device = 'cuda'

# Input images (can be of different sizes)
#images = load_images([kMast3rFolder + '/dust3r/croco/assets/Chateau1.png', kMast3rFolder + '/dust3r/croco/assets/Chateau2.png'], size=512)
# images = load_images([kMast3rFolder + '/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg', 
#                       kMast3rFolder + '/assets/NLE_tower/2679C386-1DC0-4443-81B5-93D7EDE4AB37-83120-000041DADB2EA917.jpg'], size=512)
# images = load_images([kMast3rFolder + '/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg', 
#                       kMast3rFolder + '/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg'], size=512)  # test same image
    
# Input images (can be of different sizes)
#image_paths = [kMast3rFolder + '/dust3r/croco/assets/Chateau1.png', kMast3rFolder + '/dust3r/croco/assets/Chateau2.png']
images_paths = [kMast3rFolder + '/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg', 
                kMast3rFolder + '/assets/NLE_tower/2679C386-1DC0-4443-81B5-93D7EDE4AB37-83120-000041DADB2EA917.jpg']
# images_paths = [kMast3rFolder + '/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg', 
#                 kMast3rFolder + '/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg']

if __name__ == '__main__': 
    
    min_conf_thr = 50   # percentage of the max confidence value
    inference_size = 512
    invert_colors_on_loading = True

    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    imgs = []
    for p in images_paths:
        img = cv2.imread(p)
        if img.ndim == 3 and invert_colors_on_loading:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        imgs.append(img)
        
    dust3r_preprocessor = Dust3rImagePreprocessor(inference_size=inference_size)
        
    imgs_preproc = dust3r_preprocessor.preprocess_images(imgs) 
    
    # get inference output 
    output = inference([tuple(imgs_preproc)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    # extract descriptors
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    
    # extract rgb images
    rgb_imgs = [output['view1']['img']] + [output['view2']['img']]
    for i in range(len(rgb_imgs)):
        rgb_imgs[i] = (rgb_imgs[i] + 1) / 2
        rgb_imgs[i] = rgb_imgs[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
        rgb_imgs[i] = cv2.cvtColor(rgb_imgs[i], cv2.COLOR_RGB2BGR)
                        
    # extract 3D points
    pts3d = [output['pred1']['pts3d'][0]] + [output['pred2']['pts3d_in_other_view'][0]]
    
        
    # extract predicted confidence 
    conf = [output['pred1']['conf'][0]] + [output['pred2']['conf'][0]]
    conf_vec = torch.stack([x.reshape(-1) for x in conf]) # get a monodimensional vector
    conf_sorted = conf_vec.reshape(-1).sort()[0]    
    conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
    print(f'confidence threshold: {conf_thres}')
    mask = [x >= conf_thres for x in conf]
    
    
    # estimate focals of first image
    h, w = rgb_imgs[0].shape[0:2] # [H, W]    
    conf_first = conf[0].reshape(-1) # [bs, H * W]
    conf_first_sorted = conf_first.sort()[0] # [bs, h * w]
    #conf_first_thres = conf_first_sorted[int(conf_first_sorted.shape[0] * 0.03)]  # here we use a different threshold 
    conf_first_thres = conf_first_sorted[int(conf_first_sorted.shape[0] * float(min_conf_thr) * 0.01)]
    valid_first = (conf_first_sorted >= conf_first_thres) # & valids[0].reshape(bs, -1)
    valid_first = valid_first.reshape(h, w)     
    
    focals = estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda()).cpu().item()
    
    intrinsics = torch.eye(3,)
    intrinsics[0, 0] = focals
    intrinsics[1, 1] = focals
    intrinsics[0, 2] = w / 2
    intrinsics[1, 2] = h / 2
    intrinsics = intrinsics.cuda()    


    # estimate camera poses
    y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda() # [H, W, 2]
    c2ws = []
    for (pr_pt, valid) in zip(pts3d, mask):
        c2ws_i = calibrate_camera_pnpransac(pr_pt.cuda().flatten(0,1)[None], pixel_coords.flatten(0,1)[None], valid.cuda().flatten(0,1)[None], intrinsics[None])
        c2ws.append(c2ws_i[0])
    cams2world = torch.stack(c2ws, dim=0).cpu() # [N, 4, 4]

    
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
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    print(f'view1 shape: {view1["true_shape"]}, rgb shape: {rgb_imgs[0].shape}')
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # visualize a few matches
    n_viz_percent = 50  # percentage of shown matches
    num_matches = matches_im0.shape[0]
    n_viz = int(100/n_viz_percent)
    match_idx_to_viz = np.arange(0, num_matches - 1, n_viz)  # extract 1 sample every n_viz samples
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    
    
    # convert confidence images to visualizable images 
    for i,img in enumerate(rgb_imgs):
        rgb_imgs[i] = (img*255.0).astype(np.uint8)
        print(f'conf shape: {confs[i].shape}, min: {confs[i].min()}, max: {confs[i].max()}')        
        confs[i] = img_from_floats(confs[i])
            
    rescale_images_from_inference_size = True
    if rescale_images_from_inference_size:
        #inverted_images = invert_dust3r_preprocess_images(rgb_imgs, [im.shape[0:2] for im in imgs], inference_size)
        inverted_images = dust3r_preprocessor.rescale_images(rgb_imgs)
        
    
    # extract depth at the original scale
    h_original, w_original = imgs[0].shape[0:2]
    h, w = rgb_imgs[0].shape[0:2] # [H, W] 
    depth_scale_h = h_original / h
    depth_scale_w = w_original / w
    are_scales_equal = abs(depth_scale_h-depth_scale_w) < 1e-2
    if not are_scales_equal:
        Printer.yellow(f'WARNING: depth_scale_h: {depth_scale_h} != depth_scale_w: {depth_scale_w}')
        # extract first image depth with mask 
        valid_first = mask[0].reshape(h,w)
        depth = np.zeros(shape=(h,w),dtype=pts3d[0].dtype)
        depth[valid_first] = pts3d[0][valid_first,2] # extract the z-component from the point cloud
        # extract depth at the original scale by interpolating
        #depth_map = invert_dust3r_preprocess_depth(depth, imgs[0].shape[0:2], inference_size)
        depth_map = dust3r_preprocessor.rescale_depth(depth, image_idx=0)
        depth_map_img = img_from_floats(depth_map)        
    else:
        print(f'depth_scale_h: {depth_scale_h}, depth_scale_w: {depth_scale_w}')
        # extract depth at the original scale by projecting point cloud on the image
        intrinsics_original = to_numpy(intrinsics)*depth_scale_h
        depth_map = point_cloud_to_depth(global_pc.vertices.reshape(-1, 3), intrinsics_original, w_original, h_original)
        depth_map_img = img_from_floats(depth_map)
    cv2.imshow('Depth', depth_map_img)


    show_image_tables = True
    table_resize_scale=0.8    
    if show_image_tables:
        img_table_originals = ImageTable(num_columns=2, resize_scale=table_resize_scale)
        for i, img in enumerate(imgs):
            if img.ndim == 3 and invert_colors_on_loading:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_table_originals.add(img)
        img_table_originals.render()
        cv2.imshow('Original Images', img_table_originals.image())
        
        img_table = ImageTable(num_columns=2, resize_scale=table_resize_scale)
        for i, img in enumerate(rgb_imgs):
            img_table.add(img)
        img_table.render()
        cv2.imshow('Dust3r Images', img_table.image())
        
        if rescale_images_from_inference_size:
            img_inverted_table = ImageTable(num_columns=2, resize_scale=table_resize_scale)
            for i, img in enumerate(inverted_images):
                img_inverted_table.add(img)
            img_inverted_table.render()
            cv2.imshow('Inverted Images', img_inverted_table.image())        
        
        img_table_conf = ImageTable(num_columns=2, resize_scale=table_resize_scale)
        for i, conf in enumerate(confs):
            print(f'adding confidence image {i}, shape: {conf.shape}, min: {np.min(conf)}, max: {np.max(conf)}')
            img_table_conf.add(conf)
        img_table_conf.render()
        cv2.imshow('Confidence Images', img_table_conf.image())
        
    
    out_img = draw_feature_matches(rgb_imgs[0], rgb_imgs[1], viz_matches_im0, viz_matches_im1)
    cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
    cv2.imshow('matches', out_img)
    
    viewer3D = Viewer3D()
    time.sleep(1)
    
    viz_point_cloud = VizPointCloud(points=global_pc.vertices, colors=global_pc.colors, normalize_colors=True, reverse_colors=True) if global_pc is not None else None
    viz_mesh = VizMesh(vertices=global_mesh.vertices, triangles=global_mesh.faces, vertex_colors=global_mesh.visual.vertex_colors, normalize_colors=True) if global_mesh is not None else None
    viz_camera_images = []
    for i, img in enumerate(rgb_imgs):
        img_char = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        viz_camera_images.append(VizCameraImage(image=img_char, Twc=cams2world[i], scale=0.1))
    viewer3D.draw_dense_geometry(point_cloud=viz_point_cloud, mesh=viz_mesh, camera_images=viz_camera_images)

    while viewer3D.is_running():
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            break    
        
    viewer3D.quit()