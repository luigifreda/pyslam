import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('crestereo_pytorch') 
import os

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time

from crestereo_pytorch.nets import Model


data_path = '../data'
crestereo_base_path='../../thirdparty/crestereo_pytorch'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

	print("Model Forwarding...")
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	# print(imgR_dw2.shape)
	with torch.inference_mode():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

	return pred_disp


if __name__ == "__main__":    

    # left_img_path = data_path + '/stereo_bicycle/im0.png'
    # right_img_path = data_path + '/stereo_bicycle/im1.png'
    
    left_img_path = data_path + '/kitti06-12-color.png'
    right_img_path = data_path + '/kitti06-12-R-color.png'    
    
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    in_h, in_w = left_img.shape[:2]

    print(f"input shape: {left_img.shape}")

	# Resize image in case the GPU memory overflows
    eval_h, eval_w = in_h, in_w
    #eval_h, eval_w = ((in_h//8)*8, (in_w//8)*8)
    # assert eval_h%8 == 0, "input height should be divisible by 8"
    # assert eval_w%8 == 0, "input width should be divisible by 8"
	
    if eval_h != in_h or eval_w != in_w:
        imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    else:
        imgL = left_img
        imgR = right_img

    model_path = crestereo_base_path + "/models/crestereo_eth3d.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.to(device)
    model.eval()
    
    time_start = time.time()

    pred = inference(imgL, imgR, model, n_iter=20)

    time_end = time.time()

    print(f"Inference time: {time_end-time_start}")

    if eval_h != in_h or eval_w != in_w:
        t = float(in_w) / float(eval_w)
        disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    else:
        disp = pred

    disp_min = disp.min()
    disp_max = disp.max()
    print(f"disp_min: {disp_min}, disp_max: {disp_max}")

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    
    bf = 379.8145
    depth = bf / (np.abs(disp) + 1e-6)
    
    depth_min = depth.min()
    depth_max = 50 #depth.max()
    print(f"depth_min: {depth_min}, depth_max: {depth_max}")
    
    valid_mask = (depth > 0) & (depth < 50)
    depth_vis = np.zeros_like(depth)
    depth_vis[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min) * 255.0
    depth_vis = depth_vis.astype("uint8")
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    cv2.imshow("input", left_img)
    cv2.imshow("disparity", disp_vis)
    cv2.imshow("depth", depth_vis)
    
    
    #cv2.imwrite("output.jpg", disp_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()