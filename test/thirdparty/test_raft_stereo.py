import sys
import pyslam.config as config

config.cfg.set_lib("raft_stereo")

import torch
import cv2
import numpy as np
import argparse
import time

from raft_stereo import RAFTStereo
from core.utils.utils import InputPadder

from pyslam.utilities.utils_depth import img_from_depth

data_path = "../data"
stereo_raft_base_path = "../../thirdparty/raft_stereo"

# DEVICE = 'cuda'

# def load_image(imfile):
#     img = np.array(Image.open(imfile)).astype(np.uint8)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     return img[None].to(DEVICE)


def demo(args, imgfile1, imgfile2, restore_ckpt):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(restore_ckpt))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("DepthEstimatorDepthPro: Using CUDA")
    else:
        print("DepthEstimatorDepthPro: Using CPU")

    model = model.module
    model.to(device)
    model.eval()

    with torch.no_grad():

        image1_np = cv2.imread(imgfile1)
        image2_np = cv2.imread(imgfile2)

        start_time = time.time()

        image1 = torch.from_numpy(image1_np).permute(2, 0, 1).float()
        image1 = image1[None].to(device)
        image2 = torch.from_numpy(image2_np).permute(2, 0, 1).float()
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()
        disparity_map = flow_up.cpu().numpy().squeeze()

        end_time = time.time()
        print("Time taken for stereo depth prediction: ", end_time - start_time)

        depth_img = img_from_depth(disparity_map)

        stereo_pair = np.concatenate([image1_np, image2_np], axis=1)
        cv2.imshow("stereo pair", stereo_pair)
        cv2.imshow("depth", depth_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Architecture choices
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[128] * 3,
        help="hidden state and context dimensions",
    )
    parser.add_argument(
        "--corr_implementation",
        choices=["reg", "alt", "reg_cuda", "alt_cuda"],
        default="reg",
        help="correlation volume implementation",
    )
    parser.add_argument(
        "--shared_backbone",
        action="store_true",
        help="use a single backbone for the context and feature encoders",
    )
    parser.add_argument(
        "--corr_levels", type=int, default=4, help="number of levels in the correlation pyramid"
    )
    parser.add_argument(
        "--corr_radius", type=int, default=4, help="width of the correlation pyramid"
    )
    parser.add_argument(
        "--n_downsample", type=int, default=2, help="resolution of the disparity field (1/2^K)"
    )
    parser.add_argument(
        "--context_norm",
        type=str,
        default="batch",
        choices=["group", "batch", "instance", "none"],
        help="normalization of context encoder",
    )
    parser.add_argument(
        "--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently"
    )
    parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels")

    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )

    args = parser.parse_args()

    imgfile1 = data_path + "/stereo_bicycle/im0.png"
    imgfile2 = data_path + "/stereo_bicycle/im1.png"

    model = "REALTIME"  # 'ETH3D', 'REALTIME', 'MIDDLEBURY'

    if model == "ETH":
        restore_ckpt = stereo_raft_base_path + "/models/raftstereo-eth3d.pth"
    elif model == "REALTIME":
        restore_ckpt = (
            stereo_raft_base_path + "/models/raftstereo-realtime.pth"
        )  # --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
        args.shared_backbone = True
        args.n_downsample = 3
        args.n_gru_layers = 2
        args.slow_fast_gru = True
        args.valid_iters = 7
        # args.corr_implementation = 'reg_cuda'
        args.mixed_precision = True
    elif model == "MIDDLEBURY":
        restore_ckpt = stereo_raft_base_path + "/models/raftstereo-middlebury.pth"

    demo(args, imgfile1, imgfile2, restore_ckpt)
