import sys
sys.path.append('../core')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os

from viz import SLAMFrontend
from lietorch import SE3
from networks.slam_system import SLAMSystem
from data_readers import factory


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(10)

def evaluate(poses_gt, poses_est):
    from rgbd_benchmark.evaluate_ate import evaluate_ate

    poses_gt = poses_gt.cpu().numpy()
    poses_est = poses_est.cpu().numpy()

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    results = evaluate_ate(poses_gt, poses_est)
    print(results)


@torch.no_grad()
def run_slam(tracker, datapath, frame_rate=8.0):
    """ run slam over full sequence """

    torch.multiprocessing.set_sharing_strategy('file_system')
    stream = factory.create_datastream(args.datapath, frame_rate=frame_rate)

    # start the frontend thread
    if args.viz:
        frontend = SLAMFrontend().start()
        tracker.set_frontend(frontend)

    # store groundtruth poses for evaluation
    poses_gt = []

    for (tstamp, image, depth, pose, intrinsics) in tqdm(stream):
        tracker.track(tstamp, image[None].cuda(), depth.cuda(), intrinsics.cuda())
        poses_gt.append(pose)

        if args.viz:
            show_image(image[0])
            frontend.update_pose(tstamp, pose[0], gt=True)

    # global optimization / loop closure
    if args.go:
        tracker.global_refinement()

    poses_gt = torch.cat(poses_gt, 0)
    poses_est = tracker.raw_poses()
    evaluate(poses_gt, poses_est) 
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', help='path to video for slam')
    parser.add_argument('--ckpt', help='saved network weights')
    parser.add_argument('--viz', action='store_true', help='run visualization frontent')
    parser.add_argument('--go', action='store_true', help='use global optimization')
    parser.add_argument('--frame_rate', type=float, default=8.0, help='frame rate')
    args = parser.parse_args()

    # initialize tracker / load weights
    tracker = SLAMSystem(args)
    tracker.load_state_dict(torch.load(args.ckpt))
    tracker.eval()
    tracker.cuda()

    run_slam(tracker, args.datapath, args.frame_rate)
