import sys
sys.path.append('../core')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os

from lietorch import SE3
from networks.slam_system import SLAMSystem
from data_readers import factory

def evaluate(poses_gt, poses_est):
    from rgbd_benchmark.evaluate_ate import evaluate_ate

    poses_gt = poses_gt.cpu().numpy()
    poses_est = poses_est.cpu().numpy()

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    results = evaluate_ate(poses_gt, poses_est)
    print(results)
    return results['absolute_translational_error.rmse']

@torch.no_grad()
def run_slam(tracker, datapath, global_optimization=False, frame_rate=3):
    """ run slam over full sequence """

    torch.multiprocessing.set_sharing_strategy('file_system')
    stream = factory.create_datastream(datapath, frame_rate=frame_rate)

    # store groundtruth poses for evaluatino
    poses_gt = []
    for (tstamp, image, depth, pose, intrinsics) in tqdm(stream):
        tracker.track(tstamp, image[None].cuda(), depth.cuda(), intrinsics.cuda())
        poses_gt.append(pose)

    if global_optimization:
        tracker.global_refinement()

    poses_gt = torch.cat(poses_gt, 0)
    poses_est = tracker.raw_poses()   

    ate = evaluate(poses_gt, poses_est) 
    return ate

def run_evaluation(ckpt, frame_rate=8.0):
    validation_scenes = [
        'rgbd_dataset_freiburg1_360',
        'rgbd_dataset_freiburg1_desk',
        'rgbd_dataset_freiburg1_desk2',
        'rgbd_dataset_freiburg1_floor',
        'rgbd_dataset_freiburg1_plant',
        'rgbd_dataset_freiburg1_room',
        'rgbd_dataset_freiburg1_rpy',
        'rgbd_dataset_freiburg1_teddy',
        'rgbd_dataset_freiburg1_xyz',
    ]

    results = {}
    for scene in validation_scenes:
        # initialize tracker / load weights
        tracker = SLAMSystem(None)
        tracker.load_state_dict(torch.load(ckpt))
        tracker.eval()
        tracker.cuda()
        
        datapath = os.path.join('datasets/TUM-RGBD', scene)
        results[scene] = run_slam(tracker, datapath, 
            global_optimization=args.go, frame_rate=frame_rate)

    print("Aggregate Results: ")
    for scene in results:
        print(scene, results[scene])

    print("MEAN: ", np.mean([results[key] for key in results]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='saved network weights')
    parser.add_argument('--frame_rate', type=float, default=8.0, help='frame rate')
    parser.add_argument('--go', action='store_true', help='use global optimization')
    args = parser.parse_args()

    run_evaluation(args.ckpt, frame_rate=args.frame_rate)
