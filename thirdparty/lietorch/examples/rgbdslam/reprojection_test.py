import sys
sys.path.append('../core')

import torch
import cv2
import numpy as np

from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
import geom.projective_ops as pops
from geom.sampler_utils import bilinear_sampler

def show_image(image):
    if len(image.shape) == 3:
        image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def reproj_test(args, N=2):
    """ Test to make sure project transform correctly maps points """

    db = dataset_factory(args.datasets, n_frames=N)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=0)

    for item in train_loader:
        images, poses, depths, intrinsics = [x.to('cuda') for x in item]        
        poses = SE3(poses).inv()
        disps = 1.0 / depths

        coords, _ = pops.projective_transform(poses, disps, intrinsics, [0], [1])
        imagew = bilinear_sampler(images[:,[1]], coords[...,[0,1]])

        # these two image should show camera motion
        show_image(images[0,0])
        show_image(images[0,1])

        # these two images should show the camera motion removed by reprojection / warping
        show_image(images[0,0])
        show_image(imagew[0,0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    args = parser.parse_args()

    reproj_test(args)
