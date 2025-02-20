import sys
sys.path.append('../core')

import argparse
import torch
import cv2
import numpy as np

from viz import sim3_visualization
from lietorch import SO3, SE3, Sim3
from networks.sim3_net import Sim3Net

def normalize_images(images):
    images = images[:, :, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
    return (images/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

def load_example(i=0):
    """ get demo example """
    DEPTH_SCALE = 5.0
    if i==0:
        image1 = cv2.imread('assets/image1.png')
        image2 = cv2.imread('assets/image2.png')
        depth1 = np.load('assets/depth1.npy') / DEPTH_SCALE
        depth2 = np.load('assets/depth2.npy') / DEPTH_SCALE
    
    elif i==1:
        image1 = cv2.imread('assets/image3.png')
        image2 = cv2.imread('assets/image4.png')
        depth1 = np.load('assets/depth3.npy') / DEPTH_SCALE
        depth2 = np.load('assets/depth4.npy') / DEPTH_SCALE

    images = np.stack([image1, image2], 0)
    images = torch.from_numpy(images).permute(0,3,1,2)

    depths = np.stack([depth1, depth2], 0)
    depths = torch.from_numpy(depths).float()

    intrinsics = np.array([320.0, 320.0, 320.0, 240.0])
    intrinsics = np.tile(intrinsics[None], (2,1))
    intrinsics = torch.from_numpy(intrinsics).float()

    return images[None].cuda(), depths[None].cuda(), intrinsics[None].cuda()


@torch.no_grad()
def demo(model, index=0):

    images, depths, intrinsics = load_example(index)

    # initial transformation estimate
    if args.transformation == 'SE3':
        Gs = SE3.Identity(1, 2, device='cuda')

    elif args.transformation == 'Sim3':
        Gs = Sim3.Identity(1, 2, device='cuda')
        depths[:,0] *= 2**(2*torch.rand(1) - 1.0).cuda()

    images1 = normalize_images(images)
    ests, _ = model(Gs, images1, depths, intrinsics, num_steps=12)

    # only care about last transformation
    Gs = ests[-1] 
    T = Gs[:,0] * Gs[:,1].inv()
    
    T = T[0].matrix().double().cpu().numpy()
    sim3_visualization(T, images, depths, intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformation', default='SE3', help='checkpoint to restore')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    args = parser.parse_args()

    model = Sim3Net(args)
    model.load_state_dict(torch.load(args.ckpt))

    model.cuda()
    model.eval()

    # run two demos
    demo(model, 0)
    demo(model, 1)

