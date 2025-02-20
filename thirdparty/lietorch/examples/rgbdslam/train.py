import sys
sys.path.append('../core')

import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom.losses import geodesic_loss, residual_loss

# network
from networks.rslam import RaftSLAM
from logger import Logger
from evaluate import run_evaluation

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def normalize_images(images):
    images = images[:, :, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
    return (images/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

def train(args):
    """ Test to make sure project transform correctly maps points """

    N = args.n_frames
    model = RaftSLAM(args)
    model.cuda()
    model.train()

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    db = dataset_factory(args.datasets, n_frames=N, fmin=16.0, fmax=96.0)
    train_loader = DataLoader(db, batch_size=args.batch, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            graph = OrderedDict()
            for i in range(N):
                graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
            
            images, poses, depths, intrinsics = [x.to('cuda') for x in item]
            
            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.Identity(Ps.shape, device='cuda')

            images = normalize_images(images)
            Gs, residuals = model(Gs, images, depths, intrinsics, graph, num_steps=args.iters)

            geo_loss, geo_metrics = geodesic_loss(Ps, Gs, graph)
            res_loss, res_metrics = residual_loss(residuals)

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)

            loss = args.w1 * geo_loss + args.w2 * res_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            logger.push(metrics)
            total_steps += 1

            if total_steps % 10000 == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

                run_evaluation(PATH)

            if total_steps >= args.steps:
                should_keep_training = False
                break

    return model
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')

    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--iters', type=int, default=8)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=4)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.1)

    args = parser.parse_args()

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    model = train(args)

