import sys
sys.path.append('../core')

import argparse
import torch
import cv2
import numpy as np
from collections import OrderedDict

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_readers.tartan import TartanAir, TartanAirTest

from lietorch import SO3, SE3, Sim3
from geom.losses import *

# network
from networks.sim3_net import Sim3Net
from logger import Logger


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def normalize_images(images):
    images = images[:, :, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
    return (images/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

@torch.no_grad()
def evaluate(model):
    """ evaluate trained model """

    model.cuda()
    model.eval()

    R_THRESHOLD = 0.1
    T_THRESHOLD = 0.01
    S_THRESHOLD = 0.01

    model.eval()
    db = TartanAirTest()
    test_loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=4)

    # random scales, make sure they are the same every time
    from numpy.random import default_rng
    rng = default_rng(1234)
    scales = 2 ** rng.uniform(-1.0, 1.0, 2000)
    scales = scales.astype(np.float32)

    metrics = {'t': [], 'r': [], 's': []}
    for i_batch, item in enumerate(test_loader):
        images, poses, depths, intrinsics = [x.to('cuda') for x in item]

        # convert poses w2c -> c2w
        Ps = SE3(poses).inv()
        batch, num = images.shape[:2]

        if args.transformation == 'SE3':
            Gs = SE3.Identity(Ps.shape, device='cuda')

        elif args.transformation == 'Sim3':
            Ps = Sim3(Ps)
            Gs = Sim3.Identity(Ps.shape, device='cuda')

            s = torch.as_tensor(scales[i_batch]).cuda().unsqueeze(0)
            phi = torch.zeros(batch, num, 7, device='cuda')
            phi[:,0,6] = s.log()

            Ps = Sim3.exp(phi) * Ps
            depths[:,0] *= s[:,None,None]

        images = normalize_images(images)
        Gs, _ = model(Gs, images, depths, intrinsics, num_steps=16)

        Gs = Gs[-1]
        dP = Ps[:,1] * Ps[:,0].inv()
        dG = Gs[:,1] * Gs[:,0].inv()

        dE = Sim3(dP.inv() * dG)
        r_err, t_err, s_err = pose_metrics(dE)

        t_err = t_err * TartanAir.DEPTH_SCALE

        metrics['t'].append(t_err.item())
        metrics['r'].append(r_err.item())
        metrics['s'].append(s_err.item())

    rlist = np.array(metrics['r'])
    tlist = np.array(metrics['t'])
    slist = np.array(metrics['s'])
    
    r_all = np.count_nonzero(rlist < R_THRESHOLD) / len(metrics['r'])
    t_all = np.count_nonzero(tlist < T_THRESHOLD) / len(metrics['t'])
    s_all = np.count_nonzero(slist < S_THRESHOLD) / len(metrics['s'])

    print("Rotation Acc: ", r_all)
    print("Translation Acc: ", t_all)
    print("Scale Acc: ", s_all)


def train(args):
    """ Test to make sure project transform correctly maps points """

    model = Sim3Net(args)
    model.cuda()
    model.train()

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    db = TartanAir(mode='training', n_frames=2, do_aug=True, fmin=8.0, fmax=100.0)
    train_loader = DataLoader(db, batch_size=args.batch, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, 100000, pct_start=0.01, cycle_momentum=False)

    from collections import OrderedDict
    graph = OrderedDict()
    graph[0] = [1]
    graph[1] = [0]

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()
            images, poses, depths, intrinsics = [x.to('cuda') for x in item]
            
            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            batch, num = images.shape[:2]

            if args.transformation == 'SE3':
                Gs = SE3.Identity(Ps.shape, device='cuda')

            elif args.transformation == 'Sim3':
                Ps = Sim3(Ps)
                Gs = Sim3.Identity(Ps.shape, device='cuda')

                s = 2**(2*torch.rand(batch) - 1.0).cuda()
                phi = torch.zeros(batch, num, 7, device='cuda')
                phi[:,0,6] = s.log()

                Ps = Sim3.exp(phi) * Ps
                depths[:,0] *= s[:,None,None]

            images = normalize_images(images)
            Gs, residuals = model(Gs, images, depths, intrinsics, num_steps=args.iters)

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

            if total_steps % 5000 == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

                model.train()

    return model
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--transformation', default='SE3', help='checkpoint to restore')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--train', action='store_true')

    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--iters', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.1)


    args = parser.parse_args()

    if args.train:
        import os
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')

        model = train(args)
    
    else:
        model = Sim3Net(args)
        model.load_state_dict(torch.load(args.ckpt))

    evaluate(model)

