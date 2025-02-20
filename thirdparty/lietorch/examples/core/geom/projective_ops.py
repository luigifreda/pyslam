import torch
import torch.nn.functional as F

from lietorch import SE3, Sim3

MIN_DEPTH = 0.1

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def iproj(disps, intrinsics):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    return torch.stack([X, Y, i, disps], dim=-1)

def proj(Xs, intrinsics, jacobian=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)
    d = torch.where(Z.abs() < 0.001, torch.zeros_like(Z), 1.0/Z)

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    coords = torch.stack([x,y, D*d], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 3, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False):
    """ map points from ii->jj """

    # inverse project (pinhole)
    X0 = iproj(depths[:,ii], intrinsics[:,ii])
    
    # transform
    Gij = poses[:,jj] * poses[:,ii].inv()
    X1, Ja = actp(Gij, X0, jacobian=jacobian)
    
    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian)

    # exclude points too close to camera
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)
        return x1, valid, (Ji, Jj)

    return x1, valid


def induced_flow(poses, disps, intrinsics, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj)

    return coords1[...,:2] - coords0, valid

