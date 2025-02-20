import lietorch
import torch
import torch.nn.functional as F

from .chol import block_solve
import geom.projective_ops as pops

# utility functions for scattering ops
def safe_scatter_add_mat(H, data, ii, jj, B, M, D):
    v = (ii >= 0) & (jj >= 0)
    H.scatter_add_(1, (ii[v]*M + jj[v]).view(1,-1,1,1).repeat(B,1,D,D), data[:,v])

def safe_scatter_add_vec(b, data, ii, B, M, D):
    v = ii >= 0
    b.scatter_add_(1, ii[v].view(1,-1,1).repeat(B,1,D), data[:,v])

def MoBA(target, weight, poses, disps, intrinsics, ii, jj, fixedp=1, lm=0.0001, ep=0.1):
    """ MoBA: Motion Only Bundle Adjustment """

    B, M = poses.shape[:2]
    D = poses.manifold_dim
    N = ii.shape[0]

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.view(B, N, -1, D)
    Jj = Jj.view(B, N, -1, D)
    wJiT = (.001 * w * Ji).transpose(2,3)
    wJjT = (.001 * w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    M = M - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    H = torch.zeros(B, M*M, D, D, device=target.device)
    safe_scatter_add_mat(H, Hii, ii, ii, B, M, D)
    safe_scatter_add_mat(H, Hij, ii, jj, B, M, D)
    safe_scatter_add_mat(H, Hji, jj, ii, B, M, D)
    safe_scatter_add_mat(H, Hjj, jj, jj, B, M, D)
    H = H.reshape(B, M, M, D, D)

    v = torch.zeros(B, M, D, device=target.device)
    safe_scatter_add_vec(v, vi, ii, B, M, D)
    safe_scatter_add_vec(v, vj, jj, B, M, D)

    ### 3: solve the system + apply retraction ###
    dx = block_solve(H, v, ep=ep, lm=lm)
    
    poses1, poses2 = poses[:,:fixedp], poses[:,fixedp:]
    poses2 = poses2.retr(dx)
    
    poses = lietorch.cat([poses1, poses2], dim=1)
    return poses

def SLessBA(target, weight, poses, disps, intrinsics, ii, jj, fixedp=1):
    """ Structureless Bundle Adjustment """
    pass


def BA(target, weight, poses, disps, intrinsics, ii, jj, fixedp=1):
    """ Full Bundle Adjustment """
    pass