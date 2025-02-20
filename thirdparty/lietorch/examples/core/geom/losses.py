import numpy as np
import torch
from lietorch import SO3, SE3, Sim3
from .graph_utils import graph_to_edge_list

def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err

def geodesic_loss(Ps, Gs, graph, gamma=0.9):
    """ Loss function for training network """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = Ps[:,jj] * Ps[:,ii].inv()

    n = len(Gs)
    geodesic_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        dG = Gs[i][:,jj] * Gs[i][:,ii].inv()
        
        # pose error
        d = (dG * dP.inv()).log()

        if isinstance(dG, SE3):
            tau, phi = d.split([3,3], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean())

        elif isinstance(dG, Sim3):
            tau, phi, sig = d.split([3,3,1], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean() + 
                0.05 * sig.norm(dim=-1).mean())
            
        dE = Sim3(dG * dP.inv()).detach()
        r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        'r_error': r_err.mean().item(),
        't_error': t_err.mean().item(),
        's_error': s_err.mean().item(),
    }

    return geodesic_loss, metrics


def residual_loss(residuals, gamma=0.9):
    """ loss on system residuals """
    residual_loss = 0.0
    n = len(residuals)

    for i in range(n):
        w = gamma ** (n - i - 1)
        residual_loss += w * residuals[i].abs().mean()

    return residual_loss, {'residual': residual_loss.item()}
