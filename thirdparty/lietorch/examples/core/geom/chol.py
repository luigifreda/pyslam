import torch
import torch.nn.functional as F
import geom.projective_ops as pops

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

def block_solve(H, b, ep=0.1, lm=0.0001):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    I = torch.eye(D).to(H.device)
    H = H + (ep + lm*H) * I

    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)
    b = b.reshape(B, N*D, 1)

    x = CholeskySolver.apply(H,b)
    return x.reshape(B, N, D)