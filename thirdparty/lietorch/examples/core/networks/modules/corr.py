import torch
import torch.nn.functional as F

from geom.sampler_utils import bilinear_sampler
import lietorch_extras


class CorrSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = lietorch_extras.corr_index_forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = lietorch_extras.corr_index_backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, num, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch*num*h1*w1, 1, h2, w2)
        
        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch*num, h1, w1, h2//2**i, w2//2**i))
            corr = F.avg_pool2d(corr, 2, stride=2)
            
    def __call__(self, coords):
        out_pyramid = []
        batch, num, ht, wd, _ = coords.shape
        coords = coords.permute(0,1,4,2,3)
        coords = coords.contiguous().view(batch*num, 2, ht, wd)
        
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i], coords/2**i, self.radius)
            out_pyramid.append(corr.view(batch, num, -1, ht, wd))

        return torch.cat(out_pyramid, dim=2)

    def append(self, other):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], 0)

    def remove(self, ix):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = self.corr_pyramid[i][ix].contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        """ all-pairs correlation """
        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch*num, dim, ht*wd) / 4.0
        fmap2 = fmap2.reshape(batch*num, dim, ht*wd) / 4.0
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        return corr.view(batch, num, ht, wd, ht, wd)


class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, r):
        ctx.r = r
        fmap1 = fmap1.contiguous()
        fmap2 = fmap2.contiguous()
        coords = coords.contiguous()
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = lietorch_extras.altcorr_forward(fmap1, fmap2, coords, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            lietorch_extras.altcorr_backward(fmap1, fmap2, coords, grad_corr, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None



class AltCorrBlock:
    def __init__(self, fmaps, inds, num_levels=4, radius=3):
        self.num_levels = num_levels
        self.radius = radius
        self.inds = inds

        B, N, C, H, W = fmaps.shape
        fmaps = fmaps.view(B*N, C, H, W)
        
        self.pyramid = []
        for i in range(self.num_levels):
            sz = (B, N, H//2**i, W//2**i, C)
            fmap_lvl = fmaps.permute(0, 2, 3, 1)
            self.pyramid.append(fmap_lvl.reshape(*sz))
            fmaps = F.avg_pool2d(fmaps, 2, stride=2)
  
    def corr_fn(self, coords, ii, jj):
        B, N, H, W, S, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3, 5)

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][:, ii]
            fmap2_i = self.pyramid[i][:, jj]

            coords_i = (coords / 2**i).reshape(B*N, S, H, W, 2).contiguous()
            fmap1_i = fmap1_i.reshape((B*N,) + fmap1_i.shape[2:])
            fmap2_i = fmap2_i.reshape((B*N,) + fmap2_i.shape[2:])

            corr = CorrLayer.apply(fmap1_i, fmap2_i, coords_i, self.radius)
            corr = corr.view(B, N, S, -1, H, W).permute(0, 1, 3, 4, 5, 2)
            corr_list.append(corr)

        corr = torch.cat(corr_list, dim=2)
        return corr / 16.0


    def __call__(self, coords, ii, jj):
        squeeze_output = False
        if len(coords.shape) == 5:
            coords = coords.unsqueeze(dim=-2)
            squeeze_output = True

        corr = self.corr_fn(coords, ii, jj)
        
        if squeeze_output:
            corr = corr.squeeze(dim=-1)

        return corr.contiguous()

