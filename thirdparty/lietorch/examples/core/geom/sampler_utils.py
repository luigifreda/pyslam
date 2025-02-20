import torch
import torch.nn.functional as F

def _bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def bilinear_sampler(img, coords):
    """ Wrapper for bilinear sampler for inputs with extra batch dimensions """
    unflatten = False
    if len(img.shape) == 5:
        unflatten = True
        b, n, c, h, w = img.shape
        img = img.view(b*n, c, h, w)
        coords = coords.view(b*n, h, w, 2)

    img1 = _bilinear_sampler(img, coords)
    
    if unflatten:
        return img1.view(b, n, c, h, w)

    return img1

def sample_depths(depths, coords):
    batch, num, ht, wd = depths.shape
    depths = depths.view(batch, num, 1, ht, wd)
    coords = coords.view(batch, num, ht, wd, 2)
    
    depths_proj = bilinear_sampler(depths, coords)
    return depths_proj.view(batch, num, ht, wd, 1)

