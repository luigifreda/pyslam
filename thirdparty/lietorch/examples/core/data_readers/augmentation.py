import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F


class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.4/3.14),
            transforms.ToTensor()])

        self.max_scale = 0.25

    def spatial_transform(self, images, depths, poses, intrinsics):
        """ cropping and resizing """
        ht, wd = images.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 2 ** np.random.uniform(min_scale, max_scale)
        intrinsics = scale * intrinsics
        depths = depths.unsqueeze(dim=1)

        images = F.interpolate(images, scale_factor=scale, mode='bilinear', 
            align_corners=True, recompute_scale_factor=True)
        
        depths = F.interpolate(depths, scale_factor=scale, recompute_scale_factor=True)

        # always perform center crop (TODO: try non-center crops)
        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        depths = depths.squeeze(dim=1)
        return images, poses, depths, intrinsics

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, poses, depths, intrinsics):
        images = self.color_transform(images)
        return self.spatial_transform(images, depths, poses, intrinsics)
