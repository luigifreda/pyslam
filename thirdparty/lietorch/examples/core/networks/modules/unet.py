import torch
import torch.nn as nn

# Unet model from https://github.com/usuyama/pytorch-unet



GRAD_CLIP = .01

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 5, padding=2),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(128, 128)
        self.dconv_down2 = double_conv(128, 256)
        self.dconv_down3 = double_conv(256, 256)
        # self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.dconv_up2 = double_conv(256 + 256, 128)
        self.dconv_up1 = double_conv(128 + 128, 128)

        self.conv_r = nn.Conv2d(128, 3, 1)
        self.conv_w = nn.Conv2d(128, 3, 1)


    def forward(self, x):
        b, n, c, ht, wd = x.shape
        x = x.view(b*n, c, ht, wd)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        r = self.conv_r(x)
        w = self.conv_w(x)

        w = torch.sigmoid(w)
        w = w.view(b, n, 3, ht, wd).permute(0,1,3,4,2)
        r = r.view(b, n, 3, ht, wd).permute(0,1,3,4,2)

        # w = GradClip.apply(w)
        # r = GradClip.apply(r)
        return r, w