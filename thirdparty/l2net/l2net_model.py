import torch
import torch.nn.init
import torch.nn as nn

eps = 1e-10

class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L2Net(nn.Module):

    def __init__(self):
        super(L2Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True, eps=eps),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True, eps=eps),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True, eps=eps),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True, eps=eps),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True, eps=eps),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True, eps=eps),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias=True),
            nn.BatchNorm2d(128, affine=True, eps=eps),
        )
        return

    def input_norm(self, x):
        # matlab norm
        z = x.contiguous().transpose(2, 3).contiguous().view(x.size(0),-1)
        x_minus_mean = z.transpose(0,1)-z.mean(1)
        sp = torch.std(z,1).detach()
        norm_inp =  x_minus_mean/(sp+1e-12)
        norm_inp = norm_inp.transpose(0, 1).view(-1, 1, x.size(2), x.size(3)).transpose(2,3)
        return norm_inp

    def forward(self, input):
        norm_img = self.input_norm(input)
        x_features = self.features(norm_img)
        return nn.LocalResponseNorm(256,1*256,0.5,0.5)(x_features).view(input.size(0),-1)
