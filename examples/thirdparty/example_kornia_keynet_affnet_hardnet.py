import os
import matplotlib.pyplot as plt
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *
from kornia_moons.viz import *


class ImageMatcher:
    def __init__(self, device=K.utils.get_cuda_or_mps_device_if_available()):
        self.device = device
        self.feature = KF.KeyNetAffNetHardNet(5000, True).eval().to(self.device)

    def load_torch_image(self, fname):
        # img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
        # img = K.color.bgr_to_rgb(img)
        img = cv2.imread(fname)
        if img.ndim>2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = K.image_to_tensor(img, False).to(self.device).float() / 255. 
        return img

    def match_images(self, fname1, fname2):
                
        if False: 
            img1 = K.io.load_image(fname1, K.io.ImageLoadType.GRAY32, device=self.device)[None, ...]
            img2 = K.io.load_image(fname2, K.io.ImageLoadType.GRAY32, device=self.device)[None, ...]
        else:
            img1 = self.load_torch_image(fname1)
            img2 = self.load_torch_image(fname2)
        
        # input_dict = {"image0": K.color.rgb_to_grayscale(img1),
        #               "image1": K.color.rgb_to_grayscale(img2)}
        
        hw1 = torch.tensor(img1.shape[2:], device=self.device)
        hw2 = torch.tensor(img2.shape[2:], device=self.device)
        
        adalam_config = {"device": self.device}
        with torch.inference_mode():
            #lafs1, resps1, descs1 = self.feature(K.color.rgb_to_grayscale(img1))
            #lafs2, resps2, descs2 = self.feature(K.color.rgb_to_grayscale(img2))
            lafs1, resps1, descs1 = self.feature(img1)
            lafs2, resps2, descs2 = self.feature(img2)
            
                        
            dists, idxs = KF.match_adalam(descs1.squeeze(0), descs2.squeeze(0),
                                          lafs1, lafs2,
                                          config=adalam_config,
                                          hw1=hw1, hw2=hw2)
        mkpts1, mkpts2 = self.get_matching_keypoints(lafs1, lafs2, idxs)
        Fm, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.75, 0.999, 100000)
        inliers = inliers > 0
        return img1, img2, lafs1, lafs2, idxs, inliers

    def get_matching_keypoints(self, lafs1, lafs2, idxs):
        mkpts1 = KF.get_laf_center(lafs1).squeeze()[idxs[:,0]].detach().cpu().numpy()
        mkpts2 = KF.get_laf_center(lafs2).squeeze()[idxs[:,1]].detach().cpu().numpy()
        return mkpts1, mkpts2

    def draw_matches(self, img1, img2, lafs1, lafs2, idxs, inliers):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        draw_LAF_matches(lafs1.cpu(), lafs2.cpu(), idxs.cpu(), K.tensor_to_image(img1.cpu()), K.tensor_to_image(img2.cpu()), inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2), 'tentative_color': (1, 1, 0.2, 0.3), 'feature_color': None, 'vertical': False}, 
            ax=ax)

if __name__ == '__main__':
    matcher = ImageMatcher()
    output_path = ''

    fname1 = "../data/kn_church-2.jpg"
    fname2 = "../data/kn_church-8.jpg"

    img1, img2, lafs1, lafs2, idxs, inliers = matcher.match_images(fname1, fname2)
    fig = matcher.draw_matches(img1, img2, lafs1, lafs2, idxs, inliers)
    plt.show()
    #fig.savefig(output_path)