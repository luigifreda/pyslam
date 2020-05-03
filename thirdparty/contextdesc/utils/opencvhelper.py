#!/usr/bin/env python
"""
Copyright 2018, Zixin Luo, HKUST.
OpenCV helper.
"""

from __future__ import print_function

import numpy as np
import cv2


class SiftWrapper(object):
    """"OpenCV SIFT wrapper."""

    def __init__(self, rootsift=True, n_feature=0, n_octave_layers=3,
                 peak_thld=0.0067, edge_thld=10, sigma=1.6,
                 n_sample=8192, patch_size=32):
        self.sift = None
        self.rootsift = rootsift

        self.n_feature = n_feature
        self.n_octave_layers = n_octave_layers
        self.peak_thld = peak_thld
        self.edge_thld = edge_thld
        self.sigma = sigma
        self.n_sample = n_sample
        self.down_octave = True

        self.init_sigma = 0.5
        self.sift_descr_scl_fctr = 3.
        self.sift_descr_width = 4

        self.first_octave = None
        self.max_octave = None
        self.pyr = None

        self.patch_size = patch_size
        self.output_gird = None

        self.pyr_off = False
        self.ori_off = False
        self.half_sigma = True

    def create(self):
        """Create OpenCV SIFT detector."""
        self.sift = cv2.xfeatures2d.SIFT_create(
            0,
            self.n_octave_layers, 
            self.peak_thld, 
            self.edge_thld, 
            self.sigma)

    def detect(self, gray_img):
        """Detect keypoints in the gray-scale image.
        Args:
            gray_img: The input gray-scale image.
        Returns:
            npy_kpts: (n_kpts, 6) Keypoints represented as NumPy array.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
        """

        cv_kpts = self.sift.detect(gray_img, None)
        response = np.array([kp.response for kp in cv_kpts])
        resp_sort = np.argsort(response)[::-1][0:self.n_feature].tolist()
        cv_kpts = [cv_kpts[i] for i in resp_sort]
        if self.n_feature > 0 and len(cv_kpts) > self.n_feature:
            cv_kpts = cv_kpts[0:self.n_feature]

        if len(cv_kpts) > 0:
            all_octaves = [np.int8(i.octave & 0xFF) for i in cv_kpts]
            self.first_octave = int(np.min(all_octaves))
            self.max_octave = int(np.max(all_octaves))

            npy_kpts, cv_kpts = self.sample_by_octave(cv_kpts, self.n_sample, self.down_octave)
        else:
            npy_kpts = np.zeros((0, 0))
        return npy_kpts, cv_kpts

    def compute(self, img, cv_kpts):
        """Compute SIFT descriptions on given keypoints.
        Args:
            img: The input image, can be either color or gray-scale.
            cv_kpts: A list of cv2.KeyPoint.
        Returns:
            sift_desc: (n_kpts, 128) SIFT descriptions.
        """

        _, sift_desc = self.sift.compute(img, cv_kpts)
        if self.rootsift:
            sift_desc /= (sift_desc.sum(axis=1, keepdims=True) + 1e-7)
            sift_desc = np.sqrt(sift_desc)
        return sift_desc

    def build_pyramid(self, gray_img):
        """Build pyramid.
        Args:
            gray_img: Input gray-scale image.
        Returns:
            pyr: A list of gaussian blurred images (gaussian scale space).
        """
        if self.pyr_off:
            self.pyr = gray_img
        else:
            sigma = self.sigma
            init_sigma = self.init_sigma
            if self.half_sigma:
                sigma /= 2
                init_sigma /= 2

            gray_img = gray_img.astype(np.float32)
            n_octaves = self.max_octave - self.first_octave + 1
            # create initial image.
            if self.first_octave < 0:
                sig_diff = np.sqrt(np.maximum(
                    np.square(sigma) - np.square(init_sigma) * 4, 0.01))
                base = cv2.resize(gray_img, (gray_img.shape[1] * 2, gray_img.shape[0] * 2),
                                  interpolation=cv2.INTER_LINEAR)
                base = cv2.GaussianBlur(base, None, sig_diff)
            else:
                sig_diff = np.sqrt(np.maximum(np.square(sigma) -
                                              np.square(init_sigma), 0.01))
                base = cv2.GaussianBlur(gray_img, None, sig_diff)
            # compute gaussian kernels.
            sig = np.zeros((self.n_octave_layers + 3,))
            self.pyr = [None] * (n_octaves * (self.n_octave_layers + 3))
            sig[0] = sigma
            k = np.power(2, 1. / self.n_octave_layers)
            for i in range(1, self.n_octave_layers + 3):
                sig_prev = np.power(k, i - 1) * sigma
                sig_total = sig_prev * k
                sig[i] = np.sqrt(sig_total * sig_total - sig_prev * sig_prev)
            # construct gaussian scale space.
            for o in range(0, n_octaves):
                for i in range(0, self.n_octave_layers + 1):
                    if o == 0 and i == 0:
                        dst = base
                    elif i == 0:
                        src = self.pyr[(o - 1) * (self.n_octave_layers + 3) + self.n_octave_layers]
                        dst = cv2.resize(
                            src, (src.shape[1] // 2, src.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
                    else:
                        src = self.pyr[o * (self.n_octave_layers + 3) + i - 1]
                        dst = cv2.GaussianBlur(src, None, sig[i])
                    self.pyr[o * (self.n_octave_layers + 3) + i] = dst

    def unpack_octave(self, kpt):
        """Get scale coefficients of a keypoints.
        Args:
            kpt: A keypoint object represented as cv2.KeyPoint.
        Returns:
            octave: The octave index.
            layer: The level index.
            scale: The sampling step.
        """

        octave = kpt.octave & 255
        layer = (kpt.octave >> 8) & 255
        octave = octave if octave < 128 else (-128 | octave)
        scale = 1. / (1 << octave) if octave >= 0 else float(1 << -octave)
        return octave, layer, scale

    def get_interest_region(self, scale_img, cv_kpts, standardize=True):
        """Get the interest region around a keypoint.
        Args:
            scale_img: DoG image in the scale space.
            cv_kpts: A list of OpenCV keypoints.
            standardize: (True by default) Whether to standardize patches as network inputs.
        Returns:
            Nothing.
        """
        batch_input_grid = []
        all_patches = []
        bs = 30  # limited by OpenCV remap implementation
        for idx, cv_kpt in enumerate(cv_kpts):
            # preprocess
            if self.pyr_off:
                scale = 1
            else:
                _, _, scale = self.unpack_octave(cv_kpt)
            size = cv_kpt.size * scale * 0.5
            ptf = (cv_kpt.pt[0] * scale, cv_kpt.pt[1] * scale)
            ori = 0 if self.ori_off else (360. - cv_kpt.angle) * (np.pi / 180.)
            radius = np.round(self.sift_descr_scl_fctr * size * np.sqrt(2)
                              * (self.sift_descr_width + 1) * 0.5)
            radius = np.minimum(radius, np.sqrt(np.sum(np.square(scale_img.shape))))
            # construct affine transformation matrix.
            affine_mat = np.zeros((3, 2), dtype=np.float32)
            m_cos = np.cos(ori) * radius
            m_sin = np.sin(ori) * radius
            affine_mat[0, 0] = m_cos
            affine_mat[1, 0] = m_sin
            affine_mat[2, 0] = ptf[0]
            affine_mat[0, 1] = -m_sin
            affine_mat[1, 1] = m_cos
            affine_mat[2, 1] = ptf[1]
            # get input grid.
            input_grid = np.matmul(self.output_grid, affine_mat)
            input_grid = np.reshape(input_grid, (-1, 1, 2))
            batch_input_grid.append(input_grid)

            if len(batch_input_grid) != 0 and len(batch_input_grid) % bs == 0 or idx == len(cv_kpts) - 1:
                # sample image pixels.
                batch_input_grid_ = np.concatenate(batch_input_grid, axis=0)
                patches = cv2.remap(scale_img.astype(np.float32), batch_input_grid_,
                                    None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                patches = np.reshape(patches, (len(batch_input_grid),
                                               self.patch_size, self.patch_size))
                # standardize patches.
                if standardize:
                    patches = (patches - np.mean(patches, axis=(1, 2), keepdims=True)) / \
                        (np.std(patches, axis=(1, 2), keepdims=True) + 1e-8)
                all_patches.append(patches)
                batch_input_grid = []
        if len(all_patches) != 0:
            all_patches = np.concatenate(all_patches, axis=0)
        else:
            all_patches = None
        return all_patches

    def get_patches(self, cv_kpts):
        """Get all patches around given keypoints.
        Args:
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
        Return:
            all_patches: (n_kpts, 32, 32) Cropped patches.
        """

        # generate sampling grids.
        n_pixel = np.square(self.patch_size)
        self.output_grid = np.zeros((n_pixel, 3), dtype=np.float32)
        for i in range(n_pixel):
            self.output_grid[i, 0] = (i % self.patch_size) * 1. / self.patch_size * 2 - 1
            self.output_grid[i, 1] = (i // self.patch_size) * 1. / self.patch_size * 2 - 1
            self.output_grid[i, 2] = 1

        if self.pyr_off:
            if not self.down_octave:
                cv_kpts = cv_kpts[::-1]
            all_patches = self.get_interest_region(self.pyr, cv_kpts)
        else:
            scale_index = [[] for i in range(len(self.pyr))]
            for idx, val in enumerate(cv_kpts):
                octave, layer, _ = self.unpack_octave(val)
                scale_val = (int(octave) - self.first_octave) * \
                    (self.n_octave_layers + 3) + int(layer)
                scale_index[scale_val].append(idx)

            all_patches = []
            for idx, val in enumerate(scale_index):
                tmp_cv_kpts = [cv_kpts[i] for i in val]
                scale_img = self.pyr[idx]
                patches = self.get_interest_region(scale_img, tmp_cv_kpts)
                if patches is not None:
                    all_patches.append(patches)
            if self.down_octave:
                all_patches = np.concatenate(all_patches[::-1], axis=0)
            else:
                all_patches = np.concatenate(all_patches, axis=0)

        assert len(cv_kpts) == all_patches.shape[0]
        return all_patches

    def sample_by_octave(self, cv_kpts, n_sample, down_octave=True):
        """Sample keypoints by octave.
        Args:
            cv_kpts: The list of keypoints representd as cv2.KeyPoint.
            n_sample: The sampling number of keypoint. Leave to -1 if no sampling needed
            down_octave: (True by default) Perform sampling downside of octave.
        Returns:
            npy_kpts: (n_kpts, 5) Keypoints in NumPy format, represenetd as
                      (x, y, size, orientation, octave).
            cv_kpts: A list of sampled cv2.KeyPoint.
        """

        n_kpts = len(cv_kpts)
        npy_kpts = np.zeros((n_kpts, 5))
        for idx, val in enumerate(cv_kpts):
            npy_kpts[idx, 0] = val.pt[0]
            npy_kpts[idx, 1] = val.pt[1]
            npy_kpts[idx, 2] = val.size
            npy_kpts[idx, 3] = 0 if self.ori_off else (360. - val.angle) * (np.pi / 180.)
            npy_kpts[idx, 4] = np.int8(val.octave & 0xFF)

        if down_octave:
            sort_idx = (-npy_kpts[:, 2]).argsort()
        else:
            sort_idx = (npy_kpts[:, 2]).argsort()

        npy_kpts = npy_kpts[sort_idx]
        cv_kpts = [cv_kpts[i] for i in sort_idx]

        if n_sample > 0 and n_kpts > n_sample:
            # get the keypoint number in each octave.
            _, unique_counts = np.unique(npy_kpts[:, 4], return_counts=True)

            if down_octave:
                unique_counts = list(reversed(unique_counts))

            n_keep = 0
            for i in unique_counts:
                if n_keep < n_sample:
                    n_keep += i
                else:
                    break
            print('Sampled', n_keep, 'from', n_kpts)
            npy_kpts = npy_kpts[:n_keep]
            cv_kpts = cv_kpts[:n_keep]

        return npy_kpts, cv_kpts


class MatcherWrapper(object):
    """OpenCV matcher wrapper."""

    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def get_matches(self, feat1, feat2, cv_kpts1, cv_kpts2, ratio=None, cross_check=True, err_thld=4, ransac=True, info=''):
        """Compute putative and inlier matches.
        Args:
            feat: (n_kpts, 128) Local features.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
            ratio: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            err_thld: Epipolar error threshold.
            info: Info to print out.
        Returns:
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """

        init_matches1 = self.matcher.knnMatch(feat1, feat2, k=2)
        init_matches2 = self.matcher.knnMatch(feat2, feat1, k=2)

        good_matches = []

        for i in range(len(init_matches1)):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[init_matches1[i][0].trainIdx][0].trainIdx == i
                cond *= cond1
            if ratio is not None:
                cond2 = init_matches1[i][0].distance <= ratio * init_matches1[i][1].distance
                cond *= cond2
            if cond:
                good_matches.append(init_matches1[i][0])

        if type(cv_kpts1) is list and type(cv_kpts2) is list:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx].pt for m in good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx].pt for m in good_matches])
        elif type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx] for m in good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx] for m in good_matches])
        else:
            raise Exception("Keypoint type error!")
            exit(-1)

        if ransac:
            _, mask = cv2.findFundamentalMat(
                good_kpts1, good_kpts2, cv2.RANSAC, err_thld, confidence=0.999)
            n_inlier = np.count_nonzero(mask)
            print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)
        else:
            mask = np.ones((len(good_matches), ))
            print(info, 'n_putative', len(good_matches))
        return good_matches, mask

    def draw_matches(self, img1, cv_kpts1, img2, cv_kpts2, good_matches, mask,
                     match_color=(0, 255, 0), pt_color=(0, 0, 255)):
        """Draw matches."""
        if type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            cv_kpts1 = [cv2.KeyPoint(cv_kpts1[i][0], cv_kpts1[i][1], 1)
                        for i in range(cv_kpts1.shape[0])]
            cv_kpts2 = [cv2.KeyPoint(cv_kpts2[i][0], cv_kpts2[i][1], 1)
                        for i in range(cv_kpts2.shape[0])]
        display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches,
                                  None,
                                  matchColor=match_color,
                                  singlePointColor=pt_color,
                                  matchesMask=mask.ravel().tolist(), flags=4)
        return display
