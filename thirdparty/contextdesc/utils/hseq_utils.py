#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
HSequence evaluation tools.
"""

import os
import glob
import pickle
import random
import cv2
import numpy as np

from utils.opencvhelper import SiftWrapper


class HSeqData(object):
    def __init__(self):
        self.img = []
        self.patch = []
        self.kpt_param = []
        self.coord = []
        self.homo = []
        self.img_feat = []


class HSeqUtils(object):
    def __init__(self, config):
        self.seqs = []
        self.seq_i_num = 0
        self.seq_v_num = 0

        seq_types = ['%s_*' % i for i in config['seq']]
        for files in seq_types:
            tmp_seqs = glob.glob(os.path.join(config['root'], files))
            tmp_seqs.sort()
            if files[0] == 'i':
                self.seq_i_num = len(tmp_seqs)
            if files[0] == 'v':
                self.seq_v_num = len(tmp_seqs)
            self.seqs.extend(tmp_seqs)
        self.seqs = self.seqs[config['start_idx']:]
        self.seq_num = len(self.seqs)
        self.suffix = config['suffix']
        # for detector config
        self.upright = config['upright']
        # for data parsing
        self.sample_num = config['kpt_n']
        self.patch_scale = 6

    def get_data(self, seq_idx, ori_est, dense_desc):
        random.seed(0)
        if self.suffix is None:
            sift_wrapper = SiftWrapper(n_feature=self.sample_num, peak_thld=0.04)
            sift_wrapper.ori_off = self.upright
            sift_wrapper.create()

        hseq_data = HSeqData()
        seq_name = self.seqs[seq_idx]

        for img_idx in range(1, 7):
            # read image features.
            img_feat = np.load(os.path.join(seq_name, '%d_img_feat.npy' % img_idx))
            rows = img_feat.shape[0]
            cols = img_feat.shape[1]
            x_rng = np.linspace(-1., 1., cols)
            y_rng = np.linspace(-1., 1., rows)
            xv, yv = np.meshgrid(x_rng, y_rng)
            grid_pts = np.stack((xv, yv), axis=-1)
            # read images.
            img = cv2.imread(os.path.join(seq_name, '%d.ppm' % img_idx))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = img.shape

            if self.suffix is None:
                npy_kpts, cv_kpts = sift_wrapper.detect(gray)
                if not dense_desc:
                    sift_wrapper.build_pyramid(gray)
                    patches = sift_wrapper.get_patches(cv_kpts)
                else:
                    patches = None
            else:
                with open(os.path.join(seq_name, ('%d' + self.suffix + '.pkl') % img_idx), 'rb') as handle:
                    data_dict = pickle.load(handle, encoding='latin1')
                npy_kpts = data_dict['npy_kpts']
                if not dense_desc:
                    patches = data_dict['patches']
                else:
                    patches = None

            kpt_num = npy_kpts.shape[0]

            # Sample keypoints
            if self.sample_num > 0 and kpt_num > self.sample_num:
                sample_idx = random.sample(range(kpt_num), self.sample_num)
            else:
                sample_idx = range(kpt_num)
            # Apply sampling.
            npy_kpts = npy_kpts[sample_idx]
            if patches is not None:
                patches = patches[sample_idx]
            kpt_num = npy_kpts.shape[0]

            # compose affine crop matrix.
            crop_mat = np.zeros((kpt_num, 6))
            if ori_est:
                # no initial orientation.
                m_cos = np.ones_like(npy_kpts[:, 2]) * self.patch_scale * npy_kpts[:, 2]
                m_sin = np.zeros_like(npy_kpts[:, 2]) * self.patch_scale * npy_kpts[:, 2]
            else:
                # rely on the SIFT orientation estimation.
                m_cos = np.cos(-npy_kpts[:, 3]) * self.patch_scale * npy_kpts[:, 2]
                m_sin = np.sin(-npy_kpts[:, 3]) * self.patch_scale * npy_kpts[:, 2]
            crop_mat[:, 0] = m_cos / float(img_size[1])
            crop_mat[:, 1] = m_sin / float(img_size[1])
            crop_mat[:, 2] = (npy_kpts[:, 0] - img_size[1] / 2.) / (img_size[1] / 2.)
            crop_mat[:, 3] = -m_sin / float(img_size[0])
            crop_mat[:, 4] = m_cos / float(img_size[0])
            crop_mat[:, 5] = (npy_kpts[:, 1] - img_size[0] / 2.) / (img_size[0] / 2.)
            npy_kpts = npy_kpts[:, 0:2]

            # read homography matrix.
            if img_idx > 1:
                homo_mat = open(os.path.join(seq_name, 'H_1_%d' % img_idx)).read().splitlines()
                homo_mat = np.array([float(i) for i in ' '.join(homo_mat).split()])
                homo_mat = np.reshape(homo_mat, (3, 3))
            else:
                homo_mat = None

            hseq_data.img.append(img)
            hseq_data.kpt_param.append(crop_mat)
            hseq_data.patch.append(patches)
            hseq_data.coord.append(npy_kpts)
            hseq_data.homo.append(homo_mat)
            hseq_data.img_feat.append((img_feat, grid_pts))

        return seq_name, hseq_data
