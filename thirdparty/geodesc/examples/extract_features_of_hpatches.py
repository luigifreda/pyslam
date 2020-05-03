#!/usr/bin/env python
"""
Copyright 2018, Zixin Luo, HKUST.
Feature extraction of HPatches dataset.
"""

from __future__ import print_function

import os
import sys

import cv2
import numpy as np
import tensorflow as tf

CURDIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(CURDIR, '..')))

from utils.tf import load_frozen_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_path', '../model/geodesc.pb',
                           """Path to evaluation model.""")
tf.app.flags.DEFINE_string('hpatches_root', None,
                           """Path to HPatches dataset root.""")
tf.app.flags.DEFINE_string('feat_out_path', None,
                           """Path to feature output folder.""")
tf.app.flags.DEFINE_string('hpatches_seq_list', os.path.join(CURDIR, 'hpatches_seq_list.txt'),
                           """Path to HPatches sequence list list.""")


def load_seq(seq_path):
    """Load HPatches sequences."""
    seq = cv2.imread(seq_path, 0)
    n_patch = seq.shape[0] / 65
    seq = np.reshape(seq, (n_patch, 65, 65, 1)).astype(np.float32)
    resized_seq = np.zeros((n_patch, 32, 32), np.float32)

    for i in range(n_patch):
        tmp_patch = cv2.resize(seq[i], (32, 32))
        resized_seq[i] = (tmp_patch - np.mean(tmp_patch)) / (np.std(tmp_patch) + 1e-8)

    resized_seq = np.expand_dims(resized_seq, axis=-1)
    return resized_seq


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    hpatches_seq_list = open(FLAGS.hpatches_seq_list).read().splitlines()
    n_seq = len(hpatches_seq_list)

    graph = load_frozen_model(FLAGS.model_path, print_nodes=False)
    with tf.Session(graph=graph) as sess:
        for i in range(n_seq):
            if i % 16 == 0:
                print(i, '/', n_seq)
            strs = hpatches_seq_list[i].split('/')
            save_folder = os.path.join(FLAGS.feat_out_path, 'geodesc', strs[-2])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            seq_data = load_seq(os.path.join(FLAGS.hpatches_root, hpatches_seq_list[i]))
            feat = sess.run("squeeze_1:0", feed_dict={"input:0": seq_data})
            csv_path = os.path.join(save_folder, os.path.splitext(strs[-1])[0] + '.csv')
            np.savetxt(csv_path, feat, delimiter=",", fmt='%.8f')


if __name__ == '__main__':
    tf.flags.mark_flags_as_required(['hpatches_root', 'feat_out_path'])
    tf.app.run()
