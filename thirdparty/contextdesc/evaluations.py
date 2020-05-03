#!/usr/bin/env python3

"""
Copyright 2019, Zixin Luo, HKUST.
Evaluation script.
"""

import os
import yaml

import h5py
import numpy as np
import tensorflow as tf
import progressbar

from datasets import get_dataset
from models import get_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def extract_reg_feat(config):
    """Extract regional features."""
    prog_bar = progressbar.ProgressBar()
    config['stage'] = 'reg'
    dataset = get_dataset(config['data_name'])(**config)
    prog_bar.max_value = dataset.data_length
    test_set = dataset.get_test_set()

    model = get_model('reg_model')(config['pretrained']['reg_model'], **(config['reg_feat']))
    idx = 0
    while True:
        try:
            data = next(test_set)
            dump_path = data['dump_path'].decode('utf-8')
            reg_f = h5py.File(dump_path, 'a')
            if 'reg_feat' not in reg_f or config['reg_feat']['overwrite']:
                reg_feat = model.run_test_data(data['image'])
                if 'reg_feat' in reg_f:
                    del reg_f['reg_feat']
                _ = reg_f.create_dataset('reg_feat', data=reg_feat)
            prog_bar.update(idx)
            idx += 1
        except dataset.end_set:
            break
    model.close()


def extract_loc_feat(config):
    """Extract local features."""
    prog_bar = progressbar.ProgressBar()
    config['stage'] = 'loc'
    dataset = get_dataset(config['data_name'])(**config)
    prog_bar.max_value = dataset.data_length
    test_set = dataset.get_test_set()

    model = get_model('loc_model')(config['pretrained']['loc_model'], **(config['loc_feat']))
    idx = 0
    while True:
        try:
            data = next(test_set)
            dump_path = data['dump_path'].decode('utf-8')
            loc_f = h5py.File(dump_path, 'a')
            if 'loc_info' not in loc_f and 'kpt' not in loc_f or config['loc_feat']['overwrite']:
                # detect SIFT keypoints and crop image patches.
                loc_feat, kpt_mb, npy_kpts, cv_kpts, _ = model.run_test_data(data['image'])
                loc_info = np.concatenate((npy_kpts, loc_feat, kpt_mb), axis=-1)
                raw_kpts = [np.array((i.pt[0], i.pt[1], i.size, i.angle, i.response))
                            for i in cv_kpts]
                raw_kpts = np.stack(raw_kpts, axis=0)
                loc_info = np.concatenate((raw_kpts, loc_info), axis=-1)
                if 'loc_info' in loc_f or 'kpt' in loc_f:
                    del loc_f['loc_info']
                _ = loc_f.create_dataset('loc_info', data=loc_info)
            prog_bar.update(idx)
            idx += 1
        except dataset.end_set:
            break
    model.close()


def extract_aug_feat(config):
    """Extract augmented features."""
    prog_bar = progressbar.ProgressBar()
    config['stage'] = 'aug'
    dataset = get_dataset(config['data_name'])(**config)
    prog_bar.max_value = dataset.data_length
    test_set = dataset.get_test_set()

    model = get_model('aug_model')(config['pretrained']['loc_model'], **(config['aug_feat']))
    idx = 0
    while True:
        try:
            data = next(test_set)
            dump_path = data['dump_path'].decode('utf-8')
            aug_f = h5py.File(dump_path, 'a')
            if 'aug_feat' not in aug_f or config['aug_feat']['overwrite']:
                aug_feat, _ = model.run_test_data(data['dump_data'])
                if 'aug_feat' in aug_f:
                    del aug_f['aug_feat']
                if aug_feat.dtype == np.uint8:
                    _ = aug_f.create_dataset('aug_feat', data=aug_feat, dtype='uint8')
                else:
                    _ = aug_f.create_dataset('aug_feat', data=aug_feat)
            prog_bar.update(idx)
            idx += 1
        except dataset.end_set:
            break
    model.close()


def format_data(config):
    """Post-processing and generate custom files."""
    prog_bar = progressbar.ProgressBar()
    config['stage'] = 'post_format'
    dataset = get_dataset(config['data_name'])(**config)
    prog_bar.max_value = dataset.data_length
    test_set = dataset.get_test_set()

    idx = 0
    while True:
        try:
            data = next(test_set)
            dataset.format_data(data)
            prog_bar.update(idx)
            idx += 1
        except dataset.end_set:
            break


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)
    if not os.path.exists(config['dump_root']):
        os.mkdir(config['dump_root'])
    # extract regional features.
    if config['reg_feat']['infer']:
        extract_reg_feat(config)
    # extract local features and keypoint matchability.
    if config['loc_feat']['infer']:
        extract_loc_feat(config)
    # extract augmented features.
    if config['aug_feat']['infer']:
        extract_aug_feat(config)
    if config['post_format']['enable']:
        format_data(config)


if __name__ == '__main__':
    tf.flags.mark_flags_as_required(['config'])
    tf.app.run()
