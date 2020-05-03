#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Inference script.
"""

import os

from queue import Queue
from threading import Thread

import yaml
import cv2
import numpy as np
import progressbar

import tensorflow as tf

from utils.hseq_utils import HSeqUtils
from utils.evaluator import Evaluator
from utils.tf import recoverer
from models import get_model
from models.inference_model import inference

FLAGS = tf.app.flags.FLAGS

# Params for hpatches benchmark
tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def loader(hseq_utils, ori_est, dense_desc, producer_queue):
    for seq_idx in range(hseq_utils.seq_num):
        seq_name, hseq_data = hseq_utils.get_data(seq_idx, ori_est, dense_desc)

        for i in range(6):
            gt_homo = [seq_idx, seq_name] if i == 0 else hseq_data.homo[i]
            producer_queue.put([hseq_data.img[i],
                                hseq_data.kpt_param[i],
                                hseq_data.patch[i],
                                hseq_data.img_feat[i][0],
                                hseq_data.img_feat[i][1],
                                hseq_data.coord[i],
                                gt_homo])
    producer_queue.put(None)


def extractor(patch_queue, sess, output_tensors, config, consumer_queue):
    while True:
        queue_data = patch_queue.get()
        if queue_data is None:
            consumer_queue.put(None)
            return
        img, kpt_param, patch, img_feat, grid_pts, coord, gt_homo = queue_data
        if config['dense_desc']:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            input_dict = {'ph_img:0': np.reshape(gray, (1, gray.shape[0], gray.shape[1], 1))}
        else:
            input_dict = {'ph_patch:0': patch}
        if config['aug'] or config['dense_desc']:
            input_dict['ph_kpt_param:0'] = np.expand_dims(kpt_param, axis=0)
            input_dict['ph_img_feat:0'] = np.expand_dims(img_feat, axis=0)
            input_dict['ph_grid_pts:0'] = np.expand_dims(grid_pts, axis=0)

        output_arrays = sess.run(output_tensors, input_dict)
        local_feat = output_arrays['local_feat']
        consumer_queue.put([local_feat, coord, gt_homo])
        patch_queue.task_done()


def matcher(consumer_queue, sess, evaluator):
    record = []
    while True:
        queue_data = consumer_queue.get()
        if queue_data is None:
            return
        record.append(queue_data)
        if len(record) < 6:
            continue
        ref_feat, ref_coord, seq_info = record[0]

        recall = 0

        for i in range(1, 6):
            test_feat, test_coord, gt_homo = record[i]
            putative_matches = evaluator.feature_matcher(sess, ref_feat, test_feat)
            inlier_matches = evaluator.get_inlier_matches(
                ref_coord, test_coord, putative_matches, gt_homo)
            # Calculate recall
            num_inlier = len(inlier_matches)
            gt_num = evaluator.get_gt_matches(ref_coord, test_coord, gt_homo)
            recall += (num_inlier / max(gt_num, 1)) / 5

        seq_idx = seq_info[0]
        seq_name = os.path.basename(seq_info[1])
        print(seq_idx, seq_name, recall)
        evaluator.stats['all_avg_recall'] += recall
        if seq_name[0] == 'i':
            evaluator.stats['i_avg_recall'] += recall
        if seq_name[0] == 'v':
            evaluator.stats['v_avg_recall'] += recall
        record = []


def prepare_reg_feat(hseq_utils, reg_model, overwrite):
    in_img_path = []
    out_img_feat_list = []
    for seq_name in hseq_utils.seqs:
        for img_idx in range(1, 7):
            img_feat_path = os.path.join(seq_name, '%d_img_feat.npy' % img_idx)
            if not os.path.exists(img_feat_path) or overwrite:
                in_img_path.append(os.path.join(seq_name, '%d.ppm' % img_idx))
                out_img_feat_list.append(img_feat_path)

    if len(in_img_path) > 0:
        model = get_model('reg_model')(reg_model)
        prog_bar = progressbar.ProgressBar()
        prog_bar.max_value = len(in_img_path)
        for idx, val in enumerate(in_img_path):
            img = cv2.imread(val)
            img = img[..., ::-1]
            reg_feat = model.run_test_data(img)
            np.save(out_img_feat_list[idx], reg_feat)
            prog_bar.update(idx)
        model.close()


def hseq_eval():
    with open(FLAGS.config, 'r') as f:
        test_config = yaml.load(f, Loader=yaml.FullLoader)
    # Configure dataset
    hseq_utils = HSeqUtils(test_config['hseq'])
    prepare_reg_feat(hseq_utils, test_config['eval']['reg_model'], test_config['hseq']['overwrite'])
    # Configure evaluation
    evaluator = Evaluator(test_config['eval'])
    # Construct inference networks.
    output_tensors = inference(test_config['network'])
    # Create the initializier.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        # Restore pre-trained model.
        recoverer(sess, test_config['eval']['loc_model'])

        producer_queue = Queue(maxsize=18)
        consumer_queue = Queue()

        producer0 = Thread(target=loader, args=(
            hseq_utils, test_config['network']['ori_est'], test_config['network']['dense_desc'], producer_queue))
        producer0.daemon = True
        producer0.start()

        producer1 = Thread(target=extractor, args=(
            producer_queue, sess, output_tensors, test_config['network'], consumer_queue))
        producer1.daemon = True
        producer1.start()

        consumer = Thread(target=matcher, args=(consumer_queue, sess, evaluator))
        consumer.daemon = True
        consumer.start()

        producer0.join()
        producer1.join()
        consumer.join()

    evaluator.stats['all_avg_recall'] /= max(hseq_utils.seq_num, 1)
    evaluator.stats['i_avg_recall'] /= max(hseq_utils.seq_i_num, 1)
    evaluator.stats['v_avg_recall'] /= max(hseq_utils.seq_v_num, 1)

    print(evaluator.stats)


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    tf.flags.mark_flags_as_required(['config'])
    hseq_eval()


if __name__ == '__main__':
    tf.compat.v1.app.run()
