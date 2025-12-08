"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import pyslam.config as config

config.cfg.set_lib("geodesc", prepend=True)

import os
import sys
import time

from threading import Thread
from multiprocessing import Queue

from .feature_base import BaseFeature2D
from pyslam.utilities.tensorflow import load_frozen_model, set_tf_logging, import_tf_compat_v1
from pyslam.utilities.features import (
    extract_patches_tensor,
    extract_patches_array,
    extract_patches_array_cpp,
)

import cv2
import numpy as np

import warnings  # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427

warnings.filterwarnings("ignore", category=FutureWarning)

if False:
    import tensorflow as tf
else:
    tf = import_tf_compat_v1()

# from https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96 to cope with the following error:
# "[...tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# then you must use the config as follows:
# session = tf.Session(config=tf_config, ...)


kVerbose = True


# Interface for pySLAM
class GeodescFeature2D(BaseFeature2D):
    quantize = False  # quantize or not output features: if you set this to True then you have a binary descriptor

    def __init__(self, do_tf_logging=False):
        print("Using GeodescFeature2D")
        # mag_factor is how many times the original keypoint scale
        # is enlarged to generate a patch from a keypoint
        self.mag_factor = 3

        # inference batch size
        self.batch_size = 512
        self.process_all = True  # process all the patches at once

        self.model_base_path = config.cfg.root_folder + "/thirdparty/geodesc/"
        self.model_path = self.model_base_path + "model/geodesc.pb"

        set_tf_logging(do_tf_logging)

        print("==> Loading pre-trained network.")
        # create deep feature extractor.
        self.graph = load_frozen_model(self.model_path, print_nodes=False)
        # sess = tf.Session(graph=graph)
        print("==> Successfully loaded pre-trained network.")

    def process_patches(self, patches):
        num_patch = patches.shape[0]
        if num_patch % self.batch_size > 0:
            loop_num = int(np.floor(float(num_patch) / float(self.batch_size)))
        else:
            loop_num = int(num_patch / self.batch_size - 1)

        with tf.Session(graph=self.graph, config=tf_config) as sess:

            def _worker(patch_queue, sess, des):
                """The worker thread."""
                while True:
                    patch_data = patch_queue.get()
                    if patch_data is None:
                        return
                    feat = sess.run(
                        "squeeze_1:0", feed_dict={"input:0": np.expand_dims(patch_data, -1)}
                    )
                    des.append(feat)

            des = []
            patch_queue = Queue()
            worker_thread = Thread(target=_worker, args=(patch_queue, sess, des))
            worker_thread.daemon = True
            worker_thread.start()

            start = time.time()

            # enqueue
            if not self.process_all:
                for i in range(loop_num + 1):
                    if i < loop_num:
                        patch_queue.put(patches[i * self.batch_size : (i + 1) * self.batch_size])
                    else:
                        patch_queue.put(patches[i * self.batch_size :])
            else:
                patch_queue.put(patches)
            # poison pill
            patch_queue.put(None)
            # wait for extraction.
            worker_thread.join()

            end = time.time()
            if kVerbose:
                print("Time cost in feature extraction", end - start)

            des = np.concatenate(des, axis=0)
            # quantize output features
            des = (des * 128 + 128).astype(np.uint8) if self.quantize else des
            return des

    def compute(self, frame, kps, mask=None):  # mask is a fake input
        # print('kps: ', kps)
        if len(kps) > 0:
            if False:
                # use python code
                patches = extract_patches_array(
                    frame, kps, patch_size=32, mag_factor=self.mag_factor
                )
            else:
                # use faster cpp code
                patches = extract_patches_array_cpp(
                    frame, kps, patch_size=32, mag_factor=self.mag_factor
                )
            patches = np.asarray(patches)
            des = self.process_patches(patches)
        else:
            des = []
        if kVerbose:
            print("descriptor: GEODESC, #features: ", len(kps), ", frame res: ", frame.shape[0:2])
        return kps, des
