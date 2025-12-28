"""
* This file is part of PYSLAM
*
* Adpated from https://raw.githubusercontent.com/axelBarroso/Key.Net/master/extract_multiscale_features.py, see the license therein.
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

config.cfg.set_lib("keynet")

import warnings  # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427

warnings.filterwarnings("ignore", category=FutureWarning)

import os, sys, cv2

# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from os import path, mkdir
import argparse
import keyNet.aux.tools as aux
from skimage.transform import pyramid_gaussian
import HSequences_bench.tools.geometry_tools as geo_tools
import HSequences_bench.tools.repeatability_tools as rep_tools
from keyNet.model.keynet_architecture import *
import keyNet.aux.desc_aux_function as loss_desc
from keyNet.model.hardnet_pytorch import *
from keyNet.datasets.dataset_utils import read_bw_image
import torch

from threading import RLock

from pyslam.utilities.tensorflow import set_tf_logging
from pyslam.utilities.logging import Printer, print_options
from .feature_base import BaseFeature2D

import tf_slim as slim

kVerbose = True


def build_keynet_config(keynet_base_path):
    parser = argparse.ArgumentParser(description="HSequences Extract Features")

    # parser.add_argument('--list-images', type=str, help='File containing the image paths for extracting features.',
    #                     required=True)

    # parser.add_argument('--results-dir', type=str, default='extracted_features/',
    #                     help='The output path to save the extracted keypoint.')

    parser.add_argument(
        "--network-version",
        type=str,
        default="KeyNet_default",
        help="The Key.Net network version name",
    )

    parser.add_argument(
        "--checkpoint-det-dir",
        type=str,
        default=keynet_base_path + "keyNet/pretrained_nets/KeyNet_default",
        help="The path to the checkpoint file to load the detector weights.",
    )

    parser.add_argument(
        "--pytorch-hardnet-dir",
        type=str,
        default=keynet_base_path + "keyNet/pretrained_nets/HardNet++.pth",
        help="The path to the checkpoint file to load the HardNet descriptor weights.",
    )

    # Detector Settings

    parser.add_argument(
        "--num-filters", type=int, default=8, help="The number of filters in each learnable block."
    )

    parser.add_argument(
        "--num-learnable-blocks",
        type=int,
        default=3,
        help="The number of learnable blocks after handcrafted block.",
    )

    parser.add_argument(
        "--num-levels-within-net",
        type=int,
        default=3,
        help="The number of pyramid levels inside the architecture.",
    )

    parser.add_argument(
        "--factor-scaling-pyramid",
        type=float,
        default=1.2,
        help="The scale factor between the multi-scale pyramid levels in the architecture.",
    )

    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=5,
        help="The size of the convolutional filters in each of the learnable blocks.",
    )

    # Multi-Scale Extractor Settings

    parser.add_argument(
        "--extract-MS",
        type=bool,
        default=True,
        help="Set to True if you want to extract multi-scale features.",
    )

    parser.add_argument(
        "--num-points", type=int, default=2000, help="The number of desired features to extract."
    )

    parser.add_argument(
        "--nms-size",
        type=int,
        default=15,
        help="The NMS size for computing the validation repeatability.",
    )

    parser.add_argument(
        "--border-size",
        type=int,
        default=15,
        help="The number of pixels to remove from the borders to compute the repeatability.",
    )

    parser.add_argument(
        "--order-coord",
        type=str,
        default="xysr",
        help="The coordinate order that follows the extracted points. Use yxsr or xysr.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=12345,
        help="The random seed value for TensorFlow and Numpy.",
    )

    parser.add_argument(
        "--pyramid_levels",
        type=int,
        default=5,
        help="The number of downsample levels in the pyramid.",
    )

    parser.add_argument(
        "--upsampled-levels",
        type=int,
        default=1,
        help="The number of upsample levels in the pyramid.",
    )

    parser.add_argument(
        "--scale-factor-levels",
        type=float,
        default=np.sqrt(2),
        help="The scale factor between the pyramid levels.",
    )

    parser.add_argument(
        "--scale-factor",
        type=float,
        default=2.0,
        help="The scale factor to extract patches before descriptor.",
    )

    # GPU Settings

    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.3,
        help="The fraction of GPU used by the script.",
    )

    parser.add_argument(
        "--gpu-visible-devices", type=str, default="0", help="Set CUDA_VISIBLE_DEVICES variable."
    )

    args = parser.parse_known_args()[0]

    # remove verbose bits from tf
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Set CUDA GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices

    print("Using KeyNet version:" + args.network_version)

    if not args.extract_MS:
        args.pyramid_levels = 0
        args.upsampled_levels = 0

    return args


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, sizes, levels):
    assert len(pts) == len(scores)
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        kps = [
            cv2.KeyPoint(p[0], p[1], size=sizes[i], response=scores[i], octave=levels[i])
            for i, p in enumerate(pts)
        ]
    return kps


# Interface for pySLAM
class KeyNetDescFeature2D(BaseFeature2D):
    def __init__(
        self,
        num_features=2000,
        num_levels=5,  # The number of downsample levels in the pyramid.
        scale_factor=2,  # The scale factor to extract patches before descriptor.
        scale_factor_levels=np.sqrt(2),  # The scale factor between the pyramid levels.
        do_cuda=True,
        do_tf_logging=False,
    ):
        print("Using KeyNetDescFeature2D")
        self.lock = RLock()
        self.model_base_path = config.cfg.root_folder + "/thirdparty/keynet/"

        set_tf_logging(do_tf_logging)

        self.do_cuda = do_cuda & torch.cuda.is_available()
        print("cuda:", self.do_cuda)
        device = torch.device("cuda:0" if self.do_cuda else "cpu")

        self.session = None

        self.keypoint_size = 8  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint

        self.pts = []
        self.kps = []
        self.des = []
        self.scales = []
        self.scores = []
        self.frame = None

        keynet_config = build_keynet_config(self.model_base_path)
        self.keynet_config = keynet_config
        keynet_config.num_points = num_features
        keynet_config.pyramid_levels = num_levels
        keynet_config.scale_factor = scale_factor
        keynet_config.scale_factor_levels = scale_factor_levels

        print_options(self.keynet_config, "KEYNET CONFIG")

        print("==> Loading pre-trained network.")
        self.load_model()
        print("==> Successfully loaded pre-trained network.")

    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.keynet_config.num_points = num_features

    @property
    def num_features(self):
        return self.keynet_config.num_points

    @property
    def num_levels(self):
        return self.keynet_config.pyramid_levels

    @property
    def scale_factor(self):
        return self.keynet_config.scale_factor

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def close(self):
        if self.session is not None:
            print("KEYNET: closing tf session")
            self.session.close()
            tf.reset_default_graph()

    def load_model(self):
        # Create graph before session :)
        self.graph = tf.Graph().as_default()

        # GPU Usage
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = (
            self.keynet_config.gpu_memory_fraction
        )
        tf_config.gpu_options.allow_growth = True

        # with tf.Session(config=config) as sess:
        self.session = tf.Session(config=tf_config)

        tfv2.random.set_seed(self.keynet_config.random_seed)

        with tf.name_scope("inputs"):

            # Define the input tensor shape
            tensor_input_shape = (None, None, None, 1)

            tf.compat.v1.disable_eager_execution()
            self.input_network = tf.placeholder(
                dtype=tf.float32, shape=tensor_input_shape, name="input_network"
            )
            self.dimension_image = tf.placeholder(
                dtype=tf.int32, shape=(3,), name="dimension_image"
            )
            self.kpts_coord = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="kpts_coord")
            self.kpts_batch = tf.placeholder(dtype=tf.int32, shape=(None,), name="kpts_batch")
            self.kpts_scale = tf.placeholder(dtype=tf.float32, name="kpts_scale")
            self.phase_train = tf.placeholder(tf.bool, name="phase_train")

        with tf.name_scope("model_deep_detector"):

            deep_architecture = keynet(self.keynet_config)
            output_network = deep_architecture.model(
                self.input_network, self.phase_train, self.dimension_image, reuse=False
            )
            self.maps = tf.nn.relu(output_network["output"])

        # Extract Patches from inputs:
        self.input_patches = loss_desc.build_patch_extraction(
            self.kpts_coord, self.kpts_batch, self.input_network, kpts_scale=self.kpts_scale
        )

        # Define Pytorch HardNet
        self.model = HardNet()
        checkpoint = torch.load(self.keynet_config.pytorch_hardnet_dir)
        self.model.load_state_dict(checkpoint["state_dict"])
        if self.do_cuda:
            self.model.cuda()
            print("Extracting torch model on GPU")
        else:
            print("Extracting torch model on CPU")
            self.model = self.model.cpu()
        self.model.eval()

        # Define variables
        detect_var = [v for v in tf.trainable_variables(scope="model_deep_detector")]

        if os.listdir(self.keynet_config.checkpoint_det_dir):
            init_assign_op_det, init_feed_dict_det = slim.assign_from_checkpoint(
                tf.train.latest_checkpoint(self.keynet_config.checkpoint_det_dir), detect_var
            )

        point_level = []
        tmp = 0.0
        factor_points = self.keynet_config.scale_factor_levels**2
        self.levels = self.keynet_config.pyramid_levels + self.keynet_config.upsampled_levels + 1
        # print('levels: ', [i for i in range(self.levels)])
        for idx_level in range(self.levels):
            tmp += factor_points ** (-1 * (idx_level - self.keynet_config.upsampled_levels))
            point_level.append(
                self.keynet_config.num_points
                * factor_points ** (-1 * (idx_level - self.keynet_config.upsampled_levels))
            )

        self.point_level = np.asarray(list(map(lambda x: int(x / tmp), point_level)))
        # print('self.point_level:',self.point_level)

        self.session.run(tf.global_variables_initializer())

        if os.listdir(self.keynet_config.checkpoint_det_dir):
            self.session.run(init_assign_op_det, init_feed_dict_det)

    def extract_keynet_features(self, image):
        pyramid = pyramid_gaussian(
            image,
            max_layer=self.keynet_config.pyramid_levels,
            downscale=self.keynet_config.scale_factor_levels,
        )

        score_maps = {}
        for j, resized in enumerate(pyramid):
            im = resized.reshape(1, resized.shape[0], resized.shape[1], 1)

            feed_dict = {
                self.input_network: im,
                self.phase_train: False,
                self.dimension_image: np.array([1, im.shape[1], im.shape[2]], dtype=np.int32),
            }

            im_scores = self.session.run(self.maps, feed_dict=feed_dict)

            im_scores = geo_tools.remove_borders(im_scores, borders=self.keynet_config.border_size)
            score_maps["map_" + str(j + 1 + self.keynet_config.upsampled_levels)] = im_scores[
                0, :, :, 0
            ]

        if self.keynet_config.upsampled_levels:
            for j in range(self.keynet_config.upsampled_levels):
                factor = self.keynet_config.scale_factor_levels ** (
                    self.keynet_config.upsampled_levels - j
                )
                up_image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

                im = np.reshape(up_image, (1, up_image.shape[0], up_image.shape[1], 1))

                feed_dict = {
                    self.input_network: im,
                    self.phase_train: False,
                    self.dimension_image: np.array([1, im.shape[1], im.shape[2]], dtype=np.int32),
                }

                im_scores = self.session.run(self.maps, feed_dict=feed_dict)

                im_scores = geo_tools.remove_borders(
                    im_scores, borders=self.keynet_config.border_size
                )
                score_maps["map_" + str(j + 1)] = im_scores[0, :, :, 0]

        im_pts = []
        im_pts_levels = []
        for idx_level in range(self.levels):

            scale_value = self.keynet_config.scale_factor_levels ** (
                idx_level - self.keynet_config.upsampled_levels
            )
            scale_factor = 1.0 / scale_value

            h_scale = np.asarray(
                [[scale_factor, 0.0, 0.0], [0.0, scale_factor, 0.0], [0.0, 0.0, 1.0]]
            )
            h_scale_inv = np.linalg.inv(h_scale)
            h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

            num_points_level = self.point_level[idx_level]
            # print('num_points_level:',num_points_level)
            if idx_level > 0:
                res_points = int(
                    np.asarray([self.point_level[a] for a in range(0, idx_level + 1)]).sum()
                    - len(im_pts)
                )
                num_points_level = res_points

            im_scores = rep_tools.apply_nms(
                score_maps["map_" + str(idx_level + 1)], self.keynet_config.nms_size
            )
            im_pts_tmp = geo_tools.get_point_coordinates(
                im_scores, num_points=num_points_level, order_coord="xysr"
            )

            im_pts_tmp = geo_tools.apply_homography_to_points(im_pts_tmp, h_scale_inv)

            if not idx_level:
                im_pts = im_pts_tmp
            else:
                im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)

            im_pts_levels_tmp = np.ones(len(im_pts), dtype=np.int32) * idx_level
            im_pts_levels = np.concatenate((im_pts_levels, im_pts_levels_tmp), axis=0).astype(
                np.int32
            )

        if self.keynet_config.order_coord == "yxsr":
            im_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], im_pts)))

        sorted_idxs = (-1 * im_pts[:, 3]).argsort()  # sort points with their scores
        im_pts = im_pts[sorted_idxs]
        im_pts_levels = im_pts_levels[sorted_idxs]
        # print('im_pts_levels:',im_pts_levels)
        im_pts = im_pts[: self.keynet_config.num_points]
        im_pts_levels = im_pts_levels[: self.keynet_config.num_points]

        # Extract descriptor from features
        descriptors = []
        im = image.reshape(1, image.shape[0], image.shape[1], 1)
        for idx_desc_batch in range(int(len(im_pts) / 250 + 1)):
            points_batch = im_pts[idx_desc_batch * 250 : (idx_desc_batch + 1) * 250]

            if not len(points_batch):
                break

            feed_dict = {
                self.input_network: im,
                self.phase_train: False,
                self.kpts_coord: points_batch[:, :2],
                self.kpts_scale: self.keynet_config.scale_factor * points_batch[:, 2],
                self.kpts_batch: np.zeros(len(points_batch)),
                self.dimension_image: np.array([1, im.shape[1], im.shape[2]], dtype=np.int32),
            }

            patch_batch = self.session.run(self.input_patches, feed_dict=feed_dict)
            patch_batch = np.reshape(patch_batch, (patch_batch.shape[0], 1, 32, 32))
            data_a = torch.from_numpy(patch_batch)
            data_a = data_a.cuda()
            data_a = Variable(data_a)
            with torch.no_grad():
                out_a = self.model(data_a)
            desc_batch = out_a.data.cpu().numpy().reshape(-1, 128)
            if idx_desc_batch == 0:
                descriptors = desc_batch
            else:
                descriptors = np.concatenate([descriptors, desc_batch], axis=0)

        return im_pts, descriptors, im_pts_levels

    def compute_kps_des(self, im):
        with self.lock:

            im = im.astype(float) / im.max()
            im_pts, descriptors, im_pts_levels = self.extract_keynet_features(im)

            self.pts = im_pts[:, :2]
            scales = im_pts[:, 2]
            scores = im_pts[:, 3]
            pts_levels = im_pts_levels

            # print('scales:',self.scales)

            self.kps = convert_pts_to_keypoints(
                self.pts, scores, scales * self.keypoint_size, pts_levels
            )

            return self.kps, descriptors

    def detectAndCompute(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            self.frame = frame
            self.kps, self.des = self.compute_kps_des(frame)
            if kVerbose:
                print(
                    "detector: KEYNET, descriptor: KEYNET, #features: ",
                    len(self.kps),
                    ", frame res: ",
                    frame.shape[0:2],
                )
            return self.kps, self.des

    # return keypoints if available otherwise call detectAndCompute()
    def detect(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            # if self.frame is not frame:
            self.detectAndCompute(frame)
            return self.kps

    # return descriptors if available otherwise call detectAndCompute()
    def compute(self, frame, kps=None, mask=None):  # kps is a fake input, mask is a fake input
        with self.lock:
            if self.frame is not frame:
                Printer.orange(
                    "WARNING: KEYNET is recomputing both kps and des on last input frame",
                    frame.shape,
                )
                self.detectAndCompute(frame)
            return self.kps, self.des

    # return descriptors if available otherwise call detectAndCompute()
    def compute(self, frame, kps=None, mask=None):  # kps is a fake input, mask is a fake input
        with self.lock:
            if self.frame is not frame:
                # Printer.orange('WARNING: KEYNET is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, self.des
