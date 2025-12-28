"""
* This file is part of PYSLAM
* Adapted from https://github.com/tensorflow/models/blob/master/research/delf/delf/python/examples/extract_features.py, see the license therein.
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

config.cfg.set_lib("delf")

import cv2

from threading import RLock

from pyslam.utilities.logging import Printer
from pyslam.utilities.system import import_from, is_opencv_version_greater_equal
from pyslam.utilities.tensorflow import set_tf_logging


from .feature_base import BaseFeature2D

import warnings  # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from pyslam.utilities.tensorflow import import_tf_compat_v1

# Import TensorFlow using the unified function
tf = import_tf_compat_v1()

from google.protobuf import text_format

# # Try to import app from tensorflow, handle import errors gracefully
# try:
#     from tensorflow.python.platform import app
# except (ModuleNotFoundError, ImportError, AttributeError):
#     # app might not be available in all TensorFlow versions/platforms
#     # If it's not used, we can safely ignore the import
#     app = None
#     print("WARNING: tensorflow.python.platform.app is not available, using None instead")

# from delf import delf_config_pb2
# from delf import feature_extractor
# from delf import feature_io
# from delf.protos import aggregation_config_pb2
# from delf.protos import box_pb2
# from delf.protos import datum_pb2
# from delf.protos import delf_config_pb2
# from delf.protos import feature_pb2
# from delf.python import box_io
# from delf.python import datum_io
# #from delf.python import delf_v1
# from delf.python import feature_aggregation_extractor
# from delf.python import feature_aggregation_similarity
# from delf.python import feature_extractor
# from delf.python import feature_io
# from delf.python.examples import detector
# from delf.python.examples import extractor
# from delf.python import detect_to_retrieve
# #from delf.python import google_landmarks_dataset

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import feature_io
from delf import utils
from delf import extractor
from delf import datum_io
from delf import feature_extractor


delf_base_path = config.cfg.root_folder + "/thirdparty/tensorflow_models/research/delf/delf/python/"
delf_config_file = delf_base_path + "examples/delf_config_example.pbtxt"
delf_model_path = delf_base_path + "examples/parameters/delf_gld_20190411/model/"
delf_mean_path = delf_base_path + "examples/parameters/delf_gld_20190411/pca/mean.datum"
delf_projection_matrix_path = (
    delf_base_path + "examples/parameters/delf_gld_20190411/pca/pca_proj_mat.datum"
)

kVerbose = True


# Minimum dimensions below which features are not extracted (empty
# features are returned). This applies after any resizing is performed.
_MIN_HEIGHT = 10
_MIN_WIDTH = 10


# adapted from thirdparty/tensorflow_models/research/delf/delf/python/examples/extractor.py
def MakeExtractor(config):
    """Creates a function to extract global and/or local features from an image.

    Args:
      config: DelfConfig proto containing the model configuration.

    Returns:
      Function that receives an image and returns features.

    Raises:
      ValueError: if config is invalid.
    """
    # Assert the configuration.
    if not config.use_local_features and not config.use_global_features:
        raise ValueError(
            "Invalid config: at least one of "
            "{use_local_features, use_global_features} must be True"
        )

    # Load model.
    model = tf.saved_model.load(config.model_path)

    # Input image scales to use for extraction.
    image_scales_tensor = tf.convert_to_tensor(list(config.image_scales))

    # Input (feeds) and output (fetches) end-points. These are only needed when
    # using a model that was exported using TF1.
    feeds = ["input_image:0", "input_scales:0"]
    fetches = []

    # Custom configuration needed when local features are used.
    if config.use_local_features:
        # Extra input/output end-points/tensors.
        feeds.append("input_abs_thres:0")
        feeds.append("input_max_feature_num:0")
        fetches.append("boxes:0")
        fetches.append("features:0")
        fetches.append("scales:0")
        fetches.append("scores:0")
        score_threshold_tensor = tf.constant(config.delf_local_config.score_threshold)
        max_feature_num_tensor = tf.constant(config.delf_local_config.max_feature_num)

        # If using PCA, pre-load required parameters.
        local_pca_parameters = {}
        if config.delf_local_config.use_pca:
            local_pca_parameters["mean"] = tf.constant(
                datum_io.ReadFromFile(config.delf_local_config.pca_parameters.mean_path),
                dtype=tf.float32,
            )
            local_pca_parameters["matrix"] = tf.constant(
                datum_io.ReadFromFile(
                    config.delf_local_config.pca_parameters.projection_matrix_path
                ),
                dtype=tf.float32,
            )
            local_pca_parameters["dim"] = config.delf_local_config.pca_parameters.pca_dim
            local_pca_parameters["use_whitening"] = (
                config.delf_local_config.pca_parameters.use_whitening
            )
            if config.delf_local_config.pca_parameters.use_whitening:
                local_pca_parameters["variances"] = tf.squeeze(
                    tf.constant(
                        datum_io.ReadFromFile(
                            config.delf_local_config.pca_parameters.pca_variances_path
                        ),
                        dtype=tf.float32,
                    )
                )
            else:
                local_pca_parameters["variances"] = None

    # Custom configuration needed when global features are used.
    if config.use_global_features:
        # Extra input/output end-points/tensors.
        feeds.append("input_global_scales_ind:0")
        fetches.append("global_descriptors:0")
        if config.delf_global_config.image_scales_ind:
            global_scales_ind_tensor = tf.constant(list(config.delf_global_config.image_scales_ind))
        else:
            global_scales_ind_tensor = tf.range(len(config.image_scales))

        # If using PCA, pre-load required parameters.
        global_pca_parameters = {}
        if config.delf_global_config.use_pca:
            global_pca_parameters["mean"] = tf.constant(
                datum_io.ReadFromFile(config.delf_global_config.pca_parameters.mean_path),
                dtype=tf.float32,
            )
            global_pca_parameters["matrix"] = tf.constant(
                datum_io.ReadFromFile(
                    config.delf_global_config.pca_parameters.projection_matrix_path
                ),
                dtype=tf.float32,
            )
            global_pca_parameters["dim"] = config.delf_global_config.pca_parameters.pca_dim
            global_pca_parameters["use_whitening"] = (
                config.delf_global_config.pca_parameters.use_whitening
            )
            if config.delf_global_config.pca_parameters.use_whitening:
                global_pca_parameters["variances"] = tf.squeeze(
                    tf.constant(
                        datum_io.ReadFromFile(
                            config.delf_global_config.pca_parameters.pca_variances_path
                        ),
                        dtype=tf.float32,
                    )
                )
            else:
                global_pca_parameters["variances"] = None

    if not hasattr(config, "is_tf2_exported") or not config.is_tf2_exported:
        model = model.prune(feeds=feeds, fetches=fetches)

    def ExtractorFn(image, resize_factor=1.0):
        """Receives an image and returns DELF global and/or local features.

        If image is too small, returns empty features.

        Args:
          image: Uint8 array with shape (height, width, 3) containing the RGB image.
          resize_factor: Optional float resize factor for the input image. If given,
            the maximum and minimum allowed image sizes in the config are scaled by
            this factor.

        Returns:
          extracted_features: A dict containing the extracted global descriptors
            (key 'global_descriptor' mapping to a [D] float array), and/or local
            features (key 'local_features' mapping to a dict with keys 'locations',
            'descriptors', 'scales', 'attention').
        """
        resized_image, scale_factors = utils.ResizeImage(image, config, resize_factor=resize_factor)

        # If the image is too small, returns empty features.
        if resized_image.shape[0] < _MIN_HEIGHT or resized_image.shape[1] < _MIN_WIDTH:
            extracted_features = {"global_descriptor": np.array([])}
            if config.use_local_features:
                extracted_features.update(
                    {
                        "local_features": {
                            "locations": np.array([]),
                            "descriptors": np.array([]),
                            "scales": np.array([]),
                            "attention": np.array([]),
                        }
                    }
                )
            return extracted_features

        # Input tensors.
        image_tensor = tf.convert_to_tensor(resized_image)

        # Extracted features.
        extracted_features = {}
        output = None

        if hasattr(config, "is_tf2_exported") and config.is_tf2_exported:
            predict = model.signatures["serving_default"]
            if config.use_local_features and config.use_global_features:
                output_dict = predict(
                    input_image=image_tensor,
                    input_scales=image_scales_tensor,
                    input_max_feature_num=max_feature_num_tensor,
                    input_abs_thres=score_threshold_tensor,
                    input_global_scales_ind=global_scales_ind_tensor,
                )
                output = [
                    output_dict["boxes"],
                    output_dict["features"],
                    output_dict["scales"],
                    output_dict["scores"],
                    output_dict["global_descriptors"],
                ]
            elif config.use_local_features:
                output_dict = predict(
                    input_image=image_tensor,
                    input_scales=image_scales_tensor,
                    input_max_feature_num=max_feature_num_tensor,
                    input_abs_thres=score_threshold_tensor,
                )
                output = [
                    output_dict["boxes"],
                    output_dict["features"],
                    output_dict["scales"],
                    output_dict["scores"],
                ]
            else:
                output_dict = predict(
                    input_image=image_tensor,
                    input_scales=image_scales_tensor,
                    input_global_scales_ind=global_scales_ind_tensor,
                )
                output = [output_dict["global_descriptors"]]
        else:
            if config.use_local_features and config.use_global_features:
                output = model(
                    image_tensor,
                    image_scales_tensor,
                    score_threshold_tensor,
                    max_feature_num_tensor,
                    global_scales_ind_tensor,
                )
            elif config.use_local_features:
                output = model(
                    image_tensor,
                    image_scales_tensor,
                    score_threshold_tensor,
                    max_feature_num_tensor,
                )
            else:
                output = model(image_tensor, image_scales_tensor, global_scales_ind_tensor)

        # Post-process extracted features: normalize, PCA (optional), pooling.
        if config.use_global_features:
            raw_global_descriptors = output[-1]
            global_descriptors_per_scale = feature_extractor.PostProcessDescriptors(
                raw_global_descriptors, config.delf_global_config.use_pca, global_pca_parameters
            )
            unnormalized_global_descriptor = tf.reduce_sum(
                global_descriptors_per_scale, axis=0, name="sum_pooling"
            )
            global_descriptor = tf.nn.l2_normalize(
                unnormalized_global_descriptor, axis=0, name="final_l2_normalization"
            )
            extracted_features.update(
                {
                    "global_descriptor": global_descriptor.numpy(),
                }
            )

        if config.use_local_features:
            boxes = output[0]
            raw_local_descriptors = output[1]
            feature_scales = output[2]
            attention_with_extra_dim = output[3]

            attention = tf.reshape(
                attention_with_extra_dim, [tf.shape(attention_with_extra_dim)[0]]
            )
            locations, local_descriptors = feature_extractor.DelfFeaturePostProcessing(
                boxes, raw_local_descriptors, config.delf_local_config.use_pca, local_pca_parameters
            )
            if not config.delf_local_config.use_resized_coordinates:
                locations /= scale_factors

            extracted_features.update(
                {
                    "local_features": {
                        "locations": locations.numpy(),
                        "descriptors": local_descriptors.numpy(),
                        "scales": feature_scales.numpy(),
                        "attention": attention.numpy(),
                    }
                }
            )

        return extracted_features

    return ExtractorFn


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, sizes):
    assert len(pts) == len(scores)
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        if is_opencv_version_greater_equal(4, 5, 3):
            kps = [
                cv2.KeyPoint(p[0], p[1], size=sizes[i], response=scores[i], octave=0)
                for i, p in enumerate(pts)
            ]
        else:
            kps = [
                cv2.KeyPoint(p[0], p[1], _size=sizes[i], _response=scores[i], _octave=0)
                for i, p in enumerate(pts)
            ]
    return kps


# Interface for pySLAM
class DelfFeature2D(BaseFeature2D):
    def __init__(self, num_features=1000, score_threshold=100, do_tf_logging=False):
        print("Using DelfFeature2D")
        self.lock = RLock()

        set_tf_logging(do_tf_logging)

        # Parse DelfConfig proto.
        self.delf_config = delf_config_pb2.DelfConfig()
        with tf.io.gfile.GFile(delf_config_file, "r") as f:
            text_format.Merge(f.read(), self.delf_config)
        self.delf_config.model_path = delf_model_path
        self.delf_config.delf_local_config.pca_parameters.mean_path = delf_mean_path
        self.delf_config.delf_local_config.pca_parameters.projection_matrix_path = (
            delf_projection_matrix_path
        )
        self.delf_config.delf_local_config.max_feature_num = num_features
        self.delf_config.delf_local_config.score_threshold = score_threshold
        print("DELF CONFIG\n:", self.delf_config)

        self.keypoint_size = 30  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint

        self.image_scales = list(self.delf_config.image_scales)
        # print('image scales: ',self.image_scales)
        try:
            self.scale_factor = self.image_scales[1] / self.image_scales[0]
        except:
            self.scale_factor = np.sqrt(2)  # according to default config and the paper
        # print('scale_factor: ',self.scale_factor)
        # self.image_levels = np.round(-np.log(self.image_scales)/np.log(self.scale_factor)).astype(np.int32)
        # print('image levels: ',self.image_levels)

        self.session = None

        self.pts = []
        self.kps = []
        self.des = []
        self.scales = []
        self.scores = []
        self.frame = None

        print("==> Loading pre-trained network.")
        self.load_model()
        print("==> Successfully loaded pre-trained network.")

    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.delf_config.delf_local_config.max_feature_num = num_features

    @property
    def num_features(self):
        return self.delf_config.delf_local_config.max_feature_num

    @property
    def score_threshold(self):
        return self.delf_config.delf_local_config.score_threshold

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def load_model(self):
        # Create graph before session :)
        # self.graph = tf.Graph().as_default()
        # self.session = tf.Session()
        # init_op = tf.global_variables_initializer()
        # self.session.run(init_op)
        self.extractor_fn = MakeExtractor(self.delf_config)

    def close(self):
        if self.session is not None:
            print("DELF: closing tf session")
            self.session.close()
            tf.reset_default_graph()

    def compute_kps_des(self, frame):
        with self.lock:
            # image_tf = tf.convert_to_tensor(frame, np.float32)
            # im = self.session.run(image_tf)

            # Extract and save features.
            extracted_features = self.extractor_fn(frame)
            locations_out = extracted_features["local_features"]["locations"]
            descriptors_out = extracted_features["local_features"]["descriptors"]
            feature_scales_out = extracted_features["local_features"]["scales"]
            attention_out = extracted_features["local_features"]["attention"]

            # (locations_out, descriptors_out, feature_scales_out, attention_out) = self.extractor_fn(frame)

            self.pts = locations_out[:, ::-1]
            self.des = descriptors_out
            self.scales = feature_scales_out
            self.scores = attention_out

            # N.B.: according to the paper "Large-Scale Image Retrieval with Attentive Deep Local Features":
            # We construct image pyramids by using scales that are a 2 factor apart. For the set of scales
            # with range from 0.25 to 2.0, 7 different scales are used.
            # The size of receptive field is inversely proportional to the scale; for example, for the 2.0 scale, the
            # receptive field of the network covers 146 × 146 pixels.
            # The receptive field size for the image at the original scale is 291 × 291.
            # sizes = self.keypoint_size * 1./self.scales
            sizes = self.keypoint_size * self.scales

            if False:
                # print('kps.shape', self.pts.shape)
                # print('des.shape', self.des.shape)
                # print('scales.shape', self.scales.shape)
                # print('scores.shape', self.scores.shape)
                print("scales:", self.scales)
                print("sizes:", sizes)

            self.kps = convert_pts_to_keypoints(self.pts, self.scores, sizes)

            return self.kps, self.des

    def detectAndCompute(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            self.frame = frame
            self.kps, self.des = self.compute_kps_des(frame)
            if kVerbose:
                print(
                    "detector: DELF, descriptor: DELF, #features: ",
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
                    "WARNING: DELF is recomputing both kps and des on last input frame", frame.shape
                )
                self.detectAndCompute(frame)
            return self.kps, self.des
