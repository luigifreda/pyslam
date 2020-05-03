# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Forked from:
# https://github.com/tensorflow/models/blob/master/research/delf/delf/python/examples/extract_features.py
"""Extracts DELF features from a given image and save results to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""

import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('delf') 

import cv2 

import warnings # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import os
import sys
import time
import json
import numpy as np
import h5py

if False:
    import tensorflow as tf
else: 
    # from https://stackoverflow.com/questions/56820327/the-name-tf-session-is-deprecated-please-use-tf-compat-v1-session-instead
    import tensorflow.compat.v1 as tf
    
# from https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96 to cope with the following error:
# "[...tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction=0.333  # from https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
#session = tf.Session(config=tf_config, ...)

from google.protobuf import text_format
from tensorflow.python.platform import app


# from delf import delf_config_pb2
# from delf import feature_extractor
# from delf import feature_io
from delf.protos import aggregation_config_pb2
from delf.protos import box_pb2
from delf.protos import datum_pb2
from delf.protos import delf_config_pb2
from delf.protos import feature_pb2
from delf.python import box_io
from delf.python import datum_io
from delf.python import delf_v1
from delf.python import feature_aggregation_extractor
from delf.python import feature_aggregation_similarity
from delf.python import feature_extractor
from delf.python import feature_io
from delf.python.examples import detector
from delf.python.examples import extractor
from delf.python import detect_to_retrieve
from delf.python import google_landmarks_dataset


delf_base_path='../../thirdparty/tensorflow_models/research/delf/delf/python/'
delf_config_file= delf_base_path + 'examples/delf_config_example.pbtxt'
delf_model_path=delf_base_path + 'examples/parameters/delf_gld_20190411/model/'
delf_mean_path=delf_base_path + 'examples/parameters/delf_gld_20190411/pca/mean.datum'
delf_projection_matrix_path=delf_base_path + 'examples/parameters/delf_gld_20190411/pca/pca_proj_mat.datum'


cmd_args = None

# Extension of feature files.
_DELF_EXT = '.h5'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
    """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
    with tf.gfile.GFile(list_path, 'r') as f:
        image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths


def MakeExtractor(sess, config, import_scope=None):
    """Creates a function to extract features from an image.

  Args:
    sess: TensorFlow session to use.
    config: DelfConfig proto containing the model configuration.
    import_scope: Optional scope to use for model.

  Returns:
    Function that receives an image and returns features.
  """
    tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING],
        config.model_path,
        import_scope=import_scope)
    import_scope_prefix = import_scope + '/' if import_scope is not None else ''
    input_image = sess.graph.get_tensor_by_name('%sinput_image:0' %
                                                import_scope_prefix)
    input_score_threshold = sess.graph.get_tensor_by_name(
        '%sinput_abs_thres:0' % import_scope_prefix)
    input_image_scales = sess.graph.get_tensor_by_name('%sinput_scales:0' %
                                                       import_scope_prefix)
    input_max_feature_num = sess.graph.get_tensor_by_name(
        '%sinput_max_feature_num:0' % import_scope_prefix)
    boxes = sess.graph.get_tensor_by_name('%sboxes:0' % import_scope_prefix)
    raw_descriptors = sess.graph.get_tensor_by_name('%sfeatures:0' %
                                                    import_scope_prefix)
    feature_scales = sess.graph.get_tensor_by_name('%sscales:0' %
                                                   import_scope_prefix)
    attention_with_extra_dim = sess.graph.get_tensor_by_name(
        '%sscores:0' % import_scope_prefix)
    attention = tf.reshape(attention_with_extra_dim,
                           [tf.shape(attention_with_extra_dim)[0]])

    locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
        boxes, raw_descriptors, config)

    def ExtractorFn(image):
        """Receives an image and returns DELF features.

    Args:
      image: Uint8 array with shape (height, width 3) containing the RGB image.

    Returns:
      Tuple (locations, descriptors, feature_scales, attention)
    """
        return sess.run([locations, descriptors, feature_scales, attention],
                        feed_dict={
                            input_image: image,
                            input_score_threshold: config.delf_local_config.score_threshold,
                            input_image_scales: list(config.image_scales),
                            input_max_feature_num: config.delf_local_config.max_feature_num
                        })

    return ExtractorFn


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Read list of images.
    #tf.logging.info('Reading list of images...')
    #image_paths = _ReadImageList(cmd_args.list_images_path)
    num_images = 1 #len(image_paths)
    #tf.logging.info('done! Found %d images', num_images)
    img = cv2.imread('../data/kitti06-12-color.png',cv2.IMREAD_COLOR)
    #value = tf.gfile.FastGFile('../data/kitti06-12-color.jpg', 'rb').read()

    # Parse DelfConfig proto.
    delf_config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile(cmd_args.config_path, 'r') as f:
        text_format.Merge(f.read(), delf_config)
    delf_config.model_path = delf_model_path
    delf_config.delf_local_config.pca_parameters.mean_path = delf_mean_path
    delf_config.delf_local_config.pca_parameters.projection_matrix_path = delf_projection_matrix_path
    print('config:', delf_config)

    # Create output directory if necessary.
    if not os.path.exists(cmd_args.output_dir):
        os.makedirs(cmd_args.output_dir)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Reading list of images.
        #filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
        #reader = tf.WholeFileReader()
        #_, value = reader.read(filename_queue)
        #image_tf = tf.image.decode_jpeg(value, channels=3) # Returns a `Tensor` of type `uint8`.
        
        # from https://stackoverflow.com/questions/48727264/how-to-convert-numpy-array-image-to-tensorflow-image
        #image_tf = np.array(img)[:, :, 0:3]
        #image_tf = tf.convert_to_tensor(img, np.float32)
        #print('\nimagetf info',np.info(image_tf))
        # run the network to get the predictions
        #predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

        with tf.Session(config=tf_config) as sess:  
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            extractor_fn = MakeExtractor(sess, delf_config)

            # Start input enqueue threads.
            #coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start = time.clock()

            with h5py.File(os.path.join(cmd_args.output_dir, 'keypoints.h5'), 'w') as h5_kp, \
                 h5py.File(os.path.join(cmd_args.output_dir, 'descriptors.h5'), 'w') as h5_desc, \
                 h5py.File(os.path.join(cmd_args.output_dir, 'scores.h5'), 'w') as h5_score, \
                 h5py.File(os.path.join(cmd_args.output_dir, 'scales.h5'), 'w') as h5_scale:
                for i in range(num_images):
                    key = 'img' #os.path.splitext(os.path.basename(image_paths[i]))[0]
                    print('Processing "{}"'.format(key))

                    # # Write to log-info once in a while.
                    # if i == 0:
                    #     tf.logging.info(
                    #         'Starting to extract DELF features from images...')
                    # elif i % _STATUS_CHECK_ITERATIONS == 0:
                    #     elapsed = (time.clock() - start)
                    #     tf.logging.info(
                    #         'Processing image %d out of %d, last %d '
                    #         'images took %f seconds', i, num_images,
                    #         _STATUS_CHECK_ITERATIONS, elapsed)
                    #     start = time.clock()

                    # # Get next image.
                    image_tf = tf.convert_to_tensor(img, np.float32)
                    im = sess.run(image_tf)
                    
                    #im = sess.run({'DecodeJpeg:0': image_tf}) # from https://stackoverflow.com/questions/48727264/how-to-convert-numpy-array-image-to-tensorflow-image

                    # If descriptor already exists, skip its computation.
                    # out_desc_filename = os.path.splitext(os.path.basename(
                    #     image_paths[i]))[0] + _DELF_EXT
                    # out_desc_fullpath = os.path.join(cmd_args.output_dir, out_desc_filename)
                    # if tf.gfile.Exists(out_desc_fullpath):
                    #   tf.logging.info('Skipping %s', image_paths[i])
                    #   continue

                    # Extract and save features.
                    (locations_out, descriptors_out, feature_scales_out, attention_out) = extractor_fn(im) 
                    
                    # np.savez('{}.npz'.format(config.delf_local_config.max_feature_num), keypoints=locations_out)

                    # feature_io.WriteToFile(out_desc_fullpath, locations_out,
                    #                        feature_scales_out, descriptors_out,
                    #                        attention_out)
                    h5_kp[key] = locations_out[:, ::-1]
                    h5_desc[key] = descriptors_out
                    h5_scale[key] = feature_scales_out
                    h5_score[key] = attention_out
                    print('#extracted keypoints:',len(h5_kp[key]))
                    print('des[0]:',descriptors_out[0])

            print('done!')
            # Finalize enqueue threads.
            #coord.request_stop()
            #coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument(
        '--config_path',
        type=str,
        default=delf_config_file, #'misc/delf/delf_config_example.pbtxt',
        help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
    parser.add_argument(
        '--list_images_path',
        type=str,
        help="""
          Path to list of images whose DELF features will be extracted.
          """)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./delf',
        help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)

    cmd_args, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)