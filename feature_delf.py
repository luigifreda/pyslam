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


import config
config.cfg.set_lib('delf') 

import cv2 

from threading import RLock
from utils import Printer 

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


from utils_tf import set_tf_logging
#from utils import print_options


delf_base_path = config.cfg.root_folder + '/thirdparty/tensorflow_models/research/delf/delf/python/'
delf_config_file = delf_base_path + 'examples/delf_config_example.pbtxt'
delf_model_path = delf_base_path + 'examples/parameters/delf_gld_20190411/model/'
delf_mean_path = delf_base_path + 'examples/parameters/delf_gld_20190411/pca/mean.datum'
delf_projection_matrix_path = delf_base_path + 'examples/parameters/delf_gld_20190411/pca/pca_proj_mat.datum'



kVerbose = True   

        
        
def MakeExtractor(sess, config, import_scope=None):
    """Creates a function to extract features from an image.

  Args:
    sess: TensorFlow session to use.
    config: DelfConfig proto containing the model configuration.
    import_scope: Optional scope to use for model.

  Returns:
    Function that receives an image and returns features.
  """
    tf.saved_model.loader.load( sess, [tf.saved_model.tag_constants.SERVING], config.model_path, import_scope=import_scope)
    
    import_scope_prefix = import_scope + '/' if import_scope is not None else ''
    input_image = sess.graph.get_tensor_by_name('%sinput_image:0' % import_scope_prefix)
    input_score_threshold = sess.graph.get_tensor_by_name('%sinput_abs_thres:0' % import_scope_prefix)
    input_image_scales = sess.graph.get_tensor_by_name('%sinput_scales:0' % import_scope_prefix)
    input_max_feature_num = sess.graph.get_tensor_by_name('%sinput_max_feature_num:0' % import_scope_prefix)
    boxes = sess.graph.get_tensor_by_name('%sboxes:0' % import_scope_prefix)
    raw_descriptors = sess.graph.get_tensor_by_name('%sfeatures:0' % import_scope_prefix)
    feature_scales = sess.graph.get_tensor_by_name('%sscales:0' % import_scope_prefix)
    attention_with_extra_dim = sess.graph.get_tensor_by_name('%sscores:0' % import_scope_prefix)
    attention = tf.reshape(attention_with_extra_dim,[tf.shape(attention_with_extra_dim)[0]])

    locations, descriptors = feature_extractor.DelfFeaturePostProcessing(boxes, raw_descriptors, config)

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
                            input_score_threshold:
                            config.delf_local_config.score_threshold,
                            input_image_scales: list(config.image_scales),
                            input_max_feature_num:
                            config.delf_local_config.max_feature_num
                        })

    return ExtractorFn        


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, sizes): 
    assert(len(pts)==len(scores))
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints  
        kps = [ cv2.KeyPoint(p[0], p[1], _size=sizes[i], _response=scores[i]) for i,p in enumerate(pts) ]                      
    return kps         


# interface for pySLAM 
class DelfFeature2D: 
    def __init__(self,
                 num_features=1000, 
                 score_threshold=100, 
                 do_tf_logging=False):  
        print('Using DelfFeature2D')   
        self.lock = RLock()

        set_tf_logging(do_tf_logging)
                
        # Parse DelfConfig proto.
        self.delf_config = delf_config_pb2.DelfConfig()
        with tf.gfile.FastGFile(delf_config_file, 'r') as f:
            text_format.Merge(f.read(), self.delf_config)
        self.delf_config.model_path = delf_model_path
        self.delf_config.delf_local_config.pca_parameters.mean_path = delf_mean_path
        self.delf_config.delf_local_config.pca_parameters.projection_matrix_path = delf_projection_matrix_path
        self.delf_config.delf_local_config.max_feature_num = num_features
        self.delf_config.delf_local_config.score_threshold = score_threshold
        print('DELF CONFIG\n:', self.delf_config)     
        
        self.keypoint_size = 30  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint       
        
        self.image_scales = list(self.delf_config.image_scales)  
        #print('image scales: ',self.image_scales)
        try: 
            self.scale_factor = self.image_scales[1]/self.image_scales[0]
        except:
            self.scale_factor = np.sqrt(2)  # according to default config and the paper 
        #print('scale_factor: ',self.scale_factor)
        #self.image_levels = np.round(-np.log(self.image_scales)/np.log(self.scale_factor)).astype(np.int32)    
        #print('image levels: ',self.image_levels)                
                        
        self.session = None 
                
        self.pts = []
        self.kps = []        
        self.des = []
        self.scales = []
        self.scores = []        
        self.frame = None 
        
        print('==> Loading pre-trained network.')
        self.load_model()
        print('==> Successfully loaded pre-trained network.')
            
            
    @property
    def num_features(self):         
        return self.delf_config.delf_local_config.max_feature_num    
    
    @property
    def score_threshold(self):         
        return self.delf_config.delf_local_config.score_threshold       
                    

    def __del__(self): 
        self.close()
      
      
    def load_model(self):
        # Create graph before session :)
        self.graph = tf.Graph().as_default()
        self.session = tf.Session() 
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)
        self.extractor_fn = MakeExtractor(self.session, self.delf_config)            
        
        
    def close(self):
        if self.session is not None:
            print('DELF: closing tf session')
            self.session.close()
            tf.reset_default_graph()        
        
        
    def compute_kps_des(self, frame):
        with self.lock:         
            image_tf = tf.convert_to_tensor(frame, np.float32)
            im = self.session.run(image_tf)
            
            # Extract and save features.
            (locations_out, descriptors_out, feature_scales_out, attention_out) = self.extractor_fn(im) 
            
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
            #sizes = self.keypoint_size * 1./self.scales
            sizes = self.keypoint_size * self.scales
            
            if False:            
                # print('kps.shape', self.pts.shape)
                # print('des.shape', self.des.shape)          
                # print('scales.shape', self.scales.shape)          
                # print('scores.shape', self.scores.shape)  
                print('scales:',self.scales)     
                print('sizes:',sizes)
            
            self.kps = convert_pts_to_keypoints(self.pts, self.scores, sizes)
            
            return self.kps, self.des   
        
           
    def detectAndCompute(self, frame, mask=None):  #mask is a fake input  
        with self.lock: 
            self.frame = frame         
            self.kps, self.des = self.compute_kps_des(frame)            
            if kVerbose:
                print('detector: DELF, descriptor: DELF, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])                  
            return self.kps, self.des
    
           
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input  
        with self.lock:         
            #if self.frame is not frame:
            self.detectAndCompute(frame)        
            return self.kps
    
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute(self, frame, kps=None, mask=None): # kps is a fake input, mask is a fake input
        with self.lock:         
            if self.frame is not frame:
                Printer.orange('WARNING: DELF is recomputing both kps and des on last input frame', frame.shape)            
                self.detectAndCompute(frame)
            return self.kps, self.des                
                   