"""
* This file is part of PYSLAM 
* Adapted from https://github.com/vcg-uvic/lf-net-release/blob/master/run_lfnet.py, see the license therein. 
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
config.cfg.set_lib('lfnet',prepend=True) 
  
import os
import sys
import time


from threading import RLock

import cv2
import numpy as np

import warnings # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427
warnings.filterwarnings('ignore', category=FutureWarning)

if False:
    import tensorflow as tf
else: 
    # from https://stackoverflow.com/questions/56820327/the-name-tf-session-is-deprecated-please-use-tf-compat-v1-session-instead
    import tensorflow.compat.v1 as tf



import importlib
from tqdm import tqdm
import pickle

from lfnet.mydatasets import *

from lfnet.det_tools import *
from lfnet.eval_tools import draw_keypoints
from lfnet.common.tf_train_utils import get_optimizer
from imageio import imread, imsave
from lfnet.inference import *
from lfnet.utils import embed_breakpoint, print_opt

from lfnet.common.argparse_utils import *
    
from utils_tf import set_tf_logging
from utils_img import img_from_floats
from utils import Printer, print_options


kLfNetBasePath = config.cfg.root_folder + '/thirdparty/lfnet'
kLfNetModelPath = kLfNetBasePath + '/pretrained/lfnet-norotaug'

kModelFolderPath = kLfNetBasePath + '/models'
if kModelFolderPath not in sys.path:
    sys.path.append(kModelFolderPath)


kVerbose = True   


def build_networks(lfnet_config, photo, is_training):
    # Detector 
    DET = importlib.import_module(lfnet_config.detector)
    detector = DET.Model(lfnet_config, is_training)

    if lfnet_config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)

    heatmaps, det_endpoints = build_detector_helper(lfnet_config, detector, photo)

    # extract patches
    kpts = det_endpoints['kpts']
    batch_inds = det_endpoints['batch_inds']

    kp_patches = build_patch_extraction(lfnet_config, det_endpoints, photo)

    # Descriptor
    DESC = importlib.import_module(lfnet_config.descriptor)
    descriptor = DESC.Model(lfnet_config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(kp_patches, reuse=False) # [B*K,D]

    # scale and orientation (extra)
    scale_maps = det_endpoints['scale_maps']
    ori_maps = det_endpoints['ori_maps'] # cos/sin
    degree_maps, _ = get_degree_maps(ori_maps) # degree (rgb psuedo color code)
    kpts_scale = det_endpoints['kpts_scale'] # scale factor 
    kpts_ori = det_endpoints['kpts_ori']
    kpts_ori = tf.atan2(kpts_ori[:,1], kpts_ori[:,0]) # radian

    ops = {
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts,
        'feats': desc_feats,
        # EXTRA
        'scale_maps': scale_maps,
        'kpts_scale': kpts_scale,
        'degree_maps': degree_maps,
        'kpts_ori': kpts_ori,
        'heatmaps': heatmaps, 
    }
    return ops


def build_detector_helper(lfnet_config, detector, photo):
    # if lfnet_config.detector == 'resnet_detector':
    #     heatmaps, det_endpoints = build_deep_detector(lfnet_config, detector, photo, reuse=False)
    # elif lfnet_config.detector == 'mso_resnet_detector':
    if lfnet_config.use_nms3d:
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(lfnet_config, detector, photo, reuse=False)
    else:
        heatmaps, det_endpoints = build_multi_scale_deep_detector(lfnet_config, detector, photo, reuse=False)
    # else:
    #     raise ValueError()
    return heatmaps, det_endpoints


def build_lfnet_config(): 
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8, help='the number of threads (for dataset)')

    io_arg = add_argument_group('In/Out', parser)
    #io_arg.add_argument('--in_dir', type=str, default='./samples', help='input image directory')
    # io_arg.add_argument('--in_dir', type=str, default='./release/outdoor_examples/images/sacre_coeur/dense/images', help='input image directory')
    #io_arg.add_argument('--out_dir', type=str, default='./lfnet', help='where to save keypoints')
    io_arg.add_argument('--full_output', type=str2bool, default=True, help='dump keypoint image')

    model_arg = add_argument_group('Model', parser)
    model_arg.add_argument('--model', type=str, default=kLfNetModelPath, help='model file or directory')
    model_arg.add_argument('--top_k', type=int, default=500, help='number of keypoints')
    model_arg.add_argument('--max_longer_edge', type=int, default=-1, help='resize image (do nothing if max_longer_edge <= 0)')

    tmp_config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    # restore other hyperparams to build model
    if os.path.isdir(tmp_config.model):
        config_path = os.path.join(tmp_config.model, 'config.pkl')
    else:
        config_path = os.path.join(os.path.dirname(tmp_config.model), 'config.pkl')
    try:
        with open(config_path, 'rb') as f:
            lfnet_config = pickle.load(f)
            #print_opt(lfnet_config)
    except:
        raise ValueError('Fail to open {}'.format(config_path))

    for attr, dst_val in sorted(vars(tmp_config).items()):
        if hasattr(lfnet_config, attr):
            src_val = getattr(lfnet_config, attr)
            if src_val != dst_val:
                setattr(lfnet_config, attr, dst_val)
        else:
            setattr(lfnet_config, attr, dst_val)
    return lfnet_config 


def convert_to_keypoints(kpts, scales, orientations, heatmaps):
    kps = []
    for kp,size,angle in zip(kpts,scales,orientations):
        x, y = np.round(kp).astype(np.int)
        response = heatmaps[y,x]
        kps.append(cv2.KeyPoint(x, y, _size=size, _angle=angle, _response=response))
    return kps
    

# interface for pySLAM
class LfNetFeature2D: 
    def __init__(self,
                 num_features=2000, 
                 do_tf_logging=False):  
        print('Using LfNetFeature2D')   
        self.lock = RLock()
        
        self.model_base_path = kLfNetBasePath
        self.model_path = kLfNetModelPath
        
        self.lfnet_config = build_lfnet_config()
        print_options(self.lfnet_config,'LFNET CONFIG')
        
        self.num_features=num_features
        self.lfnet_config.top_k = self.num_features

        set_tf_logging(do_tf_logging)
        
        print('==> Loading pre-trained network.')
        # Build Networks
        tf.reset_default_graph()

        self.photo_ph = tf.placeholder(tf.float32, [1, None, None, 1]) # input grayscale image, normalized by 0~1
        is_training = tf.constant(False) # Always False in testing

        self.ops = build_networks(self.lfnet_config, self.photo_ph, is_training)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True 
        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

        # load model
        saver = tf.train.Saver()
        print('Load trained models...')

        if os.path.isdir(self.lfnet_config.model):
            checkpoint = tf.train.latest_checkpoint(self.lfnet_config.model)
            model_dir = self.lfnet_config.model
        else:
            checkpoint = self.lfnet_config.model
            model_dir = os.path.dirname(self.lfnet_config.model)

        if checkpoint is not None:
            print('Checkpoint', os.path.basename(checkpoint))
            print("[{}] Resuming...".format(time.asctime()))
            saver.restore(self.session, checkpoint)
        else:
            raise ValueError('Cannot load model from {}'.format(model_dir))    
        print('==> Successfully loaded pre-trained network.')
                
        self.pts = []
        self.kps = []        
        self.des = []
        self.frame = None    
        self.keypoint_size = 20.  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint                       
    

    def __del__(self): 
        self.close()
        
        
    def close(self):
        if self.session is not None:
            print('DELF: closing tf session')
            self.session.close()
            tf.reset_default_graph()           
    
    
    def compute_kps_des(self,photo):     
        with self.lock:        
            height, width = photo.shape[:2]
            longer_edge = max(height, width)
            if self.lfnet_config.max_longer_edge > 0 and longer_edge > self.lfnet_config.max_longer_edge:
                if height > width:
                    new_height = self.lfnet_config.max_longer_edge
                    new_width = int(width * self.lfnet_config.max_longer_edge / height)
                else:
                    new_height = int(height * self.lfnet_config.max_longer_edge / width)
                    new_width = self.lfnet_config.max_longer_edge
                photo = cv2.resize(photo, (new_width, new_height))
                height, width = photo.shape[:2]
            rgb = photo.copy()
            if photo.ndim == 3 and photo.shape[-1] == 3:
                photo = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
            photo = photo[None,...,None].astype(np.float32) / 255.0 # normalize 0-1
            assert photo.ndim == 4 # [1,H,W,1]

            feed_dict = {self.photo_ph: photo,}
            #if self.lfnet_config.full_output:
            fetch_dict = {
                'kpts': self.ops['kpts'],
                'feats': self.ops['feats'],
                'kpts_scale': self.ops['kpts_scale'],
                'kpts_ori': self.ops['kpts_ori'],
                'scale_maps': self.ops['scale_maps'],
                'degree_maps': self.ops['degree_maps'],
                'heatmaps': self.ops['heatmaps'],                    
            }
            outs = self.session.run(fetch_dict, feed_dict=feed_dict)
                
            self.pts = outs['kpts']
            scales = outs['kpts_scale']     
            scale_maps = outs['scale_maps'].reshape(height, width)                           
            orientations = outs['kpts_ori'] 
            heatmaps = outs['heatmaps'].reshape(height, width)    
                                    
            # kp_img = draw_keypoints(rgb, outs['kpts']) # draw keypoints
            # scale_range = self.lfnet_config.net_max_scale-self.lfnet_config.net_min_scale
            # if scale_range == 0:
            #     scale_range = 1.0
            # scale_img = (outs['scale_maps'][0]*255/scale_range).astype(np.uint8)
            # ori_img = (outs['degree_maps'][0]*255).astype(np.uint8)                  
                            
            if False: 
                # print and draw debug stuff 
                self.debug(self.pts,scales,orientations,scale_maps,heatmaps)  
  
            self.kps = convert_to_keypoints(self.pts, scales*self.keypoint_size, np.degrees(orientations), heatmaps)     
            self.des = outs['feats']                           
            return self.kps, self.des 
        
        
    def debug(self,pts,scales,orientations,scale_maps,heatmaps):
        print('orientations:',orientations)                     
        print('scales:',scales)            
        print('heatmaps info:')
        np.info(heatmaps)
        print('scalemaps info:')
        np.info(scale_maps)        
        heatmaps_img = img_from_floats(heatmaps)
        cv2.imshow('heatmap',heatmaps_img)             
        scalemaps_img = img_from_floats(scale_maps)
        cv2.imshow('scale maps',scalemaps_img)     
        cv2.waitKey(1)        
    
    
    def detectAndCompute(self, frame, mask=None):  #mask is a fake input  
        with self.lock:
            self.frame = frame         
            self.kps, self.des = self.compute_kps_des(frame)            
            if kVerbose:
                print('detector: LFNET , descriptor: LFNET , #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])                  
            return self.kps, self.des
    
           
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input  
        with self.lock:        
            if self.frame is not frame:
                self.detectAndCompute(frame)        
            return self.kps
    
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute(self, frame, kps=None, mask=None): # kps is a fake input, mask is a fake input
        with self.lock:        
            if self.frame is not frame:
                Printer.orange('WARNING: LFNET  is recomputing both kps and des on last input frame', frame.shape)            
                self.detectAndCompute(frame)
            return self.kps, self.des                 
           
                   