# Copyright 2019 EPFL, Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://raw.githubusercontent.com/cvlab-epfl/log-polar-descriptors/master/example.py

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import h5py
import argparse
from time import time

import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('logpolar') 

from configs.defaults import _C as cfg
from modules.hardnet.models import HardNet


logpolar_base_path='../../thirdparty/logpolar/'


# Configuration
# Set use_log_polar=False to load the "Cartesian" models used in the paper
def extract_descriptors(input_filename, output_filename, use_log_polar,
                        num_keypoints, verbose):
    # Setup
    ROOT = logpolar_base_path # os.getcwd()  

    if use_log_polar:
        config_path = os.path.join(ROOT, 'configs',
                                   'init_one_example_ptn_96.yml')
        if verbose:
            print('-- Using log-polar models')
    else:
        config_path = os.path.join(ROOT, 'configs',
                                   'init_one_example_stn_16.yml')
        if verbose:
            print('-- Using cartesian models')

    cfg.merge_from_file(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        if torch.cuda.is_available():
            print('-- Using GPU')
        else:
            print('-- Using CPU')

    # Extract SIFT keypoints
    img = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)

    # A safe image size is ~1000px on the largest dimension
    # To extract features on larger images you might want to increase the padding
    max_size = 1024
    if any([s > max_size for s in img.shape]):
        h, w = img.shape
        if h > w:
            img = cv2.resize(img, (int(w * max_size / h), max_size),
                             cv2.INTER_CUBIC)
        elif w > h:
            img = cv2.resize(img, (max_size, int(h * max_size / w)),
                             cv2.INTER_CUBIC)
    h, w = img.shape

    # get keypoints, scale and locatinos from SIFT or another detector
    sift = cv2.xfeatures2d.SIFT_create(num_keypoints)
    keypoints = sift.detect(img, None)

    pts = np.array([kp.pt for kp in keypoints])
    scales = np.array([kp.size for kp in keypoints])
    oris = np.array([kp.angle for kp in keypoints])

    # Mirror-pad the image to avoid boundary effects
    if any([s > cfg.TEST.PAD_TO for s in img.shape[:2]]):
        raise RuntimeError(
            "Image exceeds acceptable size ({}x{}), please downsample".format(
                cfg.TEST.PAD_TO, cfg.TEST.PAD_TO))

    fillHeight = cfg.TEST.PAD_TO - img.shape[0]
    fillWidth = cfg.TEST.PAD_TO - img.shape[1]

    padLeft = int(np.round(fillWidth / 2))
    padRight = int(fillWidth - padLeft)
    padUp = int(np.round(fillHeight / 2))
    padDown = int(fillHeight - padUp)

    img = np.pad(img,
                 pad_width=((padUp, padDown), (padLeft, padRight)),
                 mode='reflect')
    if verbose:
        print('-- Padding image from {}x{} to {}x{}'.format(
            h, w, img.shape[0], img.shape[1]))

    # Normalize keypoint locations
    kp_norm = []
    for i, p in enumerate(pts):
        _p = 2 * np.array([(p[0] + padLeft) / (cfg.TEST.PAD_TO),
                           (p[1] + padUp) / (cfg.TEST.PAD_TO)]) - 1
        kp_norm.append(_p)

    theta = [
        torch.from_numpy(np.array(kp_norm)).float().squeeze(),
        torch.from_numpy(scales).float(),
        torch.from_numpy(np.array([np.deg2rad(o) for o in oris])).float()
    ]

    # Instantiate the model
    t = time()
    model = HardNet(transform=cfg.TEST.TRANSFORMER,
                    coords=cfg.TEST.COORDS,
                    patch_size=cfg.TEST.IMAGE_SIZE,
                    scale=cfg.TEST.SCALE,
                    is_desc256=cfg.TEST.IS_DESC_256,
                    orientCorrect=cfg.TEST.ORIENT_CORRECTION)

    # Load weights
    model.load_state_dict(torch.load(logpolar_base_path + cfg.TEST.MODEL_WEIGHTS)['state_dict'])
    model.eval()
    model.to(device)
    if verbose:
        print('-- Instantiated model in {:0.2f} sec.'.format(time() - t))

    # Extract descriptors
    imgs, img_keypoints = torch.from_numpy(img).unsqueeze(0).to(device), \
          [theta[0].to(device), theta[1].to(device), theta[2].to(device)]

    t = time()
    descriptors, patches = model({input_filename: imgs}, img_keypoints,
                                 [input_filename] * len(img_keypoints[0]))
    if verbose:
        print('-- Computed {} descriptors in {:0.2f} sec.'.format(
            descriptors.shape[0],
            time() - t))

    keypoints_array = np.concatenate([pts, scales[..., None], oris[..., None]],
                                     axis=1)

    t = time()
    with h5py.File(output_filename, 'w') as f:
        f['keypoints'] = keypoints_array
        f['descriptors'] = descriptors.cpu().detach().numpy()
        print('-- Saved {} descriptors in {:0.2f} sec.'.format(
            descriptors.shape[0],
            time() - t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default=logpolar_base_path+'testImg.jpeg',
                        help='Input image')
    parser.add_argument('--output',
                        type=str,
                        default=logpolar_base_path+'testImg.h5',
                        help='Output file')
    parser.add_argument('--use_log_polar',
                        type=bool,
                        default=True,
                        help='Use log-polar models. Set to False to use '
                        'cartesian models instead.')
    parser.add_argument('--num_keypoints',
                        type=int,
                        default=1024,
                        help='Number of keypoints')
    parser.add_argument('--verbose',
                        type=bool,
                        default=True,
                        help='Set to False to suppress feedback')

    config, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        parser.print_usage()
    else:
        extract_descriptors(config.input, config.output, config.use_log_polar,
                            config.num_keypoints, config.verbose)