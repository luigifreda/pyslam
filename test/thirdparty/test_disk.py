# adapted from https://github.com/cvlab-epfl/disk/blob/master/detect.py 

import torch, os, argparse, h5py, warnings, imageio
import numpy as np
from tqdm import tqdm

import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('disk')
config.cfg.set_lib('torch-dimcheck')
config.cfg.set_lib('torch-localize')
config.cfg.set_lib('unets')

import multiprocessing as mp 
import torch, h5py, imageio, os, argparse
import numpy as np
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_dimcheck import dimchecked

from disk import DISK, Features

class Image:
    def __init__(self, bitmap: ['C', 'H', 'W'], fname: str, orig_shape=None):
        self.bitmap     = bitmap
        self.fname      = fname
        if orig_shape is None:
            self.orig_shape = self.bitmap.shape[1:]
        else:
            self.orig_shape = orig_shape

    def resize_to(self, shape):
        return Image(
            self._pad(self._interpolate(self.bitmap, shape), shape),
            self.fname,
            orig_shape=self.bitmap.shape[1:],
        )

    @dimchecked
    def to_image_coord(self, xys: [2, 'N']) -> ([2, 'N'], ['N']):
        f, _size = self._compute_interpolation_size(self.bitmap.shape[1:])
        scaled = xys / f

        h, w = self.orig_shape
        x, y = scaled

        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)

        return scaled, mask

    def _compute_interpolation_size(self, shape):
        x_factor = self.orig_shape[0] / shape[0]
        y_factor = self.orig_shape[1] / shape[1]

        f = 1 / max(x_factor, y_factor)

        if x_factor > y_factor:
            new_size = (shape[0], int(f * self.orig_shape[1]))
        else:
            new_size = (int(f * self.orig_shape[0]), shape[1])

        return f, new_size

    @dimchecked
    def _interpolate(self, image: ['C', 'H', 'W'], shape) -> ['C', 'h', 'w']:
        _f, size = self._compute_interpolation_size(shape)
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    
    @dimchecked
    def _pad(self, image: ['C', 'H', 'W'], shape) -> ['C', 'h', 'w']:
        x_pad = shape[0] - image.shape[1]
        y_pad = shape[1] - image.shape[2]

        if x_pad < 0 or y_pad < 0:
            raise ValueError("Attempting to pad by negative value")

        return F.pad(image, (0, y_pad, 0, x_pad))


class SceneDataset:
    def __init__(self, image_path, crop_size=(None, None)):
        self.image_path = image_path
        self.crop_size  = crop_size
        self.names = [p for p in os.listdir(image_path) \
                      if p.endswith(args.image_extension)]
        print('image names:',self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, ix):
        name   = self.names[ix]
        path   = os.path.join(self.image_path, name) 
        img    = np.ascontiguousarray(imageio.imread(path))
        tensor = torch.from_numpy(img).to(torch.float32)

        if len(tensor.shape) == 2: # some images may be grayscale
            tensor = tensor.unsqueeze(-1).expand(-1, -1, 3)

        bitmap              = tensor.permute(2, 0, 1) / 255.
        extensionless_fname = os.path.splitext(name)[0]

        image = Image(bitmap, extensionless_fname)

        if self.crop_size != (None, None):
            image = image.resize_to(self.crop_size)

        return image

    @staticmethod
    def collate_fn(images):
        bitmaps = torch.stack([im.bitmap for im in images], dim=0)
        
        return bitmaps, images

def extract(dataset, save_path):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
    )

    if args.mode == 'nms':
        extract = partial(
            model.features,
            kind='nms',
            window_size=args.window,
            cutoff=0.,
            n=args.n
        )
    else:
        extract = partial(model.features, kind='rng')

    os.makedirs(os.path.join(save_path), exist_ok=True)
    keypoint_h5   = h5py.File(os.path.join(save_path, 'keypoints.h5'), 'w')
    descriptor_h5 = h5py.File(os.path.join(save_path, 'descriptors.h5'), 'w')
    if args.detection_scores:
        score_h5 = h5py.File(os.path.join(save_path, 'scores.h5'), 'w')

    print('loop:')
    pbar = tqdm(dataloader)
    for bitmaps, images in pbar:
        print('bitmaps: ', bitmaps)
        print('images: ', images)
        bitmaps = bitmaps.to(DEV, non_blocking=True)

        with torch.no_grad():
            try:
                batched_features = extract(bitmaps)
            except RuntimeError as e:
                if 'U-Net failed' in str(e):
                    msg = ('Please use input size which is multiple of 16 (or '
                           'adjust the --height and --width flags to let this '
                           'script rescale it automatically). This is because '
                           'we internally use a U-Net with 4 downsampling '
                           'steps, each by a factor of 2, therefore 2^4=16.')

                    raise RuntimeError(msg) from e
                else:
                    raise

        for features, image in zip(batched_features.flat, images):
            features = features.to(CPU)

            kps_crop_space = features.kp.T
            kps_img_space, mask = image.to_image_coord(kps_crop_space)

            keypoints   = kps_img_space.numpy().T[mask]
            descriptors = features.desc.numpy()[mask]
            scores      = features.kp_logp.numpy()[mask]

            order = np.argsort(scores)[::-1]

            keypoints   = keypoints[order]
            descriptors = descriptors[order]
            scores      = scores[order]

            assert descriptors.shape[1] == args.desc_dim
            assert keypoints.shape[1] == 2

            if args.f16:
                descriptors = descriptors.astype(np.float16)

            # keypoint_h5.create_dataset(image.fname, data=keypoints)
            # descriptor_h5.create_dataset(image.fname, data=descriptors)

            # if args.detection_scores:
            #     score_h5.create_dataset(image.fname, data=scores)

            pbar.set_postfix(n=keypoints.shape[0])
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        "Script for detection and description (but not matching) of keypoints. "
        "It processes all images with extension given by `--image-extension` found "
        "in `image-path` directory. Images are resized to `--height` x `--width` "
        "for internal processing (padding them if necessary) and the output "
        "coordinates are then transformed back to original image size."),
    
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--height', default=None, type=int,
        help='rescaled height (px). If unspecified, image is not resized in height dimension'
    )
    parser.add_argument(
        '--width', default=None, type=int,
        help='rescaled width (px). If unspecified, image is not resized in width dimension'
    )
    parser.add_argument(
        '--image-extension', default='ppm', type=str,
        help='This script ill process all files which match `image-path/*.{--image-extension}`'
    )
    parser.add_argument(
        '--f16', action='store_true',
        help='Store descriptors in fp16 (half precision) format'
    )
    parser.add_argument('--window', type=int, default=5, help='NMS window size')
    parser.add_argument(
        '--n', type=int, default=None,
        help='Maximum number of features to extract. If unspecified, the number is not limited'
    )
    parser.add_argument(
        '--desc-dim', type=int, default=128,
        help='descriptor dimension. Needs to match the checkpoint value.'
    )
    parser.add_argument(
        '--mode', choices=['nms', 'rng'], default='nms',
        help=('Whether to extract features using the non-maxima suppresion mode or '
              'through training-time grid sampling technique')
    )
    
    default_model_path = os.path.split(os.path.abspath(__file__))[0] + '/../../thirdparty/disk/depth-save.pth'
    parser.add_argument(
         '--model_path', type=str, default=default_model_path,
        help="Path to the model's .pth save file"
    )
    parser.add_argument('--detection-scores', action='store_true')
    
    parser.add_argument(
        '--h5_path',
        default='./out_disk',
        help=("Directory where keypoints.h5 and descriptors.h5 will be stored. This"
              " will be created if it doesn't already exist.")
    )
    parser.add_argument(
        '--image_path',
        default='../data/graf',
        help="Directory with images to be processed."
    )
    args = parser.parse_args()
    DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CPU   = torch.device('cpu')
    dataset = SceneDataset(args.image_path, crop_size=(args.height, args.width))
    
    state_dict = torch.load(args.model_path, map_location='cpu')
    
    # compatibility with older model saves which used the 'extractor' name
    if 'extractor' in state_dict:
        weights = state_dict['extractor']
    elif 'disk' in state_dict:
        weights = state_dict['disk']
    else:
        raise KeyError('Incompatible weight file!')
    model = DISK(window=8, desc_dim=args.desc_dim)
    model.load_state_dict(weights)
    model = model.to(DEV)
    
    described_samples = extract(dataset, args.h5_path)
