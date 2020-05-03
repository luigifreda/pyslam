import os
import glob
import h5py
import tensorflow as tf

from utils.common import Notify
from .base_dataset import BaseDataset


class Yfcc(BaseDataset):
    default_config = {
        'num_parallel_calls': 10, 'truncate': None
    }

    def _init_dataset(self, **config):
        print(Notify.INFO, "Initializing dataset:", config['data_name'], Notify.ENDC)
        base_path = config['data_root']
        seq_paths = os.listdir(base_path)
        image_paths = []
        for tmp_seq in seq_paths:
            dump_folder = os.path.join(config['dump_root'], tmp_seq)
            if not os.path.exists(dump_folder):
                os.makedirs(dump_folder)
            for data_split in config['data_split']:
                seq_path = os.path.join(base_path, tmp_seq, data_split, 'images')
                image_paths.extend(glob.glob(os.path.join(seq_path, '*.jpg')))
        if config['truncate'] is not None:
            print(Notify.WARNING, "Truncate from",
                  config['truncate'][0], "to", config['truncate'][1], Notify.ENDC)
            image_paths = image_paths[config['truncate'][0]:config['truncate'][1]]
        seq_names = [i.split('/')[-4] for i in image_paths]
        image_names = [os.path.splitext(os.path.basename(i))[0] for i in image_paths]
        dump_paths = [os.path.join(config['dump_root'], seq_names[i],
                                   image_names[i] + '.h5') for i in range(len(image_paths))]
        print(Notify.INFO, "Found images:", len(image_paths), Notify.ENDC)

        self.data_length = len(image_paths)
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])

        files = {'image_paths': image_paths, 'dump_paths': dump_paths}
        return files

    def _format_data(self, data):
        image_path = data['image_path'].decode('utf-8')
        gen_path = image_path + self.config['post_format']['suffix']
        if not os.path.exists(gen_path):
            gen_f = h5py.File(gen_path, 'w')
            aug_feat = data['dump_data'][0]
            kpt_xy = data['dump_data'][1]
            _ = gen_f.create_dataset('descriptors', data=aug_feat)
            _ = gen_f.create_dataset('keypoints', data=kpt_xy)
