import os
import glob
import h5py
import tensorflow as tf

from utils.common import Notify
from .base_dataset import BaseDataset


class Imw2020(BaseDataset):
    default_config = {
        'num_parallel_calls': 10, 'truncate': None
    }

    def _init_dataset(self, **config):
        print(Notify.INFO, "Initializing dataset:", config['data_name'], Notify.ENDC)
        if config['data_split'] == 'val':
            proj_paths = ['reichstag', 'sacre_coeur', 'st_peters_square']
            seq_paths = [os.path.join(i, 'set_100', 'images') for i in proj_paths]
        elif config['data_split'] == 'test':
            proj_paths = ['british_museum', 'lincoln_memorial_statue', 'milan_cathedral',
                     'piazza_san_marco', 'st_pauls_cathedral', 'florence_cathedral_side',
                     'london_bridge', 'mount_rushmore', 'sagrada_familia', 'united_states_capitol']
            seq_paths = proj_paths
        else:
            raise NotImplementedError

        base_path = config['data_root']
        image_paths = []
        for idx, val in enumerate(seq_paths):
            dump_folder = os.path.join(config['dump_root'], proj_paths[idx])
            if not os.path.exists(dump_folder):
                os.makedirs(dump_folder)
            seq_path = os.path.join(base_path, val)
            image_paths.extend(glob.glob(os.path.join(seq_path, '*.jpg')))

        if config['truncate'] is not None:
            print(Notify.WARNING, "Truncate from",
                  config['truncate'][0], "to", config['truncate'][1], Notify.ENDC)
            image_paths = image_paths[config['truncate'][0]:config['truncate'][1]]

        if config['data_split'] == 'val':
            seq_names = [i.split('/')[-4] for i in image_paths]
        elif config['data_split'] == 'test':
            seq_names = [i.split('/')[-2] for i in image_paths]
        else:
            raise NotImplementedError
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
        dump_path = data['dump_path'].decode('utf-8')
        seq_name = dump_path.split('/')[-2]
        basename = os.path.splitext(os.path.basename(dump_path))[0]
        if not os.path.exists(self.config['submission_root']):
            os.mkdir(self.config['submission_root'])
        seq_folder = os.path.join(self.config['submission_root'], seq_name)
        if not os.path.exists(seq_folder):
            os.mkdir(seq_folder)
        h5_kpt = os.path.join(seq_folder, 'keypoints.h5')
        h5_desc = os.path.join(seq_folder, 'descriptors.h5')
        h5_score = os.path.join(seq_folder, 'scores.h5')

        if not os.path.exists(h5_desc) and not os.path.exists(h5_kpt) and not os.path.exists(h5_score):
            gen_kpt_f = h5py.File(h5_kpt, 'w')
            gen_desc_f = h5py.File(h5_desc, 'w')
            gen_score_f = h5py.File(h5_score, 'w')
        else:
            gen_kpt_f = h5py.File(h5_kpt, 'a')
            gen_desc_f = h5py.File(h5_desc, 'a')
            gen_score_f = h5py.File(h5_score, 'a')

        if basename not in gen_kpt_f and basename not in gen_desc_f:
            feat = data['dump_data'][0]
            kpt = data['dump_data'][1]
            score = data['dump_data'][2]
            _ = gen_kpt_f.create_dataset(basename, data=kpt)
            _ = gen_desc_f.create_dataset(basename, data=feat)
            _ = gen_score_f.create_dataset(basename, data=score)