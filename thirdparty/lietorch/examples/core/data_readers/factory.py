
import pickle
import os
import os.path as osp

# RGBD-Dataset
from .tartan import TartanAir
from .nyu2 import NYUv2
from .eth3d import ETH3D
from .scannet import ScanNet

# streaming datasets for inference
from .eth3d import ETH3DStream
from .tum import TUMStream
from .tartan import TartanAirStream


def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    dataset_map = {
        'tartan': (TartanAir, 1),
        'nyu': (NYUv2, 2),
        'eth': (ETH3D, 5),
        'scannet': (ScanNet, 1)}

    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)
        db *= dataset_map[key][1]

        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)
            

def create_datastream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """

    from torch.utils.data import DataLoader

    if osp.isfile(osp.join(dataset_path, 'calibration.txt')):
        db = ETH3DStream(dataset_path, **kwargs)

    elif osp.isfile(osp.join(dataset_path, 'rgb.txt')):
        db = TUMStream(dataset_path, **kwargs)
    
    elif osp.isdir(osp.join(dataset_path, 'image_left')):
        db = TartanStream(dataset_path, **kwargs)

    stream = DataLoader(db, shuffle=False, batch_size=1, num_workers=4)
    return stream



