import sys
import numpy as np

sys.path.append("../")

from ground_truth import groundtruth_factory

# convert kitti ground truth in a simple format which can be used with video datasets
groundtruth_settings = {}
groundtruth_settings['type']='kitti'
groundtruth_settings['base_path'] ='/home/luigi/Work/datasets/rgbd_datasets/kitti/dataset'
groundtruth_settings['name'] = '06'

def main(settings = groundtruth_settings, out_filename = 'groundtruth.txt'):
    print(f'converting {groundtruth_settings}')
    grountruth = groundtruth_factory(groundtruth_settings)
    grountruth.convertToSimpleDataset()

if __name__ == '__main__':
    main()