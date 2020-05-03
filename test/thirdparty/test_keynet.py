# from https://raw.githubusercontent.com/axelBarroso/Key.Net/master/extract_multiscale_features.py

import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('keynet') 

import warnings # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427
warnings.filterwarnings('ignore', category=FutureWarning)

import os, sys, cv2
#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from os import path, mkdir
import argparse
import keyNet.aux.tools as aux
from skimage.transform import pyramid_gaussian
import HSequences_bench.tools.geometry_tools as geo_tools
import HSequences_bench.tools.repeatability_tools as rep_tools
from keyNet.model.keynet_architecture import *
import keyNet.aux.desc_aux_function as loss_desc
from keyNet.model.hardnet_pytorch import *
from keyNet.datasets.dataset_utils import read_bw_image
import torch


keynet_base_path='../../thirdparty/keynet/'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_directory(dir):
    if not path.isdir(dir):
        mkdir(dir)

def create_result_dir(path):
    directories = path.split('/')
    tmp = ''
    for idx, dir in enumerate(directories):
        tmp += (dir + '/')
        if idx == len(directories)-1:
            continue
        check_directory(tmp)

def extract_multiscale_features():

    parser = argparse.ArgumentParser(description='HSequences Extract Features')

    # parser.add_argument('--list-images', type=str, help='File containing the image paths for extracting features.',
    #                     required=True)

    parser.add_argument('--results-dir', type=str, default='extracted_features/',
                        help='The output path to save the extracted keypoint.')

    parser.add_argument('--network-version', type=str, default='KeyNet_default',
                        help='The Key.Net network version name')

    parser.add_argument('--checkpoint-det-dir', type=str, default=keynet_base_path + 'keyNet/pretrained_nets/KeyNet_default',
                        help='The path to the checkpoint file to load the detector weights.')

    parser.add_argument('--pytorch-hardnet-dir', type=str, default=keynet_base_path + 'keyNet/pretrained_nets/HardNet++.pth',
                        help='The path to the checkpoint file to load the HardNet descriptor weights.')

    # Detector Settings

    parser.add_argument('--num-filters', type=int, default=8,
                        help='The number of filters in each learnable block.')

    parser.add_argument('--num-learnable-blocks', type=int, default=3,
                        help='The number of learnable blocks after handcrafted block.')

    parser.add_argument('--num-levels-within-net', type=int, default=3,
                        help='The number of pyramid levels inside the architecture.')

    parser.add_argument('--factor-scaling-pyramid', type=float, default=1.2,
                        help='The scale factor between the multi-scale pyramid levels in the architecture.')

    parser.add_argument('--conv-kernel-size', type=int, default=5,
                        help='The size of the convolutional filters in each of the learnable blocks.')

    # Multi-Scale Extractor Settings

    parser.add_argument('--extract-MS', type=bool, default=True,
                        help='Set to True if you want to extract multi-scale features.')

    parser.add_argument('--num-points', type=int, default=1500,
                        help='The number of desired features to extract.')

    parser.add_argument('--nms-size', type=int, default=15,
                        help='The NMS size for computing the validation repeatability.')

    parser.add_argument('--border-size', type=int, default=15,
                        help='The number of pixels to remove from the borders to compute the repeatability.')

    parser.add_argument('--order-coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use yxsr or xysr.')

    parser.add_argument('--random-seed', type=int, default=12345,
                        help='The random seed value for TensorFlow and Numpy.')

    parser.add_argument('--pyramid_levels', type=int, default=5,
                        help='The number of downsample levels in the pyramid.')

    parser.add_argument('--upsampled-levels', type=int, default=1,
                        help='The number of upsample levels in the pyramid.')

    parser.add_argument('--scale-factor-levels', type=float, default=np.sqrt(2),
                        help='The scale factor between the pyramid levels.')

    parser.add_argument('--scale-factor', type=float, default=2.,
                        help='The scale factor to extract patches before descriptor.')

    # GPU Settings

    parser.add_argument('--gpu-memory-fraction', type=float, default=0.3,
                        help='The fraction of GPU used by the script.')

    parser.add_argument('--gpu-visible-devices', type=str, default="0",
                        help='Set CUDA_VISIBLE_DEVICES variable.')

    args = parser.parse_known_args()[0]

    # remove verbose bits from tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Set CUDA GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices

    version_network_name = args.network_version

    if not args.extract_MS:
        args.pyramid_levels = 0
        args.upsampled_levels = 0

    print('Extract features for : ' + version_network_name)

    aux.check_directory(args.results_dir)
    aux.check_directory(os.path.join(args.results_dir, version_network_name))

    def extract_features(image):

        pyramid = pyramid_gaussian(image, max_layer=args.pyramid_levels, downscale=args.scale_factor_levels)

        score_maps = {}
        for (j, resized) in enumerate(pyramid):
            im = resized.reshape(1, resized.shape[0], resized.shape[1], 1)

            feed_dict = {
                input_network: im,
                phase_train: False,
                dimension_image: np.array([1, im.shape[1], im.shape[2]], dtype=np.int32),
            }

            im_scores = sess.run(maps, feed_dict=feed_dict)

            im_scores = geo_tools.remove_borders(im_scores, borders=args.border_size)
            score_maps['map_' + str(j + 1 + args.upsampled_levels)] = im_scores[0, :, :, 0]

        if args.upsampled_levels:
            for j in range(args.upsampled_levels):
                factor = args.scale_factor_levels ** (args.upsampled_levels - j)
                up_image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

                im = np.reshape(up_image, (1, up_image.shape[0], up_image.shape[1], 1))

                feed_dict = {
                    input_network: im,
                    phase_train: False,
                    dimension_image: np.array([1, im.shape[1], im.shape[2]], dtype=np.int32),
                }

                im_scores = sess.run(maps, feed_dict=feed_dict)

                im_scores = geo_tools.remove_borders(im_scores, borders=args.border_size)
                score_maps['map_' + str(j + 1)] = im_scores[0, :, :, 0]

        im_pts = []
        for idx_level in range(levels):

            scale_value = (args.scale_factor_levels ** (idx_level - args.upsampled_levels))
            scale_factor = 1. / scale_value

            h_scale = np.asarray([[scale_factor, 0., 0.], [0., scale_factor, 0.], [0., 0., 1.]])
            h_scale_inv = np.linalg.inv(h_scale)
            h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

            num_points_level = point_level[idx_level]
            if idx_level > 0:
                res_points = int(np.asarray([point_level[a] for a in range(0, idx_level + 1)]).sum() - len(im_pts))
                num_points_level = res_points

            im_scores = rep_tools.apply_nms(score_maps['map_' + str(idx_level + 1)], args.nms_size)
            im_pts_tmp = geo_tools.get_point_coordinates(im_scores, num_points=num_points_level, order_coord='xysr')

            im_pts_tmp = geo_tools.apply_homography_to_points(im_pts_tmp, h_scale_inv)

            if not idx_level:
                im_pts = im_pts_tmp
            else:
                im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)

        if args.order_coord == 'yxsr':
            im_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], im_pts)))

        im_pts = im_pts[(-1 * im_pts[:, 3]).argsort()]
        im_pts = im_pts[:args.num_points]

        # Extract descriptor from features
        descriptors = []
        im = image.reshape(1, image.shape[0], image.shape[1], 1)
        for idx_desc_batch in range(int(len(im_pts) / 250 + 1)):
            points_batch = im_pts[idx_desc_batch * 250: (idx_desc_batch + 1) * 250]

            if not len(points_batch):
                break

            feed_dict = {
                input_network: im,
                phase_train: False,
                kpts_coord: points_batch[:, :2],
                kpts_scale: args.scale_factor * points_batch[:, 2],
                kpts_batch: np.zeros(len(points_batch)),
                dimension_image: np.array([1, im.shape[1], im.shape[2]], dtype=np.int32),
            }

            patch_batch = sess.run(input_patches, feed_dict=feed_dict)
            patch_batch = np.reshape(patch_batch, (patch_batch.shape[0], 1, 32, 32))
            data_a = torch.from_numpy(patch_batch)
            data_a = data_a.cuda()
            data_a = Variable(data_a)
            with torch.no_grad():
                out_a = model(data_a)
            desc_batch = out_a.data.cpu().numpy().reshape(-1, 128)
            if idx_desc_batch == 0:
                descriptors = desc_batch
            else:
                descriptors = np.concatenate([descriptors, desc_batch], axis=0)

        return im_pts, descriptors

    with tf.Graph().as_default():

        tf.set_random_seed(args.random_seed)

        with tf.name_scope('inputs'):

            # Define the input tensor shape
            tensor_input_shape = (None, None, None, 1)

            input_network = tf.placeholder(dtype=tf.float32, shape=tensor_input_shape, name='input_network')
            dimension_image = tf.placeholder(dtype=tf.int32, shape=(3,), name='dimension_image')
            kpts_coord = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='kpts_coord')
            kpts_batch = tf.placeholder(dtype=tf.int32, shape=(None,), name='kpts_batch')
            kpts_scale = tf.placeholder(dtype=tf.float32, name='kpts_scale')
            phase_train = tf.placeholder(tf.bool, name='phase_train')

        with tf.name_scope('model_deep_detector'):

            deep_architecture = keynet(args)
            output_network = deep_architecture.model(input_network, phase_train, dimension_image, reuse=False)
            maps = tf.nn.relu(output_network['output'])

        # Extract Patches from inputs:
        input_patches = loss_desc.build_patch_extraction(kpts_coord, kpts_batch, input_network, kpts_scale=kpts_scale)

        # Define Pytorch HardNet
        model = HardNet()
        checkpoint = torch.load(args.pytorch_hardnet_dir)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.cuda()

        # Define variables
        detect_var = [v for v in tf.trainable_variables(scope='model_deep_detector')]

        if os.listdir(args.checkpoint_det_dir):
            init_assign_op_det, init_feed_dict_det = tf_contrib.framework.assign_from_checkpoint(
                tf.train.latest_checkpoint(args.checkpoint_det_dir), detect_var)

        point_level = []
        tmp = 0.0
        factor_points = (args.scale_factor_levels ** 2)
        levels = args.pyramid_levels + args.upsampled_levels + 1
        for idx_level in range(levels):
            tmp += factor_points ** (-1 * (idx_level - args.upsampled_levels))
            point_level.append(args.num_points * factor_points ** (-1 * (idx_level - args.upsampled_levels)))

        point_level = np.asarray(list(map(lambda x: int(x / tmp), point_level)))

        # GPU Usage
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            if os.listdir(args.checkpoint_det_dir):
                sess.run(init_assign_op_det, init_feed_dict_det)

            # # read image and extract keypoints and descriptors
            # f = open(args.list_images, "r")
            # for path_to_image in f:
            #     path = path_to_image.split('\n')[0]

            #     if not os.path.exists(path):
            #         print('[ERROR]: File {0} not found!'.format(path))
            #         return

            #     create_result_dir(os.path.join(args.results_dir, version_network_name, path))

            if True: 
                path = 'kitti06-12-color.png'
                #im = read_bw_image(path)
                im = cv2.imread('../data/kitti06-12-color.png',cv2.IMREAD_GRAYSCALE)

                im = im.astype(float) / im.max()

                im_pts, descriptors = extract_features(im)
                
                print('# extracted points:',len(im_pts))

                file_name = os.path.join(args.results_dir, version_network_name, path)+'.kpt'
                np.save(file_name, im_pts)

                file_name = os.path.join(args.results_dir, version_network_name, path)+'.dsc'
                np.save(file_name, descriptors)


if __name__ == '__main__':
    extract_multiscale_features()
