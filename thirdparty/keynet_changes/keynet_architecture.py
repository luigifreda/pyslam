import math
import numpy as np
if False:
    import tensorflow as tf
    import tensorflow.contrib as tf_contrib     
else: 
    # from https://stackoverflow.com/questions/56820327/the-name-tf-session-is-deprecated-please-use-tf-compat-v1-session-instead
    import tensorflow.compat.v1 as tf
    import tensorflow.contrib as tf_contrib 

def gaussian_multiple_channels(num_channels, sigma):

    r = 2*sigma
    size = 2*r+1
    size = int(math.ceil(size))
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = r

    gaussian = np.float32((np.exp(-1 * (((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2))))) / ((2 * math.pi * (sigma ** 2))**0.5))
    weights = np.zeros((size, size, num_channels, num_channels), dtype=np.float32)

    for i in range(num_channels):
        weights[:, :, i, i] = gaussian

    return weights


def ones_multiple_channels(size, num_channels):

    ones = np.ones((size, size))
    weights = np.zeros((size, size, num_channels, num_channels), dtype=np.float32)

    for i in range(num_channels):
        weights[:, :, i, i] = ones

    return weights


def grid_indexes(size):

    weights = np.zeros((size, size, 1, 2), dtype=np.float32)

    columns = []
    for idx in range(1, 1+size):
        columns.append(np.ones((size))*idx)
    columns = np.asarray(columns)

    rows = []
    for idx in range(1, 1+size):
        rows.append(np.asarray(range(1, 1+size)))
    rows = np.asarray(rows)

    weights[:, :, 0, 0] = columns
    weights[:, :, 0, 1] = rows

    return weights


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def linear_upsample_weights(half_factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with linear filter
    initialization.
    """

    filter_size = get_kernel_size(half_factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = np.ones((filter_size, filter_size))
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights


def create_derivatives_kernel():
    # Sobel derivative 3x3 X
    kernel_filter_dx_3 = np.float32(np.asarray([[-1, 0, 1],
                                                [-2, 0, 2],
                                                [-1, 0, 1]]))
    kernel_filter_dx_3 = kernel_filter_dx_3[..., np.newaxis]
    kernel_filter_dx_3 = kernel_filter_dx_3[..., np.newaxis]

    # Sobel derivative 3x3 Y
    kernel_filter_dy_3 = np.float32(np.asarray([[-1, -2, -1],
                                                [0, 0, 0],
                                                [1, 2, 1]]))
    kernel_filter_dy_3 = kernel_filter_dy_3[..., np.newaxis]
    kernel_filter_dy_3 = kernel_filter_dy_3[..., np.newaxis]

    return kernel_filter_dx_3, kernel_filter_dy_3


class keynet(object):
    def __init__(self, args, MSIP_sizes=[]):

        # Start Key.Net architecture
        self.pyramid_levels = args.num_levels_within_net
        self.factor_scaling = args.factor_scaling_pyramid
        self.num_blocks = args.num_learnable_blocks
        self.num_filters = args.num_filters
        self.conv_kernel_size = args.conv_kernel_size
        self.ksize = args.nms_size

        self.batch_size = 16
        self.patch_size = 32

        tf.set_random_seed(args.random_seed)
        np.random.seed(args.random_seed)

        name_scope = tf_contrib.framework.get_name_scope()

        # Smooth Gausian Filter
        gaussian_avg = gaussian_multiple_channels(1, 1.5)
        self.gaussian_avg = tf.constant(gaussian_avg, name=name_scope + '_Gaussian_avg')

        # Sobel derivatives
        kernel_x, kernel_y = create_derivatives_kernel()
        self.kernel_filter_dx = tf.constant(kernel_x, name=name_scope + '_kernel_filter_dx')
        self.kernel_filter_dy = tf.constant(kernel_y, name=name_scope + '_kernel_filter_dy')

        # create_kernels
        self.kernels = {}

        if MSIP_sizes != []:
            self.create_kernels(MSIP_sizes, name_scope)

        if 8 not in MSIP_sizes:
            self.create_kernels([8], name_scope)

    def create_kernels(self, MSIP_sizes, name_scope):

        # Grid Indexes for MSIP
        for ksize in MSIP_sizes:

            ones_kernel = ones_multiple_channels(ksize, 1)
            indexes_kernel = grid_indexes(ksize)
            upsample_filter_np = linear_upsample_weights(int(ksize / 2), 1)

            self.ones_kernel = tf.constant(ones_kernel, name=name_scope +'_Ones_kernel_'+str(ksize))
            self.kernels['ones_kernel_'+str(ksize)] = self.ones_kernel

            self.upsample_filter_np = tf.constant(upsample_filter_np, name=name_scope+'_upsample_filter_np_'+str(ksize))
            self.kernels['upsample_filter_np_'+str(ksize)] = self.upsample_filter_np

            self.indexes_kernel = tf.constant(indexes_kernel, name=name_scope +'_indexes_kernel_'+str(ksize))
            self.kernels['indexes_kernel_'+str(ksize)] = self.indexes_kernel

            index_size = int(self.patch_size/ksize)
            zeros = np.zeros((self.batch_size, index_size, index_size, 2))
            zeros = tf.constant(zeros, name=name_scope +'zeros_ind_kernel_'+str(ksize), dtype=tf.float32)
            self.kernels['zeros_ind_kernel_'+str(ksize)] = zeros

            ones = np.ones((self.batch_size, index_size, index_size, 2))
            ones = tf.constant(ones, name=name_scope +'ones_ind_kernel_'+str(ksize), dtype=tf.float32)
            self.kernels['ones_ind_kernel_'+str(ksize)] = ones

    def get_kernels(self):
        return self.kernels

    def model(self, input_data, is_training, dim, reuse=False, train_score=True, H_vector=[], apply_homography = False):

        features, network = self.compute_features(input_data, dim, reuse, is_training)

        features = tf.layers.batch_normalization(inputs=features, scale=True, training=is_training,
                                                 name=tf_contrib.framework.get_name_scope() + '_batch_final', reuse=reuse)

        output = self.conv_block(features, 'last_layer', reuse, is_training, num_filters=1, size_kernel=self.conv_kernel_size, batchnorm=False, activation_function=False)

        if apply_homography:
            output = self.transform_map(output, H_vector)

        network['input_data'] = input_data
        network['features'] = features
        network['output'] = output

        return network

    def compute_handcrafted_features(self, image, network, idx, name_scope):

        # Sobel_conv_derivativeX
        dx = tf.nn.conv2d(image, self.kernel_filter_dx, strides=[1, 1, 1, 1], padding='SAME')
        dxx = tf.nn.conv2d(dx, self.kernel_filter_dx, strides=[1, 1, 1, 1], padding='SAME')
        dx2 = tf.multiply(dx, dx)

        # Sobel_conv_derivativeY
        dy = tf.nn.conv2d(image, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')
        dyy = tf.nn.conv2d(dy, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')
        dy2 = tf.multiply(dy, dy)

        dxy = tf.nn.conv2d(dx, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')

        dxdy = tf.multiply(dx, dy)
        dxxdyy = tf.multiply(dxx, dyy)
        dxy2 = tf.multiply(dxy, dxy)

        # Concatenate Handcrafted Features
        features_t = tf.concat([dx, dx2, dxx, dy, dy2, dyy, dxdy, dxxdyy, dxy, dxy2], axis=3)

        network['dx_' + str(idx + 1)] = dx
        network['dx2_' + str(idx + 1)] = dx2
        network['dy_' + str(idx + 1)] = dy
        network['dy2_' + str(idx + 1)] = dy2
        network['dxdy_' + str(idx + 1)] = dxdy
        network['dxxdyy_' + str(idx + 1)] = dxxdyy
        network['dxy_' + str(idx + 1)] = dxy
        network['dxy2_' + str(idx + 1)] = dxy2
        network['dx2dy2_' + str(idx + 1)] = dx2+dy2

        return features_t, network

    def local_norm_image(self, x, k_size=65, eps=1e-10):
        pad = int(k_size / 2)
        x_pad = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'REFLECT')
        x_mean = tf.nn.avg_pool(x_pad, ksize=[1, k_size, k_size, 1], strides=[1, 1, 1, 1], padding='VALID')
        x2_mean = tf.nn.avg_pool(tf.pow(x_pad, 2.0), ksize=[1, k_size, k_size, 1], strides=[1, 1, 1, 1],
                                 padding='VALID')
        x_std = (tf.sqrt(tf.abs(x2_mean - x_mean * x_mean)) + eps)
        x_norm = (x - x_mean) / (1.+x_std)

        return x_norm

    def compute_features(self, input_data, dim, reuse, is_training):

        dim_float = tf.cast(dim, tf.float32)
        features = []
        network = {}

        for idx_level in range(self.pyramid_levels):

            if idx_level == 0:
                input_data_smooth = input_data
            else:
                input_data_smooth = tf.nn.conv2d(input_data, self.gaussian_avg, strides=[1, 1, 1, 1], padding='SAME')

            input_data_resized = tf.image.resize_images(input_data_smooth, size=tf.cast(
                (dim_float[1] / (self.factor_scaling ** idx_level), dim_float[2] / (self.factor_scaling ** idx_level)),
                tf.int32), align_corners=True, method=0)

            input_data_resized = self.local_norm_image(input_data_resized)

            features_t, network = self.compute_handcrafted_features(input_data_resized, network, idx_level,
                                                                    tf_contrib.framework.get_name_scope())

            for idx_layer in range(self.num_blocks):
                features_t = self.conv_block(features_t, str(idx_layer + 1), reuse or idx_level > 0, is_training,
                                             num_filters=self.num_filters, size_kernel=self.conv_kernel_size)

            features_t = tf.image.resize_images(features_t, size=tf.cast((dim_float[1], dim_float[2]), tf.int32),
                                                align_corners=True, method=0)

            if not idx_level:
                features = features_t
            else:
                features = tf.concat([features, features_t], axis=3)

        return features, network

    def conv_block(self, features, name, reuse, is_training, num_filters, size_kernel, batchnorm=True, activation_function=True):

        features = tf.layers.conv2d(inputs=features, filters=num_filters,
                                    kernel_size=size_kernel,
                                    strides=1, padding='SAME', use_bias=True,
                                    kernel_initializer=tf_contrib.layers.variance_scaling_initializer(),
                                    kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.1),
                                    data_format='channels_last',
                                    name=tf_contrib.framework.get_name_scope() + '_conv_'+name, reuse=reuse)

        if batchnorm:
            features = tf.layers.batch_normalization(inputs=features, scale=True, training=is_training,
                                                 name=tf_contrib.framework.get_name_scope() + '_batch_'+name, reuse=reuse)

        if activation_function:
            features = tf.nn.relu(features)

        return features

    def non_maximum_supression(self, map, thresh=0.):

        pooled = tf.nn.max_pool(map, ksize=[1, self.ksize, self.ksize, 1], strides=[1, 1, 1, 1], padding='SAME')
        mask_scores = tf.where(tf.equal(map, pooled), tf.ones_like(map), tf.zeros_like(map))
        mask_th = tf.where(tf.math.greater(map, thresh * tf.ones_like(map)), tf.ones_like(map), tf.zeros_like(map))

        scores_nms = tf.multiply(mask_scores, mask_th)
        scores_nms = tf.multiply(map, scores_nms)

        return scores_nms