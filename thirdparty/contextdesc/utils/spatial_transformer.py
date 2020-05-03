"""
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf


def _meshgrid(height, width):
    with tf.variable_scope('_meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid


def locate_pts(U, xy, stack_batch, name='LocateXY'):
    """
    Sample points by bilinear interpolation given 2d-coordinates.
    Args:
        U: BxHxWxC
        theta: BxNx2
    Returns:
        output: (BxN)xC
    """
    def _interpolate(im, x, y, num_kpt):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = im.get_shape()[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(height - 1, 'int32')
            max_x = tf.cast(width - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

            # do sampling
            x_0 = tf.cast(tf.floor(x), 'int32')
            x_1 = x_0 + 1
            y_0 = tf.cast(tf.floor(y), 'int32')
            y_1 = y_0 + 1

            x_0 = tf.clip_by_value(x_0, zero, max_x)
            x_1 = tf.clip_by_value(x_1, zero, max_x)
            y_0 = tf.clip_by_value(y_0, zero, max_y)
            y_1 = tf.clip_by_value(y_1, zero, max_y)
            dim2 = width
            dim1 = width * height
            base = tf.tile(tf.expand_dims(tf.range(num_batch) * dim1, 1), [1, num_kpt])
            base = tf.reshape(base, [-1])
            base_y0 = base + y_0 * dim2
            base_y1 = base + y_1 * dim2
            idx_a = base_y0 + x_0
            idx_b = base_y1 + x_0
            idx_c = base_y0 + x_1
            idx_d = base_y1 + x_1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x_0, 'float32')
            x1_f = tf.cast(x_1, 'float32')
            y0_f = tf.cast(y_0, 'float32')
            y1_f = tf.cast(y_1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _transform(xy, input_dim):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(xy)[0]
            num_kpt = tf.shape(xy)[1]
            num_channels = input_dim.get_shape()[3]
            xy = tf.reshape(xy, (-1, 2))
            xy = tf.cast(xy, 'float32')
            x_s_flat = xy[:, 0]
            y_s_flat = xy[:, 1]
            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, num_kpt)
            if stack_batch:
                output = tf.reshape(input_transformed, tf.stack(
                    [num_batch * num_kpt, num_channels]))
            else:
                output = tf.reshape(input_transformed, tf.stack([num_batch, num_kpt, num_channels]))
            return output
    with tf.variable_scope(name):
        output = _transform(xy, U)
        return output


def transformer_crop(U, theta, out_size, stack_batch, name='SpatialTransformer', **kwargs):
    """
    Args:
        U: BxHxWxC
        theta: BxNx2x3
    Returns:
        output: (BxN)x(out_size)xC
    """
    def _interpolate(im, x, y, num_kpt, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = im.get_shape()[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(height - 1, 'int32')
            max_x = tf.cast(width - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

            # do sampling
            x_0 = tf.cast(tf.floor(x), 'int32')
            x_1 = x_0 + 1
            y_0 = tf.cast(tf.floor(y), 'int32')
            y_1 = y_0 + 1

            x_0 = tf.clip_by_value(x_0, zero, max_x)
            x_1 = tf.clip_by_value(x_1, zero, max_x)
            y_0 = tf.clip_by_value(y_0, zero, max_y)
            y_1 = tf.clip_by_value(y_1, zero, max_y)
            dim2 = width
            dim1 = width * height
            base = tf.tile(tf.expand_dims(tf.range(num_batch) * dim1, 1),
                           [1, num_kpt * out_height * out_width])
            base = tf.reshape(base, [-1])
            base_y0 = base + y_0 * dim2
            base_y1 = base + y_1 * dim2
            idx_a = base_y0 + x_0
            idx_b = base_y1 + x_0
            idx_c = base_y0 + x_1
            idx_d = base_y1 + x_1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x_0, 'float32')
            x1_f = tf.cast(x_1, 'float32')
            y0_f = tf.cast(y_0, 'float32')
            y1_f = tf.cast(y_1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(theta)[0]
            num_kpt = tf.shape(theta)[1]
            num_channels = input_dim.get_shape()[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch * num_kpt]))
            grid = tf.reshape(grid, tf.stack([num_batch * num_kpt, 3, -1]))
            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            t_g = tf.matmul(theta, grid)  # [BxN, 3, 3] * [BxN, 3, H*W]
            x_s = tf.slice(t_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(t_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, num_kpt, out_size)

            if stack_batch:
                output = tf.reshape(
                    input_transformed, tf.stack([num_batch * num_kpt, out_height, out_width, num_channels]))
            else:
                output = tf.reshape(
                    input_transformed, tf.stack([num_batch, num_kpt, out_height, out_width, num_channels]))
            return output
    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.coT_gm/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = im.get_shape()[1]
            width = im.get_shape()[2]
            channels = im.get_shape()[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(im.get_shape()[1] - 1, 'int32')
            max_x = tf.cast(im.get_shape()[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

            # do sampling
            x_0 = tf.cast(tf.floor(x), 'int32')
            x_1 = x_0 + 1
            y_0 = tf.cast(tf.floor(y), 'int32')
            y_1 = y_0 + 1

            x_0 = tf.clip_by_value(x_0, zero, max_x)
            x_1 = tf.clip_by_value(x_1, zero, max_x)
            y_0 = tf.clip_by_value(y_0, zero, max_y)
            y_1 = tf.clip_by_value(y_1, zero, max_y)
            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
            base_y0 = base + y_0 * dim2
            base_y1 = base + y_1 * dim2
            idx_a = base_y0 + x_0
            idx_b = base_y1 + x_0
            idx_c = base_y0 + x_1
            idx_d = base_y1 + x_1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x_0, 'float32')
            x1_f = tf.cast(x_1, 'float32')
            y0_f = tf.cast(y_0, 'float32')
            y1_f = tf.cast(y_1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            num_channels = input_dim.get_shape()[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            t_g = tf.matmul(theta, grid)
            x_s = tf.slice(t_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(t_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output
    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer

    Parameters
    ----------

    U : float
        tensor of inputs [num_batch,height,width,num_channels]
    thetas : float
        a set of transformations for each input [num_batch,num_transforms,6]
    out_size : int
        the size of the output [out_height,out_width]

    Returns: float
        Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
    """
    with tf.variable_scope(name):
        if len(U.get_shape()) == 3 and len(thetas.get_shape()) == 3:
            U = tf.expand_dims(U, axis=0)
            thetas = tf.expand_dims(thetas, axis=0)
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i] * num_transforms for i in xrange(num_batch)]
        input_U = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_U, thetas, out_size)


#if __name__ == '__main__':
#    import numpy as np
#    U = tf.constant(range(49), tf.float32)
#    U = tf.reshape(U, (1, 7, 7, 1))
#    U = tf.tile(U, (2, 1, 1, 1))
#    xy1 = tf.constant([0.5, 0.5, -0.5, -0.5, 0, 0, -0.25, -0.25])
#    xy1 = tf.reshape(xy1, (1, -1, 2))
#
#    xy2 = tf.constant([-0.5, -0.5, 0.5, -0.5, 0, 0, 0.25, 0.25])
#    xy2 = tf.reshape(xy2, (1, -1, 2))
#
#    xy = tf.concat((xy1, xy2), axis=0)
#
#    print(U, xy)
#    output = locate_pts(U, xy, True)
#    with tf.Session() as sess:
#        tf_out = sess.run([output, U])
#        print(tf_out[0])
#        print(np.squeeze(tf_out[1]))
