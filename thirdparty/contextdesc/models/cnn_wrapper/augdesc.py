from .network import Network

import tensorflow as tf


class VisualContext(Network):
    """Visual context feature fusion."""

    def _interpolate(self, xy1, xy2, points2):
        batch_size = tf.shape(xy1)[0]
        ndataset1 = tf.shape(xy1)[1]

        eps = 1e-6
        dist_mat = tf.matmul(xy1, xy2, transpose_b=True)
        norm1 = tf.reduce_sum(xy1 * xy1, axis=-1, keepdims=True)
        norm2 = tf.reduce_sum(xy2 * xy2, axis=-1, keepdims=True)
        dist_mat = tf.sqrt(norm1 - 2 * dist_mat + tf.linalg.matrix_transpose(norm2) + eps)
        dist, idx = tf.math.top_k(tf.negative(dist_mat), k=3)

        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        idx = tf.reshape(idx, (batch_size, -1))
        nn_points = tf.batch_gather(points2, idx)
        nn_points = tf.reshape(nn_points, (batch_size, ndataset1, 3, points2.get_shape()[-1].value))
        interpolated_points = tf.reduce_sum(weight[..., tf.newaxis] * nn_points, axis=-2)

        return interpolated_points

    def _mlp(module, inputs, mlp, last_relu=True, reuse=False, name='conv_%d'):
        new_points = tf.expand_dims(inputs, axis=2)
        for i, num_out_channel in enumerate(mlp):
            if (i == len(mlp) - 1) and not last_relu:
                new_points = module.conv_bn(new_points, 1, num_out_channel,
                                            1, padding='VALID', relu=False, reuse=reuse, name=name % (i))
            else:
                new_points = module.conv_bn(new_points, 1, num_out_channel,
                                            1, padding='VALID', reuse=reuse, name=name % (i))
        new_points = tf.squeeze(new_points, axis=2)
        return new_points

    def setup(self):
        grid_pts = self.inputs['grid_pts']
        local_feat = self.inputs['local_feat']
        vis_context_feat = self.inputs['img_feat']
        kpt_param = self.inputs['kpt_param']

        batch_size = tf.shape(vis_context_feat)[0]
        img_feat_dim = vis_context_feat.get_shape()[-1].value
        out_vis_context = None

        (self.feed('img_feat')
         .reshape((batch_size, -1, 1, img_feat_dim))
         .conv(1, 1024, 1, relu=False, name='conv1')
         .context_normalization(name='conv1_cn')
         .batch_normalization(relu=True, name='conv1_bn')
         .conv(1, 512, 1, relu=False, name='conv2')
         .context_normalization(name='conv2_cn')
         .batch_normalization(relu=True, name='conv2_bn')
         .squeeze(axis=2, name='squeeze'))

        trans_vis_context_feat = self.layers['squeeze']
        inter_vis_context_feat = self._interpolate(
            kpt_param,
            tf.reshape(grid_pts, (batch_size, -1, 2)),
            trans_vis_context_feat)

        fused_feat = tf.concat((inter_vis_context_feat, local_feat), axis=-1)
        out_vis_context = self._mlp(fused_feat, [512, 256, 128], name='fuse_photo_context_%d')
        self.terminals.append(out_vis_context)


class MatchabilityPrediction(Network):
    """Matchability prediction."""

    def setup(self):
        (self.feed('data')
         .conv_bn(8, 128, 1, padding='VALID', name='kpt_m_conv0')
         .conv_bn(1, 32, 1, padding='VALID', name='kpt_m_conv1')
         .conv_bn(1, 32, 1, padding='VALID', name='kpt_m_conv2')
         .conv(1, 1, 1, biased=True, relu=False, padding='VALID', name='kpt_m')
         .fc(1, biased=True, relu=False, flatten=False)
         .tanh(name='kpt_m_rescale'))


class LightContextNormalization(Network):
    """Context normalization definition."""

    def setup(self):
        (self.feed('points')
         .conv(1, 128, 1, relu=False, name='dim_control')
         .context_normalization(name='cn1_cn1')
         .batch_normalization(relu=True, name='cn1_bn1')
         .conv(1, 128, 1, relu=False, name='cn1_conv1'))

        (self.feed('dim_control', 'cn1_conv1')
         .add(name='res1'))

        (self.feed('res1')
         .context_normalization(name='cn2_cn1')
         .batch_normalization(relu=True, name='cn2_bn1')
         .conv(1, 128, 1, relu=False, name='cn2_conv1'))

        (self.feed('res1', 'cn2_conv1')
         .add(name='res2'))

        (self.feed('res2')
         .context_normalization(name='cn3_cn1')
         .batch_normalization(relu=True, name='cn3_bn1')
         .conv(1, 128, 1, relu=False, name='cn3_conv1'))

        (self.feed('res2', 'cn3_conv1')
         .add(name='res3'))

        (self.feed('res3')
         .context_normalization(name='cn4_cn1')
         .batch_normalization(relu=True, name='cn4_bn1')
         .conv(1, 128, 1, relu=False, name='cn4_conv1'))

        (self.feed('res3', 'cn4_conv1')
         .add(name='res4')
         .conv(1, 128, 1, relu=False, name='context_trans')
         .squeeze(axis=2, name='context_feat'))
