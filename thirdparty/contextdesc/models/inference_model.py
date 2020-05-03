import tensorflow as tf

from .cnn_wrapper.descnet import GeoDesc, DenseGeoDesc
from .cnn_wrapper.augdesc import MatchabilityPrediction
from .cnn_wrapper.augdesc import VisualContext, LightContextNormalization
from utils.spatial_transformer import transformer_crop


def inference(config):
    """Model for patch description."""
    output_tensors = {}
    if config['dense_desc']:
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(
            None, None, None, 1), name='ph_img')
    else:
        input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(
            None, 32, 32), name='ph_patch')
        input_tensor = tf.reshape(input_tensor, (-1, 32, 32, 1))
        # patch standardization
    mean, variance = tf.nn.moments(tf.cast(input_tensor, tf.float32), axes=[1, 2], keep_dims=True)
    input_tensor = tf.nn.batch_normalization(input_tensor, mean, variance, None, None, 1e-5)

    if config['aug'] or config['dense_desc']:
        kpt_param = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 6), name='ph_kpt_param')
        kpt_xy = tf.concat((kpt_param[:, :, 2, None], kpt_param[:, :, 5, None]), axis=-1)

    if config['dense_desc']:
        kpt_theta = tf.reshape(kpt_param, (tf.shape(kpt_param)[0], tf.shape(kpt_param)[1], 2, 3))
        config_dict = {}
        config_dict['pert_theta'] = kpt_theta
        config_dict['patch_sampler'] = transformer_crop
        tower = DenseGeoDesc({'data': input_tensor, 'kpt_coord': kpt_xy},
                             is_training=False, reuse=False, **config_dict)
    else:
        tower = GeoDesc({'data': input_tensor}, is_training=False, reuse=False)

    if config['aug']:
        with tf.compat.v1.variable_scope('kpt_m'):
            inter_feat = tower.get_output_by_name('conv5')
            kpt_m_tower = MatchabilityPrediction(
                {'data': inter_feat}, is_training=False, reuse=False)
            kpt_m = kpt_m_tower.get_output()
            kpt_m = tf.reshape(kpt_m, (1, -1, 1))

    tower_output = tower.get_output_by_name('conv6')
    feat = tf.reshape(tower_output, (1, -1, tower_output.get_shape()[-1]))
    aug_feat = [feat]

    if config['aug']:
        img_feat = tf.compat.v1.placeholder(
            tf.float32, shape=(1, None, None, config['reg_feat_dim']), name='ph_img_feat')
        grid_pts = tf.compat.v1.placeholder(
            tf.float32, shape=(1, None, None, 2), name='ph_grid_pts')

        with tf.compat.v1.variable_scope('vis_context'):
            pt_tower = VisualContext(
                {'grid_pts': grid_pts, 'img_feat': img_feat,
                 'local_feat': feat, 'kpt_param': kpt_xy}, is_training=False, reuse=False)
        photo_feat = pt_tower.get_output()
        aug_feat.append(photo_feat)

    if config['aug']:
        kpt_m = tf.reshape(kpt_m_tower.get_output_by_name('kpt_m_rescale'), [1, -1, 1])
        with tf.compat.v1.variable_scope('geo_context'):
            points = tf.concat([kpt_m, kpt_xy], axis=2)
            feat_trans = LightContextNormalization({'points': tf.expand_dims(points, axis=2),
                                                    'local_feat': feat},
                                                   is_training=False, reuse=False)
            geo_feat = feat_trans.get_output()
            aug_feat.append(geo_feat)
        kpt_m = tf.squeeze(kpt_m, axis=0)
        output_tensors['kpt_m'] = kpt_m

    aug_feat = tf.add_n(aug_feat)
    aug_feat = tf.nn.l2_normalize(aug_feat, axis=-1, name='l2norm')
    aug_feat = tf.squeeze(aug_feat, axis=0)

    output_tensors['local_feat'] = aug_feat
    return output_tensors
