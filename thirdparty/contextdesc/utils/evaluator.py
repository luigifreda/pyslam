import numpy as np
import cv2
import tensorflow as tf


class Evaluator(object):
    def __init__(self, config):
        self.mutual_check = False
        self.err_thld = config['err_thld']
        self.matches = self.bf_matcher_graph()
        self.stats = {
            'i_avg_recall': 0,
            'v_avg_recall': 0,
            'all_avg_recall': 0,
        }

    def homo_trans(self, coord, H):
        kpt_num = coord.shape[0]
        homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
        proj_coord = np.matmul(H, homo_coord.T).T
        proj_coord = proj_coord / proj_coord[:, 2][..., None]
        proj_coord = proj_coord[:, 0:2]
        return proj_coord

    def bf_matcher_graph(self):
        descriptors_a = tf.compat.v1.placeholder(tf.float32, (None, None), 'descriptor_a')
        descriptors_b = tf.compat.v1.placeholder(tf.float32, (None, None), 'descriptor_b')
        sim = tf.linalg.matmul(descriptors_a, descriptors_b, transpose_b=True)
        ids1 = tf.range(0, tf.shape(sim)[0])
        nn12 = tf.math.argmax(sim, axis=1, output_type=tf.int32)
        if self.mutual_check:
            nn21 = tf.math.argmax(sim, axis=0, output_type=tf.int32)
            mask = tf.equal(ids1, tf.gather(nn21, nn12))
            matches = tf.stack([tf.boolean_mask(ids1, mask), tf.boolean_mask(nn12, mask)])
        else:
            matches = tf.stack([ids1, nn12])
        return matches

    def bf_matcher(self, sess, descriptors_a, descriptors_b):
        input_dict = {
            "descriptor_a:0": descriptors_a,
            "descriptor_b:0": descriptors_b
        }
        matches = sess.run(self.matches, input_dict)
        return matches.T

    def feature_matcher(self, sess, ref_feat, test_feat):
        matches = self.bf_matcher(sess, ref_feat, test_feat)
        matches = [cv2.DMatch(matches[i][0], matches[i][1], 0) for i in range(matches.shape[0])]
        return matches

    def get_inlier_matches(self, ref_coord, test_coord, putative_matches, gt_homo, scaling=1.):
        p_ref_coord = np.float32([ref_coord[m.queryIdx] for m in putative_matches]) / scaling
        p_test_coord = np.float32([test_coord[m.trainIdx] for m in putative_matches]) / scaling

        proj_p_ref_coord = self.homo_trans(p_ref_coord, gt_homo)
        dist = np.sqrt(np.sum(np.square(proj_p_ref_coord - p_test_coord[:, 0:2]), axis=-1))
        inlier_mask = dist <= self.err_thld
        inlier_matches = [putative_matches[z] for z in np.nonzero(inlier_mask)[0]]
        return inlier_matches

    def get_gt_matches(self, ref_coord, test_coord, gt_homo, scaling=1.):
        ref_coord = ref_coord / scaling
        test_coord = test_coord / scaling
        proj_ref_coord = self.homo_trans(ref_coord, gt_homo)

        pt0 = np.expand_dims(proj_ref_coord, axis=1)
        pt1 = np.expand_dims(test_coord, axis=0)
        norm = np.linalg.norm(pt0 - pt1, ord=None, axis=2)
        min_dist = np.min(norm, axis=1)
        gt_num = np.sum(min_dist <= self.err_thld)
        return gt_num
