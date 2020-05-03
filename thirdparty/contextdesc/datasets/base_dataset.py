from abc import ABCMeta, abstractmethod
import collections
import tensorflow as tf
import numpy as np
import h5py


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class BaseDataset(metaclass=ABCMeta):
    """Base model class."""

    @abstractmethod
    def _init_dataset(self, **config):
        """Initialize the dataset and prepare data paths."""
        raise NotImplementedError

    @abstractmethod
    def _format_data(self, data):
        """Format the dumped data for later processing."""
        raise NotImplementedError

    def get_test_set(self):
        """Processed test set."""
        return self._get_set_generator()

    def format_data(self, data):
        return self._format_data(data)

    def __init__(self, **config):
        # Update config
        self.data_length = 0
        self.config = dict_update(getattr(self, 'default_config', {}), config)

        assert self.config['stage'] is not None

        if self.config['stage'] == 'reg':
            self.read_gray = False
        elif self.config['stage'] == 'loc':
            self.read_gray = True

        self.dataset = self._init_dataset(**self.config)

        with tf.device('/cpu:0'):
            self.tf_splits = self._get_data(self.dataset)
            self.tf_next = tf.compat.v1.data.make_one_shot_iterator(self.tf_splits).get_next()
        self.end_set = tf.errors.OutOfRangeError
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

    def _get_set_generator(self):
        while True:
            yield self.sess.run(self.tf_next)

    def _get_data(self, files):
        def _read_image(img_path):
            channels = 1 if self.read_gray else 3
            if 'all_jpeg' in self.config and self.config['all_jpeg']:
                img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=channels, dct_method='INTEGER_ACCURATE')
            else:
                img = tf.image.decode_image(tf.io.read_file(img_path), channels=channels)
            img.set_shape((None, None, channels))
            return tf.cast(img, tf.float32)

        def _read_dump(path):
            f = h5py.File(path, 'r')
            return (f['reg_feat'][()].astype(np.float32), f['loc_info'][()].astype(np.float32))

        def _read_gen_train(path):
            f = h5py.File(path, 'r')
            return (f['aug_feat'][()].astype(np.float32), 
                    f['loc_info'][()][:, 0:2].astype(np.float32),
                    f['loc_info'][()][:, 4].astype(np.float32))

        image_paths = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        dump_paths = tf.data.Dataset.from_tensor_slices(files['dump_paths'])

        if self.config['stage'] == 'loc' or self.config['stage'] == 'reg':
            images = image_paths.map(_read_image)
            data = tf.data.Dataset.zip(
                {'image': images, 'dump_path': dump_paths, 'image_path': image_paths})
        elif self.config['stage'] == 'aug':
            dump_data = dump_paths.map(lambda path: tf.numpy_function(
                _read_dump, [path], [tf.float32, tf.float32]))
            data = tf.data.Dataset.zip({'dump_data': dump_data, 'dump_path': dump_paths})
        elif self.config['stage'] == 'post_format':
            dump_data = dump_paths.map(lambda path: tf.numpy_function(
                _read_gen_train, [path], [tf.float32, tf.float32, tf.float32]))
            data = tf.data.Dataset.zip(
                {'dump_data': dump_data, 'dump_path': dump_paths, 'image_path': image_paths})
        else:
            raise NotImplementedError
        return data
