#!/usr/bin/env python3

"""
Copyright 2019, Zixin Luo, HKUST.
Inference script.
"""

import sys
import os
from abc import ABCMeta, abstractmethod
import collections
import tensorflow as tf

sys.path.append('..')

from ..utils.tf import load_frozen_model, recoverer


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


class BaseModel(metaclass=ABCMeta):
    """Base model class."""

    @abstractmethod
    def _run(self, data, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def _construct_network(self):
        raise NotImplementedError

    def run_test_data(self, data, **kwargs):
        """"""
        out_data = self._run(data, **kwargs)
        return out_data

    def __init__(self, model_path, **config):
        self.model_path = model_path
        # Update config
        self.config = dict_update(getattr(self, 'default_config', {}), config)
        self._init_model()
        if model_path is None:
            print("No pretrained model specified!")
            self.sess = None
        else:
            ext = os.path.splitext(model_path)[1]

            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            sess_config.gpu_options.per_process_gpu_memory_fraction=0.333

            if ext.find('.pb') == 0:
                graph = load_frozen_model(self.model_path, print_nodes=False)
                self.sess = tf.compat.v1.Session(graph=graph, config=sess_config)
            elif ext.find('.ckpt') == 0:
                self.sess = tf.compat.v1.Session(config=sess_config)
                meta_graph_path = os.path.join(model_path + '.meta')
                if not os.path.exists(meta_graph_path):
                    self._construct_network()
                    recoverer(self.sess, model_path)
                else:
                    recoverer(self.sess, model_path, meta_graph_path)

    def close(self):
        if self.sess is not None:
            self.sess.close()
            tf.compat.v1.reset_default_graph()
