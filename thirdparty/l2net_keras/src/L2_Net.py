import os



import warnings # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427
warnings.filterwarnings('ignore', category=FutureWarning)

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Lambda

# from https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, save_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
if False:
    import tensorflow as tf
else: 
    # from https://stackoverflow.com/questions/56820327/the-name-tf-session-is-deprecated-please-use-tf-compat-v1-session-instead
    import tensorflow.compat.v1 as tf
    

from utils_tf import set_tf_logging
    

import pickle
import numpy as np
from LRN import LRN
import cv2


# get the location of this file!
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def build_cnn(weights):

    model = Sequential()

    model.add(ZeroPadding2D(1, input_shape=(32,32, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=2))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=2))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(Conv2D(128, kernel_size=(8, 8)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(LRN(alpha=256,k=0,beta=0.5,n=256))

    model.set_weights(weights)

    return model


def build_L2_net(net_name):

    python_net_data = pickle.load(open(__location__ + "/../python_net_data/" + net_name + ".p", "rb"))
    return build_cnn(python_net_data['weights']), build_cnn(python_net_data['weights_cen']), python_net_data['pix_mean'], python_net_data['pix_mean_cen']


def cal_L2Net_des(net_name, testPatchs, flagCS = False):

    """
    Get descriptors for one or more patches

    Parameters
    ----------
    net_name : string
        One of "L2Net-HP", "L2Net-HP+", "L2Net-LIB", "L2Net-LIB+", "L2Net-ND", "L2Net-ND+", "L2Net-YOS", "L2Net-YOS+",
    testPatchs : array
        A numpy array of image data with deimensions (?, 32, 32, 1), or if using central-surround with deimensions (?, 64, 64, 1)
    flagCS : boolean
        If True, use central-surround network

    Returns
    -------
    descriptor
        Numpy array with size (?, 128) or if using central-surround (?, 256)

    """

    model, model_cen, pix_mean, pix_mean_cen = build_L2_net(net_name)

    # print(model.summary())
    # print(model_cen.summary())

    if flagCS:

        testPatchsCen = testPatchs[:,16:48,16:48,:]
        testPatchsCen = testPatchsCen - pix_mean_cen
        testPatchsCen = np.array([(testPatchsCen[i] - np.mean(testPatchsCen[i]))/(np.std(testPatchsCen[i]) + 1e-12) for i in range(0, testPatchsCen.shape[0])])

        testPatchs = np.array([cv2.resize(testPatchs[i], (32,32), interpolation = cv2.INTER_CUBIC) for i in range(0, testPatchs.shape[0])])
        testPatchs = np.expand_dims(testPatchs, axis=-1)

    testPatchs = testPatchs - pix_mean
    testPatchs = np.array([(testPatchs[i] - np.mean(testPatchs[i]))/(np.std(testPatchs[i]) + 1e-12) for i in range(0, testPatchs.shape[0])])

    res = np.reshape(model.predict(testPatchs), (testPatchs.shape[0], 128))

    if flagCS:
        
        resCen = np.reshape(model_cen.predict(testPatchsCen), (testPatchs.shape[0], 128))

        return np.concatenate((res, resCen), 1)

    else:

        return res 


class L2Net:

    def __init__(self, net_name, do_tf_logging=True, flagCS = False):
        set_tf_logging(do_tf_logging)
                
        # from https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        #config.log_device_placement = True     # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras
        
        model, model_cen, pix_mean, pix_mean_cen = build_L2_net(net_name)
        self.flagCS = flagCS
        self.model = model
        self.model_cen = model_cen
        self.pix_mean = pix_mean
        self.pix_mean_cen = pix_mean_cen

    def calc_descriptors(self, patches):
        if self.flagCS:

            patchesCen = patches[:,16:48,16:48,:]
            patchesCen = patchesCen - self.pix_mean_cen
            patchesCen = np.array([(patchesCen[i] - np.mean(patchesCen[i]))/(np.std(patchesCen[i]) + 1e-12) for i in range(0, patchesCen.shape[0])])

            patches = np.array([cv2.resize(patches[i], (32,32), interpolation = cv2.INTER_CUBIC) for i in range(0, patches.shape[0])])
            patches = np.expand_dims(patches, axis=-1)

        patches = patches - self.pix_mean
        patches = np.array([(patches[i] - np.mean(patches[i]))/(np.std(patches[i]) + 1e-12) for i in range(0, patches.shape[0])])

        res = np.reshape(self.model.predict(patches), (patches.shape[0], 128))

        if self.flagCS:
            
            resCen = np.reshape(self.model_cen.predict(patchesCen), (patches.shape[0], 128))

            return np.concatenate((res, resCen), 1)

        else:

            return res 


# data = np.full((1,64,64,1), 0.)

# result = cal_L2Net_des("L2Net-HP+", data, flagCS=True)

# print(result)