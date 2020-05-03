# L2-Net-Python-Keras

from [https://github.com/virtualgraham/L2-Net-Python-Keras, commit: 5eb9eebe3c5e73f8eb287a43d4fdb9b3425ae0d8]

L2-Net patch descriptor ported to Python using Keras

This is a Keras implementation of the L2-Net patch descriptor described in 

Y. Tian, B. Fan, F. Wu. "L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space", CVPR, 2017.

with weights from 

https://github.com/yuruntian/L2-Net

Requires keras, numpy, pickle and cv2

To get started use the cal_L2Net_des function in L2_Net.py which roughly corresponds to the similarly named function in https://github.com/yuruntian/L2-Net/matlab/cal_L2Net_des.m

Currently, training parameters have not been set, so the model is for inference only.