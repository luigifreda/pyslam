import numpy as np
import scipy.io as sio
import torch
import torch.nn.init
from misc.l2net.l2net_model import L2Net

eps = 1e-10

def check_ported(l2net_model, test_patch, img_mean):

    test_patch = test_patch.transpose(3, 2, 0, 1)-img_mean
    desc = l2net_model(torch.from_numpy(test_patch))
    print(desc)
    return desc

if __name__ == '__main__':

    path_to_l2net_weights = '/cvlabsrc1/cvlab/datasets_anastasiia/descriptors/sfm-evaluation-benchmarking/third_party/l2net/matlab/L2Net-LIB+.mat'

    l2net_weights = sio.loadmat(path_to_l2net_weights)

    l2net_model = L2Net()
    l2net_model.eval()

    new_state_dict = l2net_model.state_dict().copy()

    conv_layers, bn_layers = {}, {}
    all_layer_weights = l2net_weights['net']['layers'][0][0][0]
    img_mean = l2net_weights['pixMean']
    conv_layers_to_track, bn_layers_to_track = [0,3,6,9,12,15,18], \
                                               [1,4,7,10,13,16,19]
    conv_i, bn_i = 0,0

    for layer in all_layer_weights:

        if 'weights' not in layer.dtype.names:
            continue
        layer_name = layer[0][0][0][0]
        layer_value = layer['weights'][0][0][0]
        if layer_name == 'conv':
            conv_layers[conv_layers_to_track[conv_i]] = layer_value
            conv_i+=1
        elif layer_name == 'bnormPair':
            bn_layers[bn_layers_to_track[bn_i]] = layer_value
            bn_i+=1

    for key, value in new_state_dict.items():
        layer_number = int(key.split('.')[1])
        if layer_number in conv_layers.keys():
            if 'weight' in key:
                new_state_dict[key] = torch.from_numpy(conv_layers[layer_number][0].transpose((3,2,0,1)))
            elif 'bias' in key:
                new_state_dict[key] = torch.from_numpy(conv_layers[layer_number][1]).squeeze()
        elif layer_number in bn_layers.keys():
            if 'running_mean' in key:
                new_state_dict[key] = torch.from_numpy(np.array([x[0] for x in bn_layers[layer_number][2]])).squeeze()
            elif 'running_var' in key:
                new_state_dict[key] = torch.from_numpy(np.array([x[1] for x in bn_layers[layer_number][2]] )** 2 -eps).squeeze()
            elif 'weight' in key:
                new_state_dict[key] = torch.from_numpy(np.ones(value.size()[0])).squeeze()

        else:
            continue

    l2net_model.load_state_dict(new_state_dict)
    l2net_model.eval()

    torch.save(l2net_model.state_dict(),'l2net_ported_weights_lib+.pth')

    # compare desc on test patch with matlab implementation
    # test_patch_batch = sio.loadmat('test_batch_img.mat')['testPatch']
    # check_ported(l2net_model, test_patch_batch, img_mean)
    #
    # test_patch_one = sio.loadmat('test_one.mat')['testPatch']
    # check_ported(l2net_model, np.expand_dims(np.expand_dims(test_patch_one, axis=2),axis=2), img_mean)