import scipy.io as spio
import pickle
import numpy as np


def loadmat(filename):

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):

    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def extract_net_weights(data, net_name, cen):

    model = []

    for i in data['netCen' if cen else 'net']['layers']:

        if i.type == 'conv':

            weights = i.weights[0]

            # the weights for the first layer are missing a channel dimension
            # add the dimesion here
            if weights.ndim == 3:
                weights = np.expand_dims(weights, axis=2)

            bias = i.weights[1]
            model.append({'weights': weights, 'bias': bias})

        elif i.type == 'bnormPair':

            # gamma and beta are ignored in L2-Net
            # gamma = i.weights[0] 
            # beta = i.weights[1]
            mean = [row[0] for row in i.weights[2]]

            # convert stadard deviation to variance
            var = [row[1]**2 for row in i.weights[2]]
            
            model.append({'mean': mean, 'var': var})

        elif i.type == 'normalize':
            pass

        elif i.type == 'relu':
            pass 

        else:
            print('Encountered layer information unseen before, please update to account for: ' + i.type)


    return [
        model[0]['weights'],
        model[0]['bias'],
        model[1]['mean'],
        model[1]['var'],
        
        model[2]['weights'],
        model[2]['bias'],
        model[3]['mean'],
        model[3]['var'],

        model[4]['weights'],
        model[4]['bias'],
        model[5]['mean'],
        model[5]['var'],

        model[6]['weights'],
        model[6]['bias'],
        model[7]['mean'],
        model[7]['var'],

        model[8]['weights'],
        model[8]['bias'],
        model[9]['mean'],
        model[9]['var'],

        model[10]['weights'],
        model[10]['bias'],
        model[11]['mean'],
        model[11]['var'],

        model[12]['weights'],
        model[12]['bias'],
        model[13]['mean'],
        model[13]['var'],
    ]

def extract_net(net_name):

    matlab_data = loadmat('matlab_net_data/' + net_name + '.mat')

    weights = extract_net_weights(matlab_data, net_name, False)
    weights_cen = extract_net_weights(matlab_data, net_name, True)

    # add channel dimension for easy 
    pix_mean = np.expand_dims(matlab_data['pixMean'], axis=-1)
    pix_mean_cen = np.expand_dims(matlab_data['pixMeanCen'], axis=-1)

    python_data = {"weights": weights, "weights_cen": weights_cen, "pix_mean": pix_mean, "pix_mean_cen": pix_mean_cen}

    pickle.dump(python_data, open('python_net_data/' + net_name + '.p', "wb" ))

    print('Successfully scraped data from MatConvNet Model', net_name)

extract_net("L2Net-HP+")
extract_net("L2Net-HP")
extract_net("L2Net-LIB+")
extract_net("L2Net-LIB")
extract_net("L2Net-ND+")
extract_net("L2Net-ND")
extract_net("L2Net-YOS+")
extract_net("L2Net-YOS")

print('Done')