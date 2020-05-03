from keras import backend as K
from keras.layers.core import Layer

# from https://github.com/ckoren1975/Machine-learning/blob/master/googlenet_custom_layers.py
# except channels have been moved from the 2nd position to the 4th postion
# and shape of input vector is now a tensor operation
# and default args are set to L2-net params

class LRN(Layer):

    def __init__(self, alpha=256,k=0,beta=0.5,n=256, **kwargs):
        super(LRN, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def call(self, x, mask=None):

            s = K.shape(x)
            b = s[0]
            r = s[1]
            c = s[2]
            ch = s[3]

            half_n = self.n // 2 # half the local region

            input_sqr = K.square(x) # square the input

            extra_channels = K.zeros((b, r, c, ch + 2 * half_n))
            input_sqr = K.concatenate([extra_channels[:, :, :, :half_n],input_sqr, extra_channels[:, :, :, half_n + ch:]], axis = 3)

            scale = self.k # offset for the scale
            norm_alpha = self.alpha / self.n # normalized alpha
            for i in range(self.n):
                scale += norm_alpha * input_sqr[:, :, :, i:i+ch]
            scale = scale ** self.beta
            x = x / scale
            
            return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

