import tensorflow as tf
from Model._conv import *
from Model.Normalization import *

class PatchDiscriminator():
    """
    70x70 PatchGAN with spectral_norm
    """
      
    def __init__(self, dis_norm_type, cnum=(64, 128, 128, 128), name=None):
        assert cnum is None or len(cnum) == 4
        self.cnum = (64, 128, 256, 512)
        self.act = tf.nn.leaky_relu
        # TODO: diff
        #self.norm = layer_norm
        self._init_normalization(dis_norm_type)

    def _init_normalization(self, norm):
        assert norm in ['in', 'ln', 'nn', 'sn', 'wn'], 'invalid norm: {}'.format(norm)
        self.conv2d = conv2d
        self.transpose_conv2d = transpose_conv2d
        self.norm = none_norm
        if norm == 'in':
            self.norm = instance_norm
        elif norm == 'ln':
            self.norm = layer_norm
        elif norm == 'sn':
            self.conv2d = sn_conv2d
        elif norm == 'wn':
            self.conv2d = wn_conv2d
    
    def graph(self, x, reuse=None, training=True):
        with tf.variable_scope("dis_", reuse=reuse):
            cnum = self.cnum
            act = self.act
            conv2d = self.conv2d
            norm = self.norm

            x = conv2d(x, cnum[0], (4, 4), (2, 2), padding='valid', name='dis_cov1')
            x = act(x)

            x = conv2d(x, cnum[1], (4, 4), (2, 2), padding='valid', name='dis_cov2')
            x = norm(x, name='2')
            x = act(x)

            x = conv2d(x, cnum[2], (4, 4), (2, 2), padding='valid', name='dis_cov3')
            x = norm(x, name='3')
            x = act(x)

            x = conv2d(x, cnum[3], (4, 4), (1, 1), padding='valid', name='dis_cov4')
            x = norm(x, name='4')
            x = act(x)

            x = conv2d(x, 1, (4, 4), (1, 1), padding='valid', name='dis_cov5')
        return x

