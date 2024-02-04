import tensorflow as tf
from model._conv import *
layer_number = 8
class Refinement_net(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def inference(self, input_images, ldr):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(input_images, ldr)

    def down(self, x, outChannels, filterSize, block_name):
        x = tf.layers.average_pooling2d(x, 2, 2)
        for _ in range(layer_number):
            x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        return x

    def up(self, x, outChannels, skpCn, block_name):
        x = tf.image.resize_bilinear(x, 2*tf.shape(x)[1:3])
        for _ in range(layer_number):
            x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, 3, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(tf.concat([x, skpCn], -1), outChannels, 3, 1, 'same'), 0.1)
        return x
    
    def _build_model(self, input_images, ldr):
        x = tf.nn.leaky_relu(tf.layers.conv2d(input_images, 16, 7, 1, 'same'), 0.1)
        s1 = tf.nn.leaky_relu(tf.layers.conv2d(x, 16, 7, 1, 'same'), 0.1)
        s2 = self.down(s1, 32, 5, 's2')
        s3 = self.down(s2, 64, 3, 's3')
        s4 = self.down(s3, 128, 3, 's4')
        x = self.down(s4, 128, 3, 's5')
        x = self.up(x, 128, s4, 's6')
        x = self.up(x, 64, s3, 's7')
        x = self.up(x, 32, s2, 's8')
        x = self.up(x, 16, s1, 's9')
        x = tf.layers.conv2d(x, 3, 3, 1, 'same')
        output = input_images + x
        output = tf.nn.tanh(output)
        return output