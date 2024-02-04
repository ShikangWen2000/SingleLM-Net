import tensorflow as tf
from model.Normalization import *


def conv2d(input_tensor, num_outputs, kernel_size, strides, padding, name, use_bias=True):
    with tf.variable_scope(name):
        w = tf.get_variable("kernel", shape=[kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], num_outputs])
        conv = tf.nn.conv2d(input_tensor, w, strides=[1, strides[0], strides[1], 1], padding=padding.upper())
        if use_bias:
            biases = tf.get_variable("bias", [num_outputs], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv

def transpose_conv2d(input_tensor, num_outputs, kernel_size, strides, padding, name, use_bias=True):
    with tf.variable_scope(name):
        w_shape = [kernel_size[0], kernel_size[1], num_outputs, input_tensor.get_shape()[-1]]
        w = tf.get_variable("kernel", shape=w_shape)
        output_shape = tf.stack([tf.shape(input_tensor)[0], tf.shape(input_tensor)[1] * strides[0], tf.shape(input_tensor)[2] * strides[1], num_outputs])
        conv = tf.nn.conv2d_transpose(input_tensor, w, output_shape=output_shape, strides=[1, strides[0], strides[1], 1], padding=padding.upper())
        if use_bias:
            biases = tf.get_variable("bias", [num_outputs], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv

def sn_conv2d(input_tensor, num_outputs, kernel_size, strides, padding, name, use_bias=True):
    with tf.variable_scope(name):
        w_shape = [kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], num_outputs]
        #print("w1 shape:", input_tensor.get_shape())
        w = tf.get_variable("kernel", shape=w_shape)
        #print("w2 shape:", input_tensor.get_shape())
        w = spectral_norm(w)
        #print("Input tensor shape:", input_tensor.get_shape())
        #print("Weight tensor shape:", w.get_shape())
        conv = tf.nn.conv2d(input_tensor, w, strides=[1, strides[0], strides[1], 1], padding=padding.upper())
        if use_bias:
            biases = tf.get_variable("bias", [num_outputs], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv

def sn_transpose_conv2d(input_tensor, num_outputs, kernel_size, strides, padding, name, use_bias=True):
    with tf.variable_scope(name):
        w_shape = [kernel_size[0], kernel_size[1], num_outputs, input_tensor.get_shape()[-1]]
        w = tf.get_variable("kernel", shape=w_shape)
        w = spectral_norm(w)
        output_shape = tf.stack([tf.shape(input_tensor)[0], tf.shape(input_tensor)[1] * strides[0], tf.shape(input_tensor)[2] * strides[1], num_outputs])
        conv = tf.nn.conv2d_transpose(input_tensor, w, output_shape=output_shape, strides=[1, strides[0], strides[1], 1], padding=padding.upper())
        if use_bias:
            biases = tf.get_variable("bias", [num_outputs], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv

    

def wn_conv2d(input_tensor, num_outputs, kernel_size, strides, padding, name, use_bias=True, transpose=False):
    with tf.variable_scope(name):
        w_shape = [kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], num_outputs]
        w = tf.get_variable("kernel", shape=w_shape)
        w = weight_norm(w, axis=[0, 1, 2])
        conv = tf.nn.conv2d(input_tensor, w, strides=[1, strides[0], strides[1], 1], padding=padding.upper())
        if use_bias:
            biases = tf.get_variable("bias", [num_outputs], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv

def wn_transpose_conv2d(input_tensor, num_outputs, kernel_size, strides, padding, name, use_bias=True, transpose=False):
    with tf.variable_scope(name):
        w_shape = [kernel_size[0], kernel_size[1], num_outputs, input_tensor.get_shape()[-1]]
        w = tf.get_variable("kernel", shape=w_shape)
        w = weight_norm(w, axis=[0, 1, 2])
        output_shape = tf.stack([tf.shape(input_tensor)[0], tf.shape(input_tensor)[1] * strides[0], tf.shape(input_tensor)[2] * strides[1], num_outputs])
        conv = tf.nn.conv2d_transpose(input_tensor, w, output_shape=output_shape, strides=[1, strides[0], strides[1], 1], padding=padding.upper())
        if use_bias:
            biases = tf.get_variable("bias", [num_outputs], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv