import tensorflow as tf

def layer_norm(input_tensor, epsilon=1e-5, name=None):
    with tf.variable_scope(name, default_name="layer_norm"):
        scale = tf.get_variable("scale", initializer=tf.ones(input_tensor.get_shape()[-1]))
        offset = tf.get_variable("offset", initializer=tf.zeros(input_tensor.get_shape()[-1]))
        mean, variance = tf.nn.moments(input_tensor, axes=[-1], keep_dims=True)
        normalized_input = (input_tensor - mean) / tf.sqrt(variance + epsilon)
        return scale * normalized_input + offset
    
def spectral_norm(w, iteration=1):
    w_shape = tf.shape(w)
    original_w_shape = w_shape
    w = tf.reshape(w, [-1, w_shape[-1]])

    w_shape = w.get_shape().as_list()
    w_shape_int = [int(x) for x in w_shape]
    u = tf.get_variable("u", [1, w_shape_int[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for _ in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, original_w_shape)

    return w_norm

def weight_norm(weights, axis=None):
    eps = 1e-12
    weight_norm = tf.sqrt(tf.reduce_sum(tf.square(weights), axis=axis, keepdims=True))
    return weights / (weight_norm + eps)

def instance_norm(input_tensor, epsilon=1e-5, name=None):
    with tf.variable_scope(name, default_name="instance_norm"):
        scale = tf.get_variable("scale", initializer=tf.ones(input_tensor.get_shape()[-1]))
        offset = tf.get_variable("offset", initializer=tf.zeros(input_tensor.get_shape()[-1]))
        mean, variance = tf.nn.moments(input_tensor, axes=[1, 2], keep_dims=True)
        normalized_input = (input_tensor - mean) / tf.sqrt(variance + epsilon)
        return scale * normalized_input + offset

def batch_norm(x, name, is_training = False):
    normalized = tf.layers.batch_normalization(x, training = is_training,  name=name)
    return normalized

def none_norm(input_tensor, name=None, is_training = False):
    return input_tensor

def get_scope_variable(scope_name, var, shape=None, initialvals=None):
	with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
		v = tf.get_variable(var,shape,dtype=tf.float32, initializer=initialvals)
	return v
