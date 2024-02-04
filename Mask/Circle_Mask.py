import tensorflow as tf
import numpy as np


def apply_circle_mask(input_tensor):
    # Get the input tensor shape
    tensor_shape = input_tensor.shape
    if len(tensor_shape) < 4:
        raise ValueError("Input tensor should have at least 4 dimensions (batch, height, width, channels)")

    # Get the dynamic tensor shape
    dynamic_shape = tf.shape(input_tensor)

    # Calculate the mask size and center
    mask_size = dynamic_shape[1]
    center = mask_size // 2
    circle_radius = center

    # Create a coordinate grid for the mask
    Y, X = tf.meshgrid(tf.range(mask_size), tf.range(mask_size), indexing='ij')
    mask = (X - center) ** 2 + (Y - center) ** 2 <= circle_radius ** 2
    mask = tf.cast(mask, tf.float32)

    # Expand the mask tensor to match the input tensor shape
    mask_tensor = tf.expand_dims(tf.expand_dims(mask, 0), -1)

    # Apply the mask to the input tensor
    masked_tensor = tf.multiply(input_tensor, mask_tensor)

    return masked_tensor

def none_mask(input_tensor):
    return input_tensor