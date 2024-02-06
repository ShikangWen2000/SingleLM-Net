import cv2
import numpy as np 
import tensorflow as tf
import glob
import os


def _parse_function(input_filename, reference_filename, _image_size):
    image_size = (_image_size, _image_size)
    input_image = tf.read_file(input_filename)
    input_image = tf.image.decode_jpeg(input_image, channels=3)
    input_image = tf.image.resize_images(input_image, image_size)
    input_image = tf.cast(input_image, tf.float32) / 255.0
    #reference processing
    reference_image = tf.read_file(reference_filename)
    reference_image = tf.image.decode_jpeg(reference_image, channels=3)
    reference_image = tf.image.resize_images(reference_image, image_size)
    reference_image = tf.cast(reference_image, tf.float32) / 255.0
    input_filename = tf.string_split([input_filename], delimiter='\\').values[-1]
    return input_filename, input_image, reference_image

def load_dataset(args, mode_type = "train"):
    if mode_type == "train":
        dataset_path = args.dataroot
        batch_size = args.batch_size
    elif mode_type == 'Validation':
        dataset_path = args.Validation_path
        batch_size = 1
    input_path = os.path.join(dataset_path, "Input", "*.JPG")
    reference_path = os.path.join(dataset_path, "Reference", "*.JPG")

    input_filenames = glob.glob(input_path)
    reference_filenames = glob.glob(reference_path)
    input_filenames = [path.replace('\\\\', '\\') for path in input_filenames]
    reference_filenames = [path.replace('\\\\', '\\') for path in reference_filenames]

    assert len(input_filenames) == len(reference_filenames), "Mismatch in the number of input and reference files."

    dataset = tf.data.Dataset.from_tensor_slices((input_filenames, reference_filenames))
    if mode_type == "train":
        dataset = dataset.shuffle(buffer_size=len(input_filenames))
    elif mode_type == "Validation":
        dataset = dataset.repeat(1)
    dataset = dataset.map(lambda input_filename, reference_filename: _parse_function(input_filename, reference_filename, args.resize), num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    input_filename_save, input_images, reference_images = iterator.get_next()

    if mode_type == "train":
        return input_images, reference_images, len(input_filenames), iterator
    elif mode_type == "Validation":
        return input_images, reference_images, len(input_filenames), iterator, input_filename_save