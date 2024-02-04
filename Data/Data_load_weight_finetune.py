import cv2
import numpy as np 
import tensorflow as tf
import glob
import os

def _read_hdr_image(hdr_path, _image_size_=512, mode = "train"):
    image_size = (int(_image_size_), int(_image_size_))
    hdr_path = hdr_path.decode("utf-8")
    #print("Reading HDR image from path:", hdr_path)
    hdr_image = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
    hdr_image = cv2.resize(hdr_image, image_size)
    #BGR2RGB
    hdr_image = hdr_image[:, :, [2, 1, 0]]
    hdr_image = np.float32(hdr_image) / 32000.0
    return hdr_image

def _read_ldr_image(ldr_path, _image_size_=512, mode = "train"):
    image_size = (int(_image_size_), int(_image_size_))
    ldr_path = ldr_path.decode("utf-8")
    #print("Reading HDR image from path:", hdr_path)
    ldr_image = cv2.imread(ldr_path, cv2.IMREAD_UNCHANGED)
    ldr_image = cv2.resize(ldr_image, image_size)
    #BGR2RGB
    ldr_image = ldr_image[:, :, [2, 1, 0]]

    ldr_image = np.float32(ldr_image) / 255.0
    return ldr_image
def _parse_function(input_filename, reference_filename, _image_size, mode):
    input_image = tf.py_func(_read_ldr_image, [input_filename, _image_size, mode], tf.float32)
    reference_image = tf.py_func(_read_hdr_image, [reference_filename, _image_size, mode], tf.float32)
    #obtain the prefix of input_filename
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
    reference_path = os.path.join(dataset_path, "Reference", "*.hdr")
    #if mode_type == "train" and args.argumentation == "True":
        #input_path, reference_path = argumentation_list(dataset_path, input_path, reference_path)

    input_filenames = glob.glob(input_path)
    reference_filenames = glob.glob(reference_path)
    input_filenames = [path.replace('\\\\', '\\') for path in input_filenames]
    reference_filenames = [path.replace('\\\\', '\\') for path in reference_filenames]
    print(len(input_filenames))
    print(len(reference_filenames))
    assert len(input_filenames) == len(reference_filenames), "Mismatch in the number of input and reference files."

    dataset = tf.data.Dataset.from_tensor_slices((input_filenames, reference_filenames))
    if mode_type == "train":
        dataset = dataset.shuffle(buffer_size=len(input_filenames))
    elif mode_type == "Validation":
        dataset = dataset.repeat(1)
    dataset = dataset.map(lambda input_filename, reference_filename: _parse_function(input_filename, reference_filename, args.resize, mode_type), num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)

    iterator = dataset.make_initializable_iterator()
    input_filename_save, input_images, reference_images = iterator.get_next()
    if mode_type == "train":
        return input_images, reference_images, len(input_filenames), iterator
    elif mode_type == "Validation":
        return input_images, reference_images, len(input_filenames), iterator, input_filename_save
