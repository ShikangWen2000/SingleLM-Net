import os
import cv2
import numpy as np 
import argparse
import tensorflow as tf
from SingleLuminance_network import SingleLuminance_network
from Data.Metrics import *
from Data.Data_load_weight import load_dataset
from Mask.Circle_Mask import apply_circle_mask
from utils import Test_auto_mkdir, log10, save_results, _metric, get_saved_model_paths
curr_path = os.getcwd()

parser = argparse.ArgumentParser()
# Test settings
parser.add_argument('--batch_size', type=int, default = 1, help='gen and disc batch size')
parser.add_argument('--resize', type=int, default=512)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for GPU')
parser.add_argument('--opt', help='optimizer for train, default [adam]',
                        dest='OPT', default='adam', choices=['adam', 'sgd_m', 'sgd', 'RMSP'])
parser.add_argument('--gan', dest='gan', default='wgan_gp', choices=['sphere', 'wgan_gp', 'pgan', 'gan'])
parser.add_argument('--dis_norm_type', help='normalization', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn'])
parser.add_argument('--gen_norm_type', help='normalization', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn'])
parser.add_argument('--num_layers', default=3, choices=(2, 3, 4, 5), type=int)
parser.add_argument('--act', help='activation', default='leak_relu', choices=['swish','leak_relu', 'relu'])
# Data settings and save path
parser.add_argument("--save_hdr", type=str, default = "False", help="Save hdr False or True")
parser.add_argument('--Validation_path', type=str, default = 'validation', help='path to test data')
parser.add_argument('--logdir_path', type=str, default='LDR_Test_output', help='path of logs')
parser.add_argument('--model_name', type=str, default='model_name', help='folder name to save weights')
parser.add_argument('--Checkpoint_path', type=str, default='', help='path to pretrained model or ckptpoints')
# Mask settings
parser.add_argument('--mask', type=str, default = "False", help = 'use the mask')
parser.add_argument("--input_mask", type=str, default="False", help="False or True")
parser.add_argument("--output_mask", type=str, default="False", help="False or True")
parser.add_argument("--final_output_mask", type=str, default="False", help="False or True")
args = parser.parse_args()

def build_graph(
        ldr,  # [b, h, w, c]
        hdr,  # [b, h, w, c]
        is_training,
):
    model = SingleLuminance_network(ldr, hdr, args.batch_size, is_training, args)
    # Loss
    D_loss, G_loss, generator_image = model.compute_loss()
    hdr = apply_circle_mask(hdr)
    if args.final_output_mask == "True":
        generator_image = apply_circle_mask(generator_image)
    mse = tf.reduce_mean(tf.square(generator_image  - hdr))
    psnr = 20.0 * log10(1.0) - 10.0 * log10(mse)
    return generator_image, psnr

def test(args):
    test_input_images, test_reference_images, test_total_images, test_iterator, input_filenames = load_dataset(args, 'Validation')
    batch_size = args.batch_size
    input_size = args.resize
    max_val = 255.0
    ######### Prep for training
    # Path for tf.summary.FileWriter and to store model checkpoints
    output_path, filewriter_path, Image_path, json_path = Test_auto_mkdir(args, curr_path)
    """evaluation metrics output"""
    index_position = 0
    sheet, workbook2 = _metric()
    # TF placeholder for graph input
    image_A = tf.placeholder(tf.float32, [None, input_size, input_size, 3])
    image_B = tf.placeholder(tf.float32, [None, input_size, input_size, 3])
    is_training = tf.placeholder(tf.bool)
    # Config
    config = tf.ConfigProto()
    # Specify the GPU device you want to use. Use device number, e.g., "0" for the first GPU.
    config.gpu_options.visible_device_list = args.gpu_ids
    # Create a TensorFlow session with the modified configuration.
    generator_image, psnr= build_graph(image_A, image_B, is_training)
    
    saver = tf.train.Saver(tf.all_variables(), max_to_keep = 100)
    ######### Start training
    with tf.Session(config=config) as sess: 
        with tf.device('/gpu:'+str(args.gpu_ids)):
            # Initialize all variables and start queue runners
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            # To continue training from one of the checkpoints
            if args.Checkpoint_path.endswith('.ckpt'):
                model_checkpoints = [args.Checkpoint_path]
            else:
                model_checkpoints = get_saved_model_paths(args.Checkpoint_path)
            
            best_score = float('-inf')
            # 初始化一个列表用于保存所有参数和分数
            results = []
            results_filename_path = json_path
            #start test
            print("start test...")
            for model_checkpoint in model_checkpoints:
                print(model_checkpoint)
                saver.restore(sess, model_checkpoint)
                # Loop over number of epochs
                step = 0
                sess.run(test_iterator.initializer)
                # calculate the performance on test set
                u_psnr = 0
                l_psnr = 0
                ssim_avg = 0
                mse = 0
                num_batches = test_total_images // batch_size
                for test_iter in range(num_batches):
                    
                    test_batch_A, test_batch_B, filename = sess.run([test_input_images, test_reference_images, input_filenames])
                    #test_batch_A, test_batch_B= sess.run([test_input_images, test_reference_images])
                    test_psnr_val, generator_image_val = sess.run([psnr, generator_image], feed_dict={image_A: test_batch_A.astype('float32'), image_B: test_batch_B.astype('float32'), is_training: False})
                    u_psnr += test_psnr_val
                    index_position += 1
                    sheet.write(index_position, 0, 'Gan' + str(step))
                    fake_B = np.array(generator_image_val) * 255.0
                    print("{} / {} :Normlization PSNR: {}".format(test_iter, num_batches, test_psnr_val))
                    if args.save_hdr == "True":
                        print("Normlization PSNR: {}".format(test_psnr_val))
                        for num_img in range(args.batch_size):                            
                            img1 = fake_B[num_img]
                            #save image with epoch
                            filename = filename[num_img].decode('utf-8')
                            filename1 = filename.split('/')[-1].split('.')[0]
                            filename2 = filename1 + '.jpg'
                            save_path = os.path.join(Image_path, filename2)
                            image_rgb = np.flip(img1, axis=-1) 
                            cv2.imwrite(save_path, image_rgb)
                            cur_psnr, cur_ssim, cur_mse = calculate_evaluation(fake_B[num_img], test_batch_B[num_img] * max_val, max_val)
                            l_psnr += cur_psnr
                            ssim_avg += cur_ssim
                            mse += cur_mse
                    else:
                        for num_img in range(args.batch_size):
                            cur_psnr, cur_ssim, cur_mse = calculate_evaluation(fake_B[num_img], test_batch_B[num_img] * max_val, max_val)
                            l_psnr += cur_psnr
                            ssim_avg += cur_ssim
                            mse += cur_mse
                u_psnr /= num_batches
                l_psnr /= num_batches
                ssim_avg /= num_batches
                mse /= num_batches
                print("Average normlization PSNR on test set: {}".format(u_psnr))
                print("Average PSNR on test set: {}".format(l_psnr))
                print("Average SSIM on test set: {}".format(ssim_avg))
                print("Average MSE on test set: {}".format(mse))
                results.append({
                'model_name': str(model_checkpoint),
                'score': l_psnr
                    })
                save_results(results, results_filename_path)
                if l_psnr > best_score:
                    best_score = l_psnr
                    best_params = str(model_checkpoint)
            print("Best score:", best_score)
            print("Best model:", best_params)
            return best_score

def main(args):
    psnr_score = test(args)
    print("The best PSNR score is: ", psnr_score)

if __name__ == '__main__':
    main(args)
