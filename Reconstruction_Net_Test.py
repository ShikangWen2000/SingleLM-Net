import argparse
import os
import tensorflow as tf
import numpy as np
import cv2
from model.Generator import GeneratorModel
from refinement_net import Refinement_net
from Data.Data_load_weight_finetune import load_dataset
from Data.Metrics import *
from Mask.Circle_Mask import apply_circle_mask
from utils import Test_auto_mkdir, log10, save_results, _metric, get_saved_model_paths
# ---
parser = argparse.ArgumentParser()
# Test settings
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--resize', type=int, default=512)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--gan', dest='gan', default='wgan_gp', choices=['sphere', 'wgan_gp', 'pgan', 'gan'])
parser.add_argument('--dis_norm_type', help='normalization', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn'])
parser.add_argument('--gen_norm_type', help='normalization', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn', 'bn'])
parser.add_argument('--num_layers', default=3, choices=(2, 3, 4, 5), type=int)
parser.add_argument('--act', help='activation', default='leak_relu', choices=['swish','leak_relu', 'relu'])
parser.add_argument('--two_stage_network', type=str, default='Unet', help='two stage network')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for GPU')
# Data settings and save path
parser.add_argument('--logdir_path', type=str, default = 'Test_output')
parser.add_argument('--model_name', type=str, required=True, default = 'model_name')
parser.add_argument('--Validation_path', type=str, default ='', help='validation dataset path')
parser.add_argument('--Checkpoint_path', type=str, default='', help='path to pretrained ckpt or ckptpoints direction')
parser.add_argument("--save_hdr", type=str, default = "False", help=" save hdr False or True")
# Mask settings
parser.add_argument('--mask', type=str, default = "False", help = 'use the mask')
parser.add_argument("--input_mask", type=str, default = "False", help="False or True")
parser.add_argument("--output_mask", type=str, default = "False", help="False or True")
parser.add_argument("--final_output_mask", type=str, default= "False", help="False or True")

args = parser.parse_args()

# ---
curr_path = os.getcwd()


def build_graph(
        ldr,  # [b, h, w, c]
        hdr,  # [b, h, w, c]
        is_training,
):
    Gen_model = GeneratorModel(args, is_training)
    fake_B = Gen_model.graph(ldr)
    # Refinement-Net
    if args.two_stage_network == 'Unet':
        with tf.variable_scope("Refinement_Net"):
            refinement_model = Refinement_net(is_train=is_training)
            refinement_output = refinement_model.inference(fake_B, ldr)
    hdr = apply_circle_mask(hdr)
    if args.final_output_mask == "True":
        refinement_output = apply_circle_mask(refinement_output)
    mse = tf.reduce_mean(tf.square(refinement_output - hdr))
    psnr = 20.0 * log10(1.0) - 10.0 * log10(mse)
    return refinement_output, psnr, mse

b = args.batch_size
h = args.resize
w = args.resize
c = 3
ldr = tf.placeholder(tf.float32, [None, h, w, c])
hdr = tf.placeholder(tf.float32, [None, h, w, c])
is_training = tf.placeholder(tf.bool)

HDR_out, _psnr, _mse = build_graph(ldr, hdr, is_training)
class Tester:
    def __init__(self):
        return
    def test_it(self):
        # To continue training from one of the checkpoints
        if args.Checkpoint_path.endswith('.ckpt'):
            model_checkpoints = [args.Checkpoint_path]
        else:
            model_checkpoints = get_saved_model_paths(args.Checkpoint_path)
        best_score = float('-inf')
        # initialize the results list
        results = []
        # auto mkdir
        output_path, filewriter_path, Image_path, results_filename_path = Test_auto_mkdir(args, curr_path)

        """evaluation metrics output"""
        metric_name = 'evaluation_metrics.xls'
        output_xls = os.path.join(filewriter_path, metric_name)
        index_position = 0
        sheet, workbook2 = _metric()
        # Output the parameters and scores to a JSON file

        # load test data
        test_input_images, test_reference_images, test_total_images, test_iterator, input_filenames = load_dataset(args, 'Validation')
        #start test
        for model_checkpoint in model_checkpoints:
            print("Testing model: {}".format(model_checkpoint))
            restorer.restore(sess, model_checkpoint)
            # Loop over number of epochs
            sess.run(test_iterator.initializer)
            # PSNR
            u_psnr = 0
            l_psnr = 0
            ssim_avg = 0
            mse = 0
            index_position = 0
            max_val = 32000.0
            num_batches = test_total_images // args.batch_size
            
            for test_iter in range(num_batches):
                test_batch_A, test_batch_B, filename = sess.run([test_input_images, test_reference_images, input_filenames])
                HDR_out_val, psnr_val, mse_val = sess.run([HDR_out, _psnr, _mse], {
                    ldr: test_batch_A,
                    hdr: test_batch_B,
                    is_training: False,
                })
                #print result and save image
                u_psnr += psnr_val
                fake_B = np.array(HDR_out_val) * 32000.0
                
                if args.save_hdr == "True":
                    print("{} / {} :Normlization PSNR: {}".format(test_iter, num_batches, psnr_val))
                    for num_img in range(args.batch_size):
                        img1 = fake_B[num_img]
                        ref = test_batch_B[num_img] * 32000.0
                        #save image with epoch
                        filename = filename[num_img].decode('utf-8')
                        filename1 = filename.split('/')[-1].split('.')[0]
                        filename2 = filename1 + '.hdr'
                        index_position += 1                   
                        sheet.write(index_position, 0, filename2)   
                        _save_path = os.path.join(Image_path, filename2)
                        cv2.imwrite(_save_path, img1)
                        cur_psnr, cur_ssim, cur_mse = calculate_evaluation(img1, ref, max_val)
                        l_psnr += cur_psnr
                        ssim_avg += cur_ssim
                        mse += cur_mse
                        sheet.write(index_position, 1, float(psnr_val))
                        sheet.write(index_position, 2, float(cur_ssim))
                        sheet.write(index_position, 3, float(cur_mse))
                else:
                    for num_img in range(args.batch_size):
                        cur_psnr, cur_ssim, cur_mse = calculate_evaluation(fake_B[num_img], test_batch_B[num_img] * 32000.0, max_val)
                        l_psnr += cur_psnr
                        ssim_avg += cur_ssim
                        mse += cur_mse
                    
            u_psnr /= num_batches
            l_psnr /= num_batches
            ssim_avg /= num_batches
            mse /= num_batches
            # print the result
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
            workbook2.save(output_xls)
        
        print("Best score:", best_score)
        print("Best model:", best_params)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
# Specify the GPU device you want to use. Use device number, e.g., "0" for the first GPU.
config.gpu_options.visible_device_list = args.gpu_ids
sess = tf.Session(config=config)
restorer = tf.train.Saver()
tester = Tester()
tester.test_it()
