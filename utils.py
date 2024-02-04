import tensorflow as tf
import os
import json
import re
import xlwt
import time
from Data.Metrics import *

def auto_mkdir(curr_path, args):
    results_filename = 'params_and_scores.json'
    pre_path = os.path.join(curr_path, args.logdir_path)
    output_path = os.path.join(pre_path, args.model_name)    
    if not os.path.isdir(pre_path): os.mkdir(pre_path)
    if os.path.isdir(output_path):
        #Model_name + random sign
        output_path = output_path + "_" + str(int(time.time()))
        os.mkdir(output_path)
    else:
        os.mkdir(output_path)
    filewriter_path = output_path + "/TBoard_files"
    checkpoint_path = output_path + "/checkpoints"
    Image_path = output_path + "/Images"
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
    if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
    if not os.path.isdir(Image_path): os.mkdir(Image_path)
    json_path = os.path.join(output_path, results_filename)

    return output_path, filewriter_path, checkpoint_path, Image_path, json_path

def Test_auto_mkdir(args, curr_path):
    results_filename = 'params_and_scores.json'
    pre_path = os.path.join(curr_path, args.logdir_path)
    output_path = os.path.join(pre_path, args.model_name)    
    if not os.path.isdir(pre_path): os.mkdir(pre_path)
    if os.path.isdir(output_path):
        #Model_name + random sign
        output_path = output_path + "_" + str(int(time.time()))
        os.mkdir(output_path)
    else:
        os.mkdir(output_path)
    filewriter_path = output_path + "/Metric"
    Image_path = output_path + "/Images"
    json_path = os.path.join(output_path, results_filename)
    if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
    if not os.path.isdir(Image_path): os.mkdir(Image_path)

    return output_path, filewriter_path, Image_path, json_path
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def save_results(results, filename):
    # save the results to a file
    with open(filename, 'w') as f:
        json.dump(results, f)
        
def select_optimizer(opt_name, learning_rate):
    if opt_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif opt_name == 'sgd_m':
        return tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif opt_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif opt_name == 'RMSP':
        return tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise ValueError('Invalid optimizer choice')

def restore_training(saver, sess, args):
    start_epoch = 0
    if args.restore == "True":
        ckpt = tf.train.get_checkpoint_state(args.restore_path)  
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)#restore all variables
            start_epoch = re.findall(r"\d+\.?\d*", str(ckpt))   #obtain the number of the last epoch
            #start_epoch = re.findall(r"(?<=-)\d+",ckpt)
            print(start_epoch)
            print('the epoch finished last time is ' + start_epoch[-1])  #print the number of the last epoch
            start_epoch = int(float(start_epoch[-1]))                #keep the number of the last epoch
            print('Model restored...')
            start_epoch = start_epoch + 1
        else:
            start_epoch = 0
            print('No model')
        #saver.restore(sess, args.checkpoint_name)
    return start_epoch

def select_optimizer(opt_name, learning_rate):
    if opt_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif opt_name == 'sgd_m':
        return tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif opt_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif opt_name == 'RMSP':
        return tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise ValueError('Invalid optimizer choice')

def _metric():
    workbook2 = xlwt.Workbook(encoding='utf-8')
    sheet = workbook2.add_sheet('Evaluation_metric')
    sheet.write(0, 0, label = 'Name')
    sheet.write(0, 1, label = 'PSNR')
    sheet.write(0, 2, label = 'SSIM')
    sheet.write(0, 3, label = 'MSE')

    return sheet, workbook2

def get_saved_model_paths(args, checkpoint_file):
    saved_model_paths = []
    #if win
    if args.system == 'win':
        with open(checkpoint_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("model_checkpoint_path"):
                    continue
                model_path = line.split(": ")[-1].strip().split('"')[1]
                
                # Replace forward slashes with double backslashes
                model_path = model_path.replace('/', '\\\\')
                model_path = model_path.replace('\\\\', '\\')
                saved_model_paths.append(model_path)
    
    if args.system == 'linux':
        with open(checkpoint_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("model_checkpoint_path"):
                    continue
                elif line.startswith("all_model_checkpoint_paths"):
                    model_path = line.split(": ")[-1].strip().split('"')[1] + ".ckpt"
                    saved_model_paths.append(model_path)

    return saved_model_paths

def save_training_image(fake_B, batch_B, max_val, sheet, index_position, workbook2, output_xls, iter_id, batch_size):
    _psnr, _ssim, _mse, flag = 0 ,0, 0, 0
    range_num = 2 if batch_size > 2 else batch_size
    for num_img in range(range_num):
        cur_psnr, cur_ssim, cur_mse = calculate_evaluation(fake_B[num_img], batch_B[num_img], max_val)
        _psnr += cur_psnr
        _ssim += cur_ssim
        _mse += cur_mse
    flag += range_num
    #print('psnr is : {}, ssim is : {}, mse is : {}'.format(str(_psnr/flag), str(_ssim/flag),str(_mse/flag)))
    sheet.write(index_position, 1, float(_psnr/flag))
    sheet.write(index_position, 2, float(_ssim/flag))
    sheet.write(index_position, 3, float(_mse/flag))
    workbook2.save(output_xls)
    print("iteration {} for GAN training's psnr {}, _ssim {}, _mse {}".format(str(iter_id),str(_psnr), str(_ssim), str(_mse)))
    return sheet, workbook2