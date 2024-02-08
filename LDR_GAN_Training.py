import os
import cv2
import numpy as np
import glob
import tensorflow as tf
from SingleLuminance_network import SingleLuminance_network
import argparse
from Data.Data_load_weight import load_dataset
from Model.VGG import Vgg16
from Mask.Circle_Mask import apply_circle_mask
from utils import select_optimizer, auto_mkdir, restore_training, log10, save_results
curr_path = os.getcwd()

parser = argparse.ArgumentParser()
# Training settings
parser.add_argument("--num_epochs", dest='num_epochs', type=int, default=400, help="specify number of epochs")
parser.add_argument("--D_lr", type=float, default=0.00001, help="specify learning rate")
parser.add_argument("--G_lr", type=float, default=0.00001, help="specify learning rate")
parser.add_argument("--restore", default='', type=str,  help="False or True")
parser.add_argument("--restore_path", default='', type=str, help="specify the ckpt folder")
parser.add_argument('--batch_size', type=int, default = 8, help='gen and disc batch size')
parser.add_argument('--resize', type=int, default=512)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=400, help='number of epochs for training')
parser.add_argument('--model_save_interval', type=int, default=2, help='save weights interval -> step record')
parser.add_argument('--opt', help='optimizer for train, default [adam]',
                        dest='OPT', default='adam', choices=['adam', 'sgd_m', 'sgd', 'RMSP'])
parser.add_argument('--gan', dest='gan', default='wgan_gp', choices=['sphere', 'wgan_gp', 'pgan', 'gan'])
parser.add_argument('--dis_norm_type', help='normalization', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn'])
parser.add_argument('--gen_norm_type', help='normalization', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn'])
parser.add_argument('--num_layers', default=3, choices=(2, 3, 4, 5), type=int)
parser.add_argument('--act', help='activation', default='leak_relu', choices=['swish','leak_relu', 'relu'])
parser.add_argument('--vgg_ratio', type=float, default=0.001)
parser.add_argument('--ckpt_vgg', type=str, default = 'VGG_ckpt/vgg16.npy', help='the ckpt_vgg file')
parser.add_argument('--vgg', type=str, default="False", help='the ckpt_vgg file')
# Data settings and save path
parser.add_argument('--dataroot', type=str, default='', help='path to training data')
parser.add_argument("--Validation", type=str, default = "False", help="False or True")
parser.add_argument('--Validation_path', type=str, default='validation', help='path to Validation dataset')
parser.add_argument('--logdir_path', type=str, default='LDR_Training_output', help='path of logs')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for GPU')
parser.add_argument('--model_name', type=str, default='auto', help='folder name to save weights')
# Mask settings
parser.add_argument("--input_mask", type=str, default = "False", help="False or True")
parser.add_argument("--output_mask", type=str, default = "False", help="False or True")
parser.add_argument('--mask', type=str, default = "False", help = 'use the mask')
parser.add_argument("--final_output_mask", type=str, default="False", help="False or True")
args = parser.parse_args()

_clip = lambda x: tf.clip_by_value(x, 0, 1)

def build_graph(
        ldr,  # [b, h, w, c]
        hdr,  # [b, h, w, c]
        lambda_l1,
        is_training,
):

    hdr = apply_circle_mask(hdr)
    model = SingleLuminance_network(ldr, hdr, args.batch_size, is_training, args)
    # Loss
    D_loss, G_loss_GAN, generator_image = model.compute_loss()
    if args.final_output_mask == "True":
        generator_image = apply_circle_mask(generator_image)
    #G_loss = G_loss_GAN + lambda_l1 * G_loss_L1 + lambda_underexposed_attention * U_loss + lambda_overexposed_attention * L_loss
    L1_loss = lambda_l1 * tf.reduce_mean(tf.abs(generator_image - hdr), axis=[1, 2, 3], keepdims=True)
    if args.vgg == 'True':
        vgg = Vgg16(args.ckpt_vgg)
        vgg.build(generator_image)
        vgg2 = Vgg16(args.ckpt_vgg)
        vgg2.build(hdr)
        perceptual_loss = tf.reduce_mean(tf.abs((vgg.pool1 - vgg2.pool1)), axis=[1, 2, 3], keepdims=True)
        perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool2 - vgg2.pool2)), axis=[1, 2, 3], keepdims=True)
        perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool3 - vgg2.pool3)), axis=[1, 2, 3], keepdims=True)
        loss = tf.reduce_mean((lambda_l1 * L1_loss + args.vgg_ratio * perceptual_loss))
    else:
        loss = lambda_l1 * tf.reduce_mean(L1_loss)

    G_loss = G_loss_GAN + loss
    D_vars = [v for v in tf.trainable_variables() if v.name.startswith("dis_")]
    G_vars = [v for v in tf.trainable_variables() if v.name.startswith("gen_")]
    learning_rate = args.D_lr
    D_optimizer = select_optimizer(args.OPT, learning_rate)
    G_optimizer = select_optimizer(args.OPT, learning_rate)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_train_opt = D_optimizer.minimize(D_loss, var_list=D_vars)
        G_train_opt = G_optimizer.minimize(G_loss, var_list=G_vars)
    mse = tf.reduce_mean(tf.square(generator_image  - hdr))
    psnr = 20.0 * log10(1.0) - 10.0 * log10(mse)
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss_GAN", G_loss_GAN)
    tf.summary.scalar("PSNR", psnr)
    merged = tf.summary.merge_all()
    return D_train_opt, G_train_opt, D_loss, G_loss,\
        tf.reduce_mean(L1_loss), generator_image, merged, psnr, G_loss_GAN


def train(args, lambda_l1):
    # data IO
    input_images, reference_images, total_train_images, iterator = load_dataset(args, 'train')
    if args.Validation == 'True':
        Validation_input_images, Validation_reference_images, Validation_total_images, Validation_iterator, filename_save = load_dataset(args, 'Validation')
    # Training variables
    num_epochs = args.num_epochs   
    batch_size = args.batch_size
    input_size = args.resize
    max_val = 255.0
    D_loss_accum = 0.0
    G_loss_accum = 0.0

    # Path for tf.summary.FileWriter and to store model checkpoints
    output_path, filewriter_path, checkpoint_path, Image_path, json_path = auto_mkdir(curr_path,args)
    """evaluation metrics output"""
    # TF placeholder for graph input
    image_A = tf.placeholder(tf.float32, [None, input_size, input_size, 3])
    image_B = tf.placeholder(tf.float32, [None, input_size, input_size, 3])
    is_training = tf.placeholder(tf.bool)
    # Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Specify the GPU device you want to use. Use device number, e.g., "0" for the first GPU.
    config.gpu_options.visible_device_list = "0"

    # Create a TensorFlow session with the modified configuration.
    D_train_opt, G_train_opt, D_loss, G_loss, G_loss_L1, generator_image, merged,\
         psnr, G_loss_GAN = build_graph(image_A, image_B, lambda_l1, is_training)
    
    results_filename_path = json_path
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=200)
    ######### Start training
    with tf.Session(config=config) as sess: 
        with tf.device('/gpu:'+str(args.gpu_ids)):
            # Initialize all variables and start queue runners
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(filewriter_path, sess.graph)
            # To continue training from one of the checkpoints
            start_epoch = restore_training(saver, sess, args)
            # Loop over number of epochs
            step = 0
            results = []
            
            best_performance = float('-inf')
            for epoch in range(start_epoch,num_epochs):
                sess.run(iterator.initializer) 
                # Validation network
                print("GAN training epcoh {} begins: ".format(str(epoch)))
                # After each epoch, calculate the PSNR on the Validation set
                
                if args.Validation == "True":
                    if epoch % 1 == 0:
                        # 初始化测试集迭代器
                        sess.run(Validation_iterator.initializer)
                        # 在测试集上计算性能指标，例如PSNR
                        performance_score = 0
                        num_batches = Validation_total_images
                        for Validation_iter in range(num_batches):
                            Validation_batch_A, Validation_batch_B = sess.run([Validation_input_images, Validation_reference_images])
                            Validation_psnr_val = sess.run(psnr, feed_dict={image_A: Validation_batch_A.astype('float32'), image_B: Validation_batch_B.astype('float32'), is_training: False})
                            performance_score += Validation_psnr_val
                        psnr_score = performance_score / num_batches
                        print("PSNR on Validation dataset: {}".format(psnr_score))
                        results.append({
                            'params': epoch,
                            'score': psnr_score
                        })
                        save_results(results, results_filename_path)                  
                        
                    if performance_score > best_performance:
                        # if the performance score is better than the best performance score, save the model
                        print("start save")
                        for f in glob.glob(checkpoint_path + "/model_best_*"):
                            os.remove(f)
                        # save the model
                        checkpoint_name = os.path.join(checkpoint_path, 'model_best_{}.ckpt'.format(str(epoch)))
                        saver.save(sess, checkpoint_name)
                        # update the best performance score
                        best_performance = performance_score
                        print("end")
                else:
                    if epoch == 0 or epoch % args.model_save_interval == 0:
                        print('start save')
                        for f in glob.glob(checkpoint_path+args.model_name + "/model_epoch"+str(epoch-1)+"*"):
                            os.remove(f)
                        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch{}.ckpt'.format(str(epoch)))
                        saver.save(sess, checkpoint_name)
                        print('finish save')
                    
                for iter_id in np.arange(0, total_train_images - batch_size, batch_size):
                    # Get a batch of images (paired)                
                    step += 1
                    batch_A, batch_B = sess.run([input_images, reference_images])
                    summary, _, d_loss = sess.run([merged, D_train_opt, D_loss],\
                        feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), is_training: True})
                    _, g_loss, psnr_val, G_loss_GAN_val, l1_val = sess.run([G_train_opt, G_loss,\
                        psnr, G_loss_GAN, G_loss_L1], feed_dict={image_A: batch_A.astype('float32'), \
                            image_B: batch_B.astype('float32'), is_training: True}) 

                    # Record losses for display
                    D_loss_accum += d_loss
                    G_loss_accum += g_loss
                    average_d_loss = (float)(D_loss_accum)/step
                    average_g_loss = (float)(G_loss_accum)/step
                    print("iteration {} of epcoh {} for GAN training's D_loss {}, G_loss {}, G_loss_GAN_val {}".format(str(iter_id),\
                            str(epoch),str(average_d_loss), str(average_g_loss), str(G_loss_GAN_val)))
                    print("GAN training's G_loss_l1 {}, PSNR {}".format(str(l1_val), str(psnr_val)))
                    train_writer.add_summary(summary, epoch*total_train_images + iter_id)

                    if iter_id % 6000 == 0:
                        """Calculate the index after each epoch"""
                        fake_B = sess.run(generator_image, feed_dict={image_A: batch_A, image_B: batch_B, is_training: False})
                        fake_B = np.array(fake_B)
                        range_num = 2 if args.batch_size > 2 else args.batch_size
                        for num_img in range(range_num):
                            img1 = fake_B[num_img] * max_val
                            save_path = os.path.join(Image_path, 'epoch{}_step{}_preTrain_{}.jpg'.format(str(epoch), str(iter_id), str(num_img)))
                            cv2.imwrite(save_path, img1)
            train_writer.close()
            return best_performance

def main(args):
    lambda_l1 = 100.0
    psnr_score = train(args, lambda_l1)
    print('PSNR on Validation dataset: {}'.format(psnr_score))
        
if __name__ == '__main__':
    main(args)
