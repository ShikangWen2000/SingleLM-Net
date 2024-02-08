import argparse
import tensorflow as tf
import os
import numpy as np
import cv2
import glob
from Model.Generator import GeneratorModel
from Model.VGG import Vgg16
from refinement_net import Refinement_net
from Data.Data_load_weight_finetune import load_dataset
from Mask.Circle_Mask import apply_circle_mask
from utils import auto_mkdir, log10, save_results, restore_training
# ---
parser = argparse.ArgumentParser()
# Training settings
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--resize', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--vgg_ratio', type=float, default=0.001)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--restore', type=str, default=False)
parser.add_argument('--restore_gan', type=str, default=False)
parser.add_argument('--restore_path', type=str, default='Refinement_out\Model\checkpoints')
parser.add_argument('--gan', dest='gan', default='wgan_gp', choices=['sphere', 'wgan_gp', 'pgan', 'gan'])
parser.add_argument('--dis_norm_type', help='normalization', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn'])
parser.add_argument('--gen_norm_type', help='normalization', default='sn', choices=['in', 'ln', 'nn', 'wn', 'sn', 'bn'])
parser.add_argument('--num_layers', default=3, choices=(2, 3, 4, 5), type=int)
parser.add_argument('--act', help='activation', default='leak_relu', choices=['swish','leak_relu', 'relu'])
parser.add_argument('--ckpt_gan', type=str, default = '', help='the gan ckpt file')
parser.add_argument('--ckpt_vgg', type=str, default = 'VGG_ckpt/vgg16.npy', help='the ckpt_vgg file')
parser.add_argument('--loss', type=str, default = 'L1', help='loss function')
parser.add_argument('--two_stage_network', type=str, default='Unet', help='two stage network')
# Data settings and save path
parser.add_argument('--logdir_path', type=str, default = 'Refinement_out')
parser.add_argument('--model_name', type=str, required=True, default = 'model_name')
parser.add_argument('--dataroot', type=str, required=True, help='The training dataroot')
parser.add_argument('--model_save_interval', type=int, default=2)
parser.add_argument('--Validation', type=str, default = "False", help='Validation or not')
parser.add_argument('--Validation_path', type=str, default = '', help='validation path')
# Mask settings
parser.add_argument('--mask', type=str, default = "False", help = 'use the mask')
parser.add_argument("--input_mask", type=str, default = "False", help="False or True")
parser.add_argument("--output_mask", type=str, default = "False", help="False or True")
parser.add_argument("--final_output_mask", type=str, default = "False", help="False or True")
args = parser.parse_args()
curr_path = os.getcwd()
def build_graph(
        ldr,  # [b, h, w, c]
        hdr,  # [b, h, w, c]
        is_training,
):
    Gen_model = GeneratorModel(args, is_training)
    fake_B = Gen_model.graph(ldr)
    # Refinement-Ne
    if args.two_stage_network == 'Unet':
        with tf.variable_scope("Refinement_Net"):
            refinement_model = Refinement_net(is_train=is_training)
            refinement_output = refinement_model.inference(fake_B, ldr)
    hdr = apply_circle_mask(hdr)
    if args.final_output_mask == "True":
        refinement_output = apply_circle_mask(refinement_output)

    if args.loss == 'L1':
        loss = tf.reduce_mean(tf.abs(refinement_output - hdr), axis=[1, 2, 3], keepdims=True)
    elif args.loss == 'L2':
        squared_difference = tf.square(refinement_output - hdr)
        loss = tf.reduce_mean(squared_difference, axis=[1, 2, 3], keepdims=True)
    # Add the perceptual loss
    if args.vgg_ratio != 0:
        vgg = Vgg16(args.ckpt_vgg)
        vgg.build(refinement_output)
        vgg2 = Vgg16(args.ckpt_vgg)
        vgg2.build(hdr)
        perceptual_loss = tf.reduce_mean(tf.abs((vgg.pool1 - vgg2.pool1)), axis=[1, 2, 3], keepdims=True)
        perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool2 - vgg2.pool2)), axis=[1, 2, 3], keepdims=True)
        perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool3 - vgg2.pool3)), axis=[1, 2, 3], keepdims=True)
        all_loss = tf.reduce_mean((loss + args.vgg_ratio * perceptual_loss))
    else:
        perceptual_loss = tf.constant(0)
        all_loss = tf.reduce_mean((loss))

    learning_rate = args.learning_rate
    trainable_vars = tf.trainable_variables()
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
            epsilon=1e-8, use_locking=False).minimize(all_loss, global_step=global_step, var_list=trainable_vars)
    
    #calculate psnr
    mse = tf.reduce_mean(tf.square(apply_circle_mask(refinement_output) - hdr))
    psnr = 20.0 * log10(1.0) - 10.0 * log10(mse)
    tf.summary.scalar('loss', tf.reduce_mean(loss))
    return train_op, tf.reduce_mean(loss), psnr, refinement_output, perceptual_loss

input_images, reference_images, total_train_images, iterator = load_dataset(args, 'train')
if args.Validation == "True":
    Validation_input_images, Validation_reference_images, Validation_total_images, Validation_iterator, input_filename_save = load_dataset(args, 'Validation')

batch_size = args.batch_size
b, h, w, c = batch_size, 512, 512, 3
# TF placeholder for graph input
image_A = tf.placeholder(tf.float32, [None, h, w, 3])
image_B = tf.placeholder(tf.float32, [None, h, w, 3])
is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0, trainable=False, name='global_step')
# Build the graph for the deep net
train_op, loss, psnr, refinement_output, perceptual_loss = build_graph(image_A, image_B, is_training)
saver = tf.train.Saver(tf.all_variables(), max_to_keep=200)
output_path, filewriter_path, checkpoint_path, Image_path, json_path = auto_mkdir(curr_path, args)
# ---
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord=coord)
sess.run(tf.global_variables_initializer())
total_parameters = 0

for variable in tf.trainable_variables():
    if 'Refinement_Net' or 'gen_' in variable.name:
    # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
print(total_parameters)
#restore the ckpt in the name space of GAN
if args.restore_gan == 'True':
    restorer1 = tf.train.Saver(
        #var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'gen_' in var.name or 'dis_' in var.name])
        var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'gen_' in var.name])
    restorer1.restore(sess, args.ckpt_gan)

summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(
    os.path.join(filewriter_path, 'summary'),
    sess.graph,
)

# To continue training from one of the checkpoints
start_epoch = 0
start_epoch = restore_training(saver, sess, args)

best_performance = float('-inf')
# initialize the results list
results = []
results_filename_path = json_path

for epoch in range(start_epoch, args.epoch):
    sess.run(iterator.initializer)
    if args.Validation == "True":
        if epoch % 1 == 0:
            # initialize the validation iterator
            sess.run(Validation_iterator.initializer)
            # calculate the performance score
            performance_score = 0
            num_batches = Validation_total_images
            for Validation_iter in range(num_batches):
                Validation_batch_A, Validation_batch_B = sess.run([Validation_input_images, Validation_reference_images])
                psnr_val = sess.run(psnr, feed_dict={image_A: Validation_batch_A.astype('float32'), image_B: Validation_batch_B.astype('float32'), is_training: False})
                performance_score += psnr_val
            performance_score /= num_batches
            print("PSNR on Validation set: {}".format(performance_score))
            # record the results
            results.append({
                'model_name': str(epoch),
                'score': performance_score
                    })
            save_results(results, results_filename_path)
        if performance_score > best_performance:
            # if the performance score is better than the best performance score
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
        it = epoch*total_train_images + iter_id
        batch_A, batch_B = sess.run([input_images, reference_images])
        _, summary_val, loss_val, psnr_val, perceptual_loss_val = sess.run([train_op, summary, loss, psnr, perceptual_loss],\
            feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), is_training: True})
        print("epoch {}, it {}, loss {}".format(str(epoch), str(iter_id), str(loss_val)))
        print("perceptual_loss: {}".format(perceptual_loss_val))
        print("Psnr: {}".format(psnr_val))
        if iter_id % 6000 == 0:
            fake_B = sess.run(refinement_output, feed_dict={image_A: batch_A, image_B: batch_B, is_training: True})
            fake_B = np.array(fake_B)
            range_num = 2 if args.batch_size > 2 else args.batch_size
            for num_img in range(range_num):
                img1 = fake_B[num_img] * 32768.0
                save_path = os.path.join(Image_path, 'epoch{}_step{}_preTrain_{}.hdr'.format(str(epoch), str(iter_id), str(num_img)))
                cv2.imwrite(save_path, img1)
            summary_writer.add_summary(summary_val, it)
coord.request_stop()
coord.join(threads)
