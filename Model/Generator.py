import tensorflow as tf

from model.Normalization import *
from model._conv import *
from Mask.Circle_Mask import apply_circle_mask, none_mask

class GeneratorModel:
        
    def __init__(self, args, is_training):
        self.ngf = args.ngf
        self.args = args
        self.batch_size = args.batch_size
        self.num_layers = args.num_layers
        self.size = args.resize
        self.is_training = is_training
        self.act = tf.nn.swish if args.act == 'swish' else tf.nn.leaky_relu
        self.mask = apply_circle_mask if args.mask == True else none_mask
        self._init_normalization(args.gen_norm_type)

    def _init_normalization(self, norm):
        assert norm in ['in', 'ln', 'nn', 'sn', 'wn', 'bn'], 'invalid norm: {}'.format(norm)
        self.conv2d = conv2d
        self.transpose_conv2d = transpose_conv2d
        self.norm = none_norm
        #self.norm = instance_norm
        if norm == 'in':
            self.norm = instance_norm
        elif norm == 'ln':
            self.norm = layer_norm
        elif norm == 'bn':
            self.norm = batch_norm
        elif norm == 'sn':
            self.conv2d = sn_conv2d
            self.transpose_conv2d = sn_transpose_conv2d
        elif norm == 'wn':
            self.conv2d = wn_conv2d
            self.transpose_conv2d = wn_transpose_conv2d

    def conv_block(self, input_tensor, num_outputs, block_name):
        conv = self.conv2d(input_tensor, num_outputs, kernel_size=(3, 3), strides=(1, 1), padding='same', name=block_name + '_conv')
        norm = self.norm(conv, name = block_name + '_batchnorm', is_training = self.is_training)
        activation = self.act(norm, alpha=0.2, name= block_name + '_activation')
        activation  = self.mask(activation)
        return activation

    def down_sampling(self, input_tensor, num_outputs, block_name):
        conv = self.conv2d(input_tensor, num_outputs, kernel_size=(3, 3), strides=(2, 2), padding='same', name=block_name + '_downsampling')
        norm = self.norm(conv, name= block_name + '_batchnorm', is_training = self.is_training)
        activation = self.act(norm, alpha=0.2, name= block_name + '_activation')
        activation  = self.mask(activation)
        return activation

    def up_sampling(self, input_tensor, num_outputs, block_name):
        deconv = self.transpose_conv2d(input_tensor, num_outputs, kernel_size=(3, 3), strides=(2, 2), padding='same', name=block_name + '_upsampling')
        norm = self.norm(deconv, name= block_name + '_batchnorm', is_training = self.is_training)
        activation = self.act(norm, alpha=0.2, name= block_name + '_activation')
        activation  = self.mask(activation)
        return activation

    def graph(self, image_A_input):
        image_A_input_mask = self.mask(image_A_input)
        if self.args.input_mask == 'True':
            image_A_input_mask = apply_circle_mask(image_A_input)

        with tf.variable_scope("gen_"):
            # G3 encoder area
            layer_number = 6
            scope_name = 'gen_G3_conv1'
            self.G3_conv1 = self.conv_block(image_A_input_mask, num_outputs=self.ngf, block_name = scope_name)

            scope_name = 'gen_G3_conv1_downsampling'
            self.G3_conv1_downsampling = self.down_sampling(self.G3_conv1, num_outputs=self.ngf*2, block_name = scope_name)
            
            # G2 encoder area
            scope_name = 'gen_G2_conv1'
            self.G2_conv1 = self.conv_block(self.G3_conv1_downsampling, num_outputs=self.ngf*2, block_name = scope_name)


            scope_name = 'gen_G2_conv1_downsampling'
            self.G2_conv1_downsampling = self.down_sampling(self.G2_conv1, num_outputs=self.ngf*4, block_name = scope_name)
            
            # G1 encoder area
            scope_name = 'gen_G1_pre_block'
            self.G1_pre_block = self.conv_block(self.G2_conv1_downsampling, num_outputs=self.ngf*8, block_name = scope_name)
            
            #G1_downsampling
            
            scope_name = "gen_G1_downsampling1"
            self.G1_downsampling = self.down_sampling(self.G1_pre_block, num_outputs=self.ngf*8, block_name = scope_name)


            #G1 res_block area
            for i in range(layer_number):
                scope_name = 'gen_G1_res_block' + str(i)
                self.G1_res_block = self.conv_block(self.G1_downsampling, num_outputs=self.ngf*16, block_name = scope_name)
                self.G1_downsampling = self.G1_res_block
            
            # G1 back area
            scope_name = 'gen_G1_back_block'
            self.G1_back_block = self.conv_block(self.G1_res_block, num_outputs=self.ngf*8, block_name = scope_name)

            #G1_upsampling
            scope_name = "gen_G1_upsampling1"
            self.G1_upsampling = self.up_sampling(self.G1_back_block, num_outputs=self.ngf*4, block_name = scope_name)

            # G2 structure
            self.G2_conv2 = tf.add(self.G1_upsampling, self.G2_conv1_downsampling)
            # G2 res_block area
            for i in range(layer_number):
                scope_name = 'gen_G2_res_block' + str(i)
                self.G2_res_block = self.conv_block(self.G2_conv2, num_outputs=self.ngf*2, block_name = scope_name)
                self.G2_conv2 = self.G2_res_block
            
            # G2 up_sampling area
            scope_name = 'gen_G2_upsampling'
            self.G2_upsampling = self.up_sampling(self.G2_res_block, num_outputs=self.ngf*2, block_name = scope_name)

            # G3 structure
            self.G3_conv2 = tf.add(self.G2_upsampling, self.G3_conv1_downsampling)
            # G3 res_block area
            for i in range(layer_number):
                scope_name = 'gen_G3_res_block' + str(i)
                self.G3_res_block = self.conv_block(self.G3_conv2, num_outputs=self.ngf, block_name = scope_name)
                self.G3_conv2 = self.G3_res_block
            
            # Output layer
            scope_name = "gen_G3_upsampling"
            self.G3_upsampling = self.up_sampling(self.G3_res_block, num_outputs=self.ngf, block_name = scope_name)
            
            #self.G3_conv3 = tf.add(self.G3_upsampling, self.G3_conv1)

            scope_name = "gen_output_layer"
            self.output_layer = self.conv2d(self.G3_upsampling, 3, kernel_size=(3, 3), strides=(1, 1), padding='same', name = scope_name)
            # tanh activation
            
            self.output_layer_activation = self.output_layer
            if self.args.mode == 'Train':
                self.generator_output_final = self.mask(self.output_layer_activation)
            if self.args.mode == 'Validation':
                self.generator_output_final = self.mask(self.output_layer_activation)
            if self.args.output_mask == 'True':
                self.generator_output_final = apply_circle_mask(self.output_layer_activation)
            return self.generator_output_final