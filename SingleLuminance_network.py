import numpy as np
import tensorflow as tf 
from utils import *
from Loss.Loss_function import gan_loss
from Loss.Loss_function import wgan_gp_loss
from Loss.Loss_function import pgan_loss
from Loss.Loss_function import sphere as sphere_gan
from model.Discriminator import PatchDiscriminator
from model.Generator import GeneratorModel


class SingleLuminance_network(object):
	def __init__(self, image_A_batch, image_B_batch, batch_size, is_training, args):    
		# Parse input arguments into class variables
		self.image_A = image_A_batch
		self.image_B = image_B_batch
		self.batch_size = batch_size
		self.is_training = is_training
		self.args = args

	def compute_loss(self):
		eps = 1e-12
		Gen_model = GeneratorModel(self.args, self.is_training)
		fake_B = Gen_model.graph(self.image_A)
		#fake_B = network.model(self.image_A)
		dis_model = PatchDiscriminator(self.args.dis_norm_type)
		fake_output_D = dis_model.graph(fake_B, reuse=None)
		real_output_D = dis_model.graph(self.image_B, reuse=True)
		"""GAN Function"""
		if self.args.gan == 'gan':
			self.d_loss, self.g_loss_gan = gan_loss(real_output_D, fake_output_D)
		elif self.args.gan == 'sphere':
			dis_real = real_output_D
			dis_fake = fake_output_D
			self.g_loss_gan, self.d_loss, (distance_real, distance_fake, g_convergence_to_zero, d_convergence_to_min) = \
                    sphere_gan(dis_real, dis_fake, None, 3, reuse=self.args.batch_size != 0)
		elif self.args.gan == 'wgan_gp':
			self.d_loss, self.g_loss_gan = wgan_gp_loss(dis_model, real_output_D, fake_output_D, self.image_B, fake_B, 10)
		elif self.args.gan == 'pgan':
			self.d_loss, self.g_loss_gan = pgan_loss(real_output_D, fake_output_D)
		
		#return self.d_loss, self.g_loss_gan, self.g_loss_l1, self.g_loss_mse, self.under_loss, self.Lum_loss, fake_B
		return self.d_loss, self.g_loss_gan, fake_B