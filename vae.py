# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:05:50 2020

@author: Jonathan Kadowaki

"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model


class SamplingLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super().__init__(**kwargs)
    
    def call(self, inputs):
        z_mu, z_log_sigma = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mu))
        return z_mu + tf.math.exp(z_log_sigma)*epsilon
    
    def compute_output_shape(self,input_shape):
        return input_shape[0]
    
    
class KLDivergenceLayer(tf.keras.layers.Layer):

    def __init__(self, beta=0.5, **kwargs):
        self.beta = beta
        self.is_placeholder = True
        super().__init__(**kwargs)
    
    def call(self, inputs):
        z_mu, z_log_sigma = inputs
        z_var = tf.math.square(tf.math.exp(z_log_sigma))
        kl_loss = self.beta * tf.reduce_sum(tf.square(z_mu) + z_var - 2*z_log_sigma - 1, axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss))
        return inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'beta': self.beta})
        return config
    
    
class ReconstructionLossLayer(tf.keras.layers.Layer):

    def __init__(self, dim, bernoulli=False, **kwargs):
        self.dim = dim
        self.bernoulli = bernoulli
        super().__init__(**kwargs)
    
    def call(self, inputs):
        x_in, x_recon = inputs
        if not self.bernoulli:
            # from multivariate normal distribution assumption
            recon_loss = tf.reduce_sum(tf.math.squared_difference(x_in, x_recon), axis=-1)
        else:
            # multivariate bernoulli assumption
            recon_loss = self.dim*tf.keras.losses.binary_crossentropy(x_in, x_recon)
        self.add_loss(tf.reduce_mean(recon_loss))
        return inputs[-1]
    
    def compute_output_shape(self, input_shape):
        return input_shape[-1]
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'dim': self.dim,
                       'bernoulli': self.bernoulli})
        return config
 
 
class VAE():

    def __init__(self,
                 input_dim,
                 latent_dim=12,
                 beta=0.5,
                 output_activation='sigmoid'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.output_activation = output_activation
        if self.output_activation == 'sigmoid':
            self.bernoulli = True
        else:
            self.bernoulli = False
        
    def build(self):
        input_img = Input(self.input_dim, name='image-input')
        x = Dense(256, activation='relu', name='hidden-downsample-1')(input_img)
        x = Dense(64, activation='relu', name='hidden-downsample-2')(x)
        z_mu = Dense(self.latent_dim, activation=None, name='z-mean')(x)
        z_log_sigma = Dense(self.latent_dim, activation=None, name='z-log-sigma')(x)
        z_mu, z_log_sigma = KLDivergenceLayer(beta=self.beta)([z_mu, z_log_sigma])
        encoder = Model(inputs=input_img, outputs=[z_mu, z_log_sigma], name='encoder')
        z = SamplingLayer(name='reparametrization-trick-sampling')([z_mu, z_log_sigma])
        
        latent_input = Input(self.latent_dim)
        x = Dense(64, activation='relu', name='hidden-upsample-1')(latent_input)
        x = Dense(256, activation='relu', name='hidden-upsample-2')(x)
        x = Dense(self.input_dim, activation=self.output_activation, name='reconstructed-image')(x)
        decoder = Model(inputs=latent_input, outputs=x, name='decoder')
        
        vae_recon = decoder(z)
        vae_recon = ReconstructionLossLayer(dim=self.input_dim, bernoulli=self.bernoulli)([input_img, vae_recon])
        vae_model = Model(inputs=input_img, outputs=vae_recon, name='vae')
        return encoder, decoder, vae_model


class CVAE():

    def __init__(self,
                 input_dim,
                 latent_dim=12,
                 aux_dim=10,
                 beta=0.5,
                 output_activation='sigmoid'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.beta = beta
        self.output_activation = output_activation
        if self.output_activation == 'sigmoid':
            self.bernoulli = True
        else:
            self.bernoulli = False
        
    def build(self):
        input_img = Input(self.input_dim, name='image-input')
        input_aux_1 = Input(self.aux_dim, name='aux-input-1')
        x = Dense(256, activation='relu', name='hidden-downsample-1')(input_img)
        x = Concatenate(name='merge-inputs')([x, input_aux_1])
        x = Dense(64, activation='relu', name='hidden-downsample-2')(x)
        z_mu = Dense(self.latent_dim, activation=None, name='z-mean')(x)
        z_log_sigma = Dense(self.latent_dim, activation=None, name='z-log-sigma')(x)
        z_mu, z_log_sigma = KLDivergenceLayer(beta=self.beta)([z_mu, z_log_sigma])
        encoder = Model(inputs=[input_img, input_aux_1], outputs=[z_mu, z_log_sigma], name='encoder')
        z = SamplingLayer(name='reparametrization-trick-sampling')([z_mu, z_log_sigma])
        
        latent_input = Input(self.latent_dim)
        input_aux_2 = Input(self.aux_dim, name='aux-input-2')
        x = Dense(64, activation='relu', name='hidden-upsample-1')(latent_input)
        x = Concatenate(name='merge-inputs')([x, input_aux_2])
        x = Dense(256, activation='relu', name='hidden-upsample-2')(x)
        x = Dense(self.input_dim, activation=self.output_activation, name='reconstructed-image')(x)
        decoder = Model(inputs=[latent_input, input_aux_2], outputs=x, name='decoder')
        
        vae_recon = decoder([z, input_aux_1])
        vae_recon = ReconstructionLossLayer(dim=self.input_dim, bernoulli=self.bernoulli)([input_img, vae_recon])
        vae_model = Model(inputs=[input_img, input_aux_1], outputs=vae_recon, name='cvae')
        
        return encoder, decoder, vae_model
