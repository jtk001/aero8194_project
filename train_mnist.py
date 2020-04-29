# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:51:23 2020

@author: Jonathan Kadowaki

"""

import argparse as ap
import os
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from vae import VAE, CVAE


parser = ap.ArgumentParser()
parser.add_argument('-c', '--cond',
                    help='use CVAE',
                    action='store_true')
parser.add_argument('-ep', '--epochs',
                    help='number of epochs to train',
                    type=int,
                    default=200)
parser.add_argument('-lr', '--learning_rate',
                    help='learning rate',
                    type=float,
                    default=1e-3)
parser.add_argument('-ld', '--latent_dim',
                    help='latent dimension',
                    type=int,
                    default=12)
parser.add_argument('-bs', '--batch_size',
                    help='batch size',
                    type=int,
                    default=256)
parser.add_argument('-b', '--beta',
                    help='KL Divergence weight',
                    type=float,
                    default=0.5)
args = parser.parse_args()

conditional = args.cond
epochs = args.epochs
lr = args.learning_rate
latent_dim = args.latent_dim
bs = args.batch_size
beta = args.beta

model_dir = 'models'
data_dir = 'train_data'
dirs = [model_dir, data_dir]
for direc in dirs:
    if not os.path.isdir(direc):
        os.makedirs(direc)

    
def preprocess_input(x):
    return x / 255.


def train_conditional(data, optimizer):
    (X_train, y_train), (X_test, y_test) = data
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    fname = os.path.join(data_dir, 'cvae_train.csv')
    logger = tf.keras.callbacks.CSVLogger(filename=fname)
    encoder, decoder, cvae = CVAE(input_dim=X_train.shape[-1],
                                  latent_dim=latent_dim,
                                  aux_dim=y_train.shape[-1],
                                  beta=beta,
                                  output_activation='sigmoid').build()
    print(cvae.summary())
    cvae.compile(optimizer, loss=None)
    cvae.fit(x=[X_train, y_train],
             y=None,
             validation_data=([X_test, y_test], None),
             epochs=epochs,
             batch_size=bs,
             shuffle=True,
             verbose=2,
             callbacks=[logger])
    print('\nfinished training, saving models...')
    cvae.save(os.path.join(model_dir, 'cvae.h5'))
    encoder.save(os.path.join(model_dir, 'conditional_encoder.h5'))
    decoder.save(os.path.join(model_dir, 'conditional_decoder.h5'))
    
    
def train_vanilla(data, optimizer):
    (X_train, _), (X_test, _) = data
    fname = os.path.join(data_dir, 'vae_train.csv')
    logger = tf.keras.callbacks.CSVLogger(filename=fname)
    encoder, decoder, vae = VAE(input_dim=X_train.shape[-1],
                                latent_dim=latent_dim,
                                beta=beta,
                                output_activation='sigmoid').build()
    print(vae.summary())
    vae.compile(optimizer, loss=None)
    vae.fit(x=X_train,
            y=None,
            validation_data=(X_test, None),
            epochs=epochs,
            batch_size=bs,
            shuffle=True,
            verbose=2,
            callbacks=[logger])
    print('\nfinished training, saving models...')
    vae.save(os.path.join(model_dir, 'vae.h5'))
    encoder.save(os.path.join(model_dir, 'encoder.h5'))
    decoder.save(os.path.join(model_dir, 'decoder.h5'))
  
  
def main():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = preprocess_input(X_train.reshape(X_train.shape[0],-1))
    X_test = preprocess_input(X_test.reshape(X_test.shape[0],-1))
    opt = tf.keras.optimizers.Adam(lr)
    data = [(X_train, y_train), (X_test, y_test)]
    if conditional:
        train_conditional(data, opt)
    else:
        train_vanilla(data, opt)


if __name__ == '__main__':
    main()