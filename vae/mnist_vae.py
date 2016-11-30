from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

import argparse

from vae import GaussianMLP, BernoulliMLP, GaussianVAE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)
    parser.add_argument("--filepath", "-f", type=str, default='checkpoint/mnist.ckpt')
    parser.add_argument("--n-itr", "-n", type=int, default=int(1e6))
    args = parser.parse_args()
    return args


def mnist_postprocessor(image):
    image = sps.expit(image)
    num_samples, _ = image.shape
    num_rows = int(np.floor(np.sqrt(num_samples)))
    images = image.reshape(num_samples, 28, 28)
    fig = plt.figure()
    for i, image in enumerate(images):
        a = fig.add_subplot(num_rows, int(np.ceil(num_samples / num_rows)), i+1)
        plt.imshow(image, cmap='gray_r')
    plt.show()


def mnist_vae(latent_dim=2, enc_args={}, dec_args={}):
    return GaussianVAE(
        input_dim=784,
        latent_dim=latent_dim,
        encoder=GaussianMLP(784, latent_dim, name="encoder", **enc_args),
        decoder=BernoulliMLP(latent_dim, 784, name="decoder", **dec_args),
        postprocessor=mnist_postprocessor,
    )


def train_mnist(filepath, n_itr=int(1e6), latent_dim=2, enc_args={}, dec_args={}):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST')
    vae = mnist_vae(latent_dim, enc_args, dec_args)
    vae.train(mnist.train, filepath=filepath, n_itr=n_itr)


def generate_mnist(filepath, latent_dim=2, enc_args={}, dec_args={}):
    vae = mnist_vae(latent_dim, enc_args, dec_args)
    vae.generate(num_samples=16, filepath=filepath, use_logits=True)


if __name__ == '__main__':
    args = get_args()
    enc_args, dec_args = {"hidden_sizes": (400,)}, {"hidden_sizes": (400,)}
    if args.mode == "train":
        train_mnist(args.filepath, n_itr=args.n_itr, latent_dim=20, enc_args=enc_args, dec_args=dec_args)
    elif args.mode == "generate":
        generate_mnist(args.filepath, latent_dim=20, enc_args=enc_args, dec_args=dec_args)
    else:
        raise NotImplementedError
