from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name=name)


class ConditionalDistribution:
    def distribution(self, z):
        raise NotImplementedError

    def log_prob(self, x, z):
        return self.distribution(z).log_prob(x)

    def sample(self, z):
        return self.distribution(z).sample()


class ParameterizedGaussian(ConditionalDistribution):
    def __init__(self, name=None):
        self.name = name

    def distribution(self, x):
        with tf.name_scope(self.name):
            mu, logvar = self.mu_and_logvar(x)
            std = tf.exp(0.5 * logvar)
            return tf.contrib.distributions.MultivariateNormalDiag(mu, std)

    def mu_and_logvar(self, x):
        raise NotImplementedError


class MLP:
    """
    Assumes rank 2 input where first axis is batch.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(20,),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 name=None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity if output_nonlinearity is not None else lambda z: z
        self.name = name

        with tf.variable_scope(self.name + "_init"):
            self.weights = []
            self.biases = []
            self.sizes = list(self.hidden_sizes) + [self.output_dim]
            dim = self.input_dim
            for i, z_dim in enumerate(self.sizes):
                with tf.variable_scope("layer{}_init".format(i)):
                    self.weights.append(weight_variable([dim, z_dim], name="W{}".format(i)))
                    self.biases.append(bias_variable([z_dim], name="b{}".format(i)))
                    dim = z_dim

    def fn(self, x):
        with tf.name_scope(self.name):
            self.layers = [x]
            for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                with tf.name_scope("layer{}".format(i)):
                    pre_activation = tf.matmul(self.layers[-1], weight) + bias
                    nonlinearity = self.hidden_nonlinearity if i < len(self.weights) - 1 else self.output_nonlinearity
                    self.layers.append(nonlinearity(pre_activation))
            return self.layers[-1]


class GaussianMLP(ParameterizedGaussian):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(20,),
                 nonlinearity=tf.nn.relu,
                 name=None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.final_hidden_dim = hidden_sizes[-1]
        self.name = name

        with tf.variable_scope(self.name + "_init"):
            self.network = MLP(input_dim=input_dim,
                               output_dim=self.final_hidden_dim,
                               hidden_sizes=hidden_sizes[:-1],
                               hidden_nonlinearity=nonlinearity,
                               output_nonlinearity=nonlinearity,
                               name="hidden_network",
            )

            self.logvar_weight = weight_variable([self.final_hidden_dim, self.output_dim], name="W_logvar")
            self.logvar_bias = bias_variable([self.output_dim], name="b_logvar")
            self.mu_weight = weight_variable([self.final_hidden_dim, self.output_dim], name="W_mu")
            self.mu_bias = bias_variable([self.output_dim], name="b_mu")

    def mu_and_logvar(self, x, name=None):
        with tf.name_scope(name):
            with tf.name_scope(self.name):
                final_hidden = self.network.fn(x)

                with tf.name_scope('mu'):
                    mu = tf.matmul(final_hidden, self.mu_weight) + self.mu_bias
                with tf.name_scope('logvar'):
                    logvar = tf.matmul(final_hidden, self.logvar_weight) + self.logvar_bias
                return mu, logvar


class BernoulliMLP(ConditionalDistribution):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(20,),
                 hidden_nonlinearity=tf.nn.relu,
                 name=None,
    ):
        self.name = name

        with tf.variable_scope(self.name + "_init"):
            self.network = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                name="network",
            )

    def distribution(self, x, name=None):
        with tf.name_scope(name):
            with tf.name_scope(self.name):
                logits = self.network.fn(x)
                return tf.contrib.distributions.Bernoulli(logits=logits)


class GaussianVAE:
    """
    Variational Autoencoder in which the prior is an (unparameterized) centered isotropic Gaussian and
    the approximate posterior (encoder) is parameterized as a multivariate Gaussian with diagonal
    covariance. The conditional distribution is any parameterized distribution.

    Concretely, self.encoder is a ParameterizedGaussian object and self.decoder is a ConditionalDistribution
    object.
    """
    def __init__(self,
                 input_dim,
                 latent_dim,
                 logdir,
                 encoder=None,
                 decoder=None,
                 postprocessor=None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.logdir = logdir
        self.encoder = encoder
        self.decoder = decoder
        self.postprocessor = postprocessor if postprocessor is not None else lambda _: None
        if self.encoder is None:
            self.encoder = GaussianMLP(input_dim, latent_dim)
        if self.decoder is None:
            self.decoder = BernoulliMLP(latent_dim, input_dim)
        self.initialize_model()

    def initialize_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="x")
        mu_latent, logvar_latent = self.encoder.mu_and_logvar(self.x)

        with tf.name_scope("kld"):
            kl_div = -0.5 * tf.reduce_sum(1 + logvar_latent - tf.square(mu_latent) - tf.exp(logvar_latent), reduction_indices=1)

        with tf.name_scope("epsilon"):
            epsilon = tf.random_normal(tf.shape(mu_latent))

        with tf.name_scope("latent_code"):
            with tf.name_scope("std"):
                std = tf.exp(0.5 * logvar_latent)
            latent_code = mu_latent + epsilon * std

        evidence = tf.reduce_sum(self.decoder.log_prob(self.x, latent_code), reduction_indices=1, name="evidence")

        with tf.name_scope("loss"):
            lower_bound = tf.reduce_mean(-kl_div + evidence, name="lower_bound")
            self.loss = -lower_bound 

        self.saver = tf.train.Saver()

        writer = tf.train.SummaryWriter(self.logdir)
        writer.add_graph(tf.get_default_graph())

    def train(self,
              train_data_queue,
              checkpoint="model.ckpt",
              batch_size=100,
              n_itr=int(1e6),
              learning_rate=0.01,
              print_every=100,
    ):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        checkpoint_path = self.logdir + "/" + checkpoint
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for itr in range(1, n_itr+1):
                batch, _ = train_data_queue.next_batch(batch_size)
                _, curr_loss = sess.run([train_step, self.loss], feed_dict={self.x: batch})

                if itr % print_every == 0:
                    self.saver.save(sess, checkpoint_path)
                    print("Itr {} : Loss = {}".format(itr, curr_loss))

    def generate(self, checkpoint="model.ckpt", num_samples=1, use_logits=False):
        checkpoint_path = self.logdir + "/" + checkpoint
        epsilon = tf.random_normal([num_samples, self.latent_dim])
        if use_logits:  # Hack to reproduce MNIST results from VAE paper
            sample = self.decoder.distribution(epsilon).logits
        else:
            sample = self.decoder.sample(epsilon)

        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint_path)

            sample_output = sess.run(sample)
            self.postprocessor(sample_output)
