from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name=name)


class ParameterizedDistribution:
    def distribution(self, input):
        raise NotImplementedError

    def regularization_loss(self, reg_fn):
        raise NotImplementedError


class ParameterizedGaussian(ParameterizedDistribution):
    def distribution(self, x):
        mu, logvar = self.mu_and_logvar(x)
        std = tf.exp(0.5 * logvar)
        return tf.contrib.distributions.MultivariateNormalDiag(mu, std)

    def mu_and_logvar(self, x):
        raise NotImplementedError


class MLP:
    def __init__(self,
                 input_dim,
                 output_dim,
                 name,
                 hidden_sizes=(20,),
                 nonlinearity=tf.nn.relu,
                 activate_output=False,
    ):
        self.input_dim = input_dim      
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.activate_output = activate_output

        with tf.variable_scope(name):
            self.weights = []
            self.biases = []
            self.sizes = list(self.hidden_sizes) + [self.output_dim]
            dim = self.input_dim
            for i, z_dim in enumerate(self.sizes):
                self.weights.append(weight_variable([dim, z_dim], name="W{}".format(i)))
                self.biases.append(bias_variable([z_dim], name="b{}".format(i)))
                dim = z_dim

    def fn(self, x):
        self.layers = [x]
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre_activation = tf.matmul(self.layers[-1], weight) + bias
            if i == len(self.weights) - 1 and not self.activate_output:
                layer = pre_activation
            else:
                layer = self.nonlinearity(pre_activation)
            self.layers.append(layer)
        return self.layers[-1]

    def regularization_loss(self, reg_fn):
        return sum([reg_fn(weight) for weight in self.weights])


class GaussianMLP(ParameterizedGaussian):
    def __init__(self,
                 input_dim,
                 output_dim,
                 name,
                 hidden_sizes=(20,),
                 nonlinearity=tf.nn.relu,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.final_hidden_dim = hidden_sizes[-1]

        with tf.variable_scope(name):
            self.network = MLP(input_dim=input_dim,
                               output_dim=self.final_hidden_dim,
                               name="network",
                               hidden_sizes=hidden_sizes[:-1],
                               nonlinearity=nonlinearity,
                               activate_output=True,
            )

            self.logvar_weight = weight_variable([self.final_hidden_dim, self.output_dim], name="W_logvar")
            self.logvar_bias = bias_variable([self.output_dim], name="b_logvar")
            self.mu_weight = weight_variable([self.final_hidden_dim, self.output_dim], name="W_mu")
            self.mu_bias = bias_variable([self.output_dim], name="b_mu")

    def mu_and_logvar(self, x):
        final_hidden = self.network.fn(x)

        mu = tf.matmul(final_hidden, self.mu_weight) + self.mu_bias
        logvar = tf.matmul(final_hidden, self.logvar_weight) + self.logvar_bias
        return mu, logvar

    def regularization_loss(self, reg_fn):
        return self.network.regularization_loss(reg_fn) + reg_fn(self.mu_weight) + reg_fn(self.logvar_weight)


class BernoulliMLP(ParameterizedDistribution):
    def __init__(self, input_dim, output_dim, name, hidden_sizes=(20,), nonlinearity=tf.nn.relu):
        with tf.variable_scope(name):
            self.network = MLP(input_dim, output_dim, "network", hidden_sizes, nonlinearity)

    def distribution(self, x):
        logits = self.network.fn(x)
        return tf.contrib.distributions.Bernoulli(logits=logits)

    def regularization_loss(self, reg_fn):
        return self.network.regularization_loss(reg_fn)


class GaussianVAE:
    """
    Variational Autoencoder in which the prior is an (unparameterized) centered isotropic Gaussian and
    the approximate posterior (encoder) is parameterized as a multivariate Gaussian with diagonal
    covariance. The conditional distribution is any parameterized distribution.

    Concretely, self.encoder is a ParameterizedGaussian object and self.decoder is a ParameterizedDistribution
    object.
    """
    def __init__(self,
                 input_dim,
                 latent_dim,
                 encoder=None,
                 decoder=None,
                 reg_fn=None,
                 postprocessor=None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.reg_fn = reg_fn if reg_fn is not None else lambda _: 0
        self.postprocessor = postprocessor if postprocessor is not None else lambda _: None
        if self.encoder is None:
            self.encoder = GaussianMLP(input_dim, latent_dim)
        if self.decoder is None:
            self.decoder = BernoulliMLP(latent_dim, input_dim)

    def train(self,
              train_data_queue,
              filepath,
              batch_size=100,
              n_itr=int(1e6),
              learning_rate=0.01,
              print_every=100,
    ):
        x = tf.placeholder(tf.float32, shape=[None, self.input_dim])

        mu_latent, logvar_latent = self.encoder.mu_and_logvar(x)
        kl_div = -0.5 * tf.reduce_sum(1 + logvar_latent - tf.square(mu_latent) - tf.exp(logvar_latent), reduction_indices=1)

        epsilon = tf.random_normal(tf.shape(mu_latent))
        latent_code = mu_latent + epsilon * tf.exp(0.5 * logvar_latent)

        evidence = tf.reduce_sum(self.decoder.distribution(latent_code).log_prob(x), reduction_indices=1)
        x_hat = self.decoder.network.fn(latent_code)
        BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, x), reduction_indices=1)
        evidence = -BCE

        lower_bound = tf.reduce_mean(-kl_div + evidence)
        loss = -lower_bound
        regularized_loss = loss \
            + self.encoder.regularization_loss(self.reg_fn) \
            + self.decoder.regularization_loss(self.reg_fn)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(regularized_loss)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for itr in range(1, n_itr+1):
                batch, _ = train_data_queue.next_batch(batch_size)
                _, curr_loss = sess.run([train_step, loss], feed_dict={x: batch})

                if itr % print_every == 0:
                    saver.save(sess, filepath)
                    print("Itr {} : Loss = {}".format(itr, curr_loss))

    def generate(self, filepath, num_samples=1, use_logits=False):
        epsilon = tf.random_normal([num_samples, self.latent_dim])
        if use_logits:
            sample = self.decoder.distribution(epsilon).logits
        else:
            sample = self.decoder.distribution(epsilon).sample()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, filepath)

            sample_output = sess.run(sample)
            self.postprocessor(sample_output)
