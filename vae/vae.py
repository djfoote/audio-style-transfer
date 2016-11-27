import tensorflow as tf


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


class ParameterizedDistribution:
    def distribution(self, input):
        raise NotImplementedError

    def regularization_loss(self, reg_fn):
        raise NotImplementedError


class ParameterizedGaussian(ParameterizedDistribution):
    def distribution(self, x):
        mu, logvar = self.mu_and_logvar(x)
        diag_std_vector = tf.tile(logvar_conditional, [self.input_dim])
        return tf.contrib.distributions.MultivariateNormalDiag(mu_conditional, diag_std_vector)

    def mu_and_logvar(self, x):
        raise NotImplementedError


class MLP:
    def __init__(self,
                 input_dim,
                 output_dim,
                 name,
                 hidden_sizes=(20,),
                 nonlinearity=tf.nn.relu,
    ):
        self.input_dim = input_dim      
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity

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
        for weight, bias in zip(self.weights, self.biases):
            self.layers.append(tf.matmul(self.layers[-1], weight) + bias)
        return self.layers[-1]

    def regularization_loss(self, reg_fn):
        return sum([reg_fn(weight) for weight in self.weights])


class GaussianMLP(ParameterizedGaussian):
    def __init__(self,
                 input_dim,
                 mu_output_dim,
                 name,
                 hidden_sizes=(20,),
                 nonlinearity=tf.nn.relu,
    ):
        self.input_dim = input_dim      
        self.mu_output_dim = mu_output_dim
        self.final_hidden_dim = hidden_sizes[-1]

        with tf.variable_scope(name):
            self.network = MLP(input_dim=input_dim,
                               output_dim=self.final_hidden_dim,
                               name="hidden",
                               hidden_sizes=hidden_sizes[:-1],
                               nonlinearity=nonlinearity,
            )

            self.mu_weight = weight_variable([self.final_hidden_dim, self.mu_output_dim], name="W_mu")
            self.mu_bias = bias_variable([self.mu_output_dim], name="b_mu")
            self.logvar_weight = weight_variable([self.final_hidden_dim, 1], name="W_logvar")
            self.logvar_bias = bias_variable([1], name="b_logvar")

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
    the approximate posterior (encoder) is parameterized as a multivariate Gaussian with scaled isotropic
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
              learning_rate=1e-3,
              print_every=100,
    ):
        x = tf.placeholder(tf.float32, shape=[None, self.input_dim])

        mu_latent, logvar_latent = self.encoder.mu_and_logvar(x)
        kl_div = -0.5 * tf.reduce_sum(1 + logvar_latent - tf.square(mu_latent) - tf.exp(logvar_latent))

        epsilon = tf.random_normal([self.latent_dim])
        latent_code = mu_latent + epsilon * tf.exp(0.5 * logvar_latent)

        evidence = self.decoder.distribution(latent_code).log_prob(x)

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
                # import pdb; pdb.set_trace()
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
