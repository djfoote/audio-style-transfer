from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from vae import weight_variable, bias_variable


class Distribution(object):
	def log_prob(self, x):
		raise NotImplementedError

	def sample(self):
		raise NotImplementedError


class SemiMarkovModel(Distribution):
	def __init__(self, partial_conditionals, conditional):
		"""
		partial_conditionals[i] is the distribution P(X_i | X_1, ..., X_{i-1}). 
		Thus, partial_conditionals[0] is the prior.
		All distributions here are functions which take conditional values as input and return Distribution objects.
		"""
		self.partial_conditionals = partial_conditionals
		self.lookback = len(self.partial_conditionals)
		self.conditional = conditional

	def log_prob(self, x):
		lp = 0
		for i, cond_dist in enumerate(self.partial_conditionals):
			lp += cond_dist(x[:i]).log_prob(x[i])
		for i in range(self.lookback, len(x)):
			lp += self.conditional(x[i-self.lookback:i]).log_prob(x[i])
		return lp

	def sample(self, sample_length=100):
		result = []
		for cond_dist in self.partial_conditionals:
			result.append(cond_dist(result).sample())
		for _ in range(self.lookback, sample_length):
			result.append(self.conditional(result[-self.lookback:]).sample())
		return result


class SimpleSMM(SemiMarkovModel):
	"""
	Subclasses define a conditional distribution generator to build all distributions from given lookback

	If pad, pad with zeros at the beginning to apply same conditional distribution at all steps.
	If not pad, add parameters for all incomplete conditionals at the beginning of the sequence.
	"""
	def __init__(self, lookback, pad=False):
		self.lookback = lookback
		self.pad = pad
		if self.pad:
			raise NotImplementedError
		else:
			self.partial_conditionals = [self.conditional_from_prev_i(i) for i in range(self.lookback)]
		self.conditional = self.conditional_from_prev_i(self.lookback)

	def conditional_from_prev_i(self, i):
		raise NotImplementedError


class LinearCategoricalSMM(SimpleSMM):
	def __init__(self, lookback, num_categories, pad=False):
		self.num_categories = num_categories
		super(LinearCategoricalSMM, self).__init__(lookback)

	def conditional_from_prev_i(self, i):
		W = weight_variable(shape=(i, self.num_categories))
		b = bias_variable(shape=(self.num_categories,))
		return lambda prev_i: tf.contrib.distributions.Categorical(tf.matmul(prev_i, W) + b)
