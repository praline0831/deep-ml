import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None
		self.optimizer = None

	def initialize(self, optimizer):
		# Initialize weights W, biases w0, and optimizers
		limit = math.sqrt(1.0 / self.input_shape[0])
		self.W = np.random.uniform(-limit, limit, size=(self.input_shape[0], self.n_units))
		self.w0 = np.zeros(self.n_units)
		self.optimizer = optimizer

	def parameters(self):
		# Return total number of parameters
		return self.n_units * self.input_shape[0] + self.n_units

	def forward_pass(self, X, training=True):
		# Compute and return the forward pass
		self.layer_input = X
		return X @ self.W + self.w0

	def backward_pass(self, accum_grad):
		# Compute gradients, update weights if trainable, return grad w.r.t. input
		grad_W = self.layer_input.T @ accum_grad   # (in, B) @ (B, out) → (in, out) ✅
		grad_w0 = accum_grad.sum(axis=0)           # 不要直接把 accum_grad 丢进去
		grad_input = accum_grad @ self.W.T           # ← 更新前算
		self.W = self.optimizer.update(self.W, grad_W)
		self.w0 = self.optimizer.update(self.w0, grad_w0)
		return grad_input

	def output_shape(self):
		# Return output shape tuple
		return (self.n_units,)
