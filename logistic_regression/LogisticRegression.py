import numpy as np
import os

class LogisticRegression():
	def __init__(self, theta, alpha=0.1, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.eps = 1e-15

	def	predict_(self, x):
		x_ = np.hstack((np.ones((x.shape[0], 1)), x))
		return 1 / (1 + np.exp(-x_ @ self.theta))

	def	loss_elem_(self, y, y_hat):
		return (y * np.log(y_hat + self.eps) + (1 - y) * np.log(1 - y_hat + self.eps))

	def	loss_(self, x, y):
		y_hat = self.predict_(x)
		return -np.mean(self.loss_elem_(y, y_hat))

	def	gradient_(self, x, y):
		m, _ = x.shape
		x_p = np.hstack((np.ones((m, 1)), x))
		return x_p.T @ (self.predict_(x) - y) / m

	def	normalize_data(self, x):
		mean = np.mean(x, axis=0)
		std_dev = np.std(x, axis=0)
		norm = (x - mean) / std_dev
		return norm

	def	fit_(self, x: np.array, y: np.array):
		x = self.normalize_data(x)
		for _ in range(self.max_iter):
			print(self.loss_(x, y))
			self.theta -= (self.alpha * self.gradient_(x, y))
			loss = self.loss_(x, y)
			if loss < 0.01:
				break
		return self.theta
