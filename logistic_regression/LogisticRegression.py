import numpy as np
from tqdm import tqdm

class LogisticRegression():
	def __init__(self, theta, alpha=0.005, max_iter=4000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.eps = 1e-15

	def	sigmoid(self, x):
		return 1 / (1 + np.exp(-x @ self.theta))

	def	predict(self, x):
		x_ = np.hstack((np.ones((x.shape[0], 1)), x))
		return self.sigmoid(x_)

	def	loss_elem(self, y, y_hat):
		return (y * np.log(y_hat + self.eps) + (1 - y) * np.log(1 - y_hat + self.eps))

	def	loss(self, x, y):
		y_hat = self.predict(x)
		return -np.mean(self.loss_elem(y, y_hat))

	def	gradient(self, x, y):
		m, _ = x.shape
		x_p = np.hstack((np.ones((m, 1)), x))
		return x_p.T @ (self.predict(x) - y) / m

	def	normalize_data(self, x):
		mean = np.mean(x, axis=0)
		std_dev = np.std(x, axis=0)
		norm = (x - mean) / std_dev
		return norm

	def	fit(self, x: np.array, y: np.array):
		x = self.normalize_data(x)
		for _ in tqdm(range(self.max_iter)):
			self.theta -= (self.alpha * self.gradient(x, y))
			if self.loss(x, y) < 0.01:
				break
		return self.theta
