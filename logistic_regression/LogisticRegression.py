import numpy as np
from tqdm import tqdm

class LogisticRegression():
	def __init__(self, theta: np.ndarray, alpha=0.005, max_iter=4000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.eps = 1e-15

	def	sigmoid(self, x: np.ndarray):
		return 1 / (1 + np.exp(-x @ self.theta))

	def	predict(self, x: np.ndarray):
		x_ = np.hstack((np.ones((x.shape[0], 1)), x))
		return self.sigmoid(x_)

	def	loss_elem(self, y: np.ndarray, y_hat: np.ndarray):
		return (y * np.log(y_hat + self.eps) + (1 - y) * np.log(1 - y_hat + self.eps))

	def	loss(self, x: np.ndarray, y: np.ndarray):
		y_hat = self.predict(x)
		return -np.mean(self.loss_elem(y, y_hat))

	def	gradient(self, x: np.ndarray, y: np.ndarray):
		m, _ = x.shape
		x_p = np.hstack((np.ones((m, 1)), x))
		return x_p.T @ (self.predict(x) - y) / m

	def	normalize_data(self, x: np.ndarray):
		mean = np.mean(x, axis=0)
		std_dev = np.std(x, axis=0)
		norm = (x - mean) / std_dev
		return norm

	def	stochastic_fit(self, x: np.ndarray, y: np.ndarray):
		x = self.normalize_data(x)
		m, d = x.shape
		self.theta = np.random.randn(d + 1).reshape(-1, 1)

		for i in range(self.max_iter):
			indices = np.random.permutation(m)
			x = x[indices]
			y = y[indices]

			for i in range(0, m, 32):
				X_batch = x[i:i+32]
				y_batch = y[i:i+32]
				grad = self.gradient(X_batch, y_batch)
				self.theta -= self.alpha * grad

			loss = self.loss(x, y)
			if loss < 0.01:
				break

	def	fit(self, x: np.ndarray, y: np.ndarray):
		x = self.normalize_data(x)
		for _ in tqdm(range(self.max_iter)):
			self.theta -= (self.alpha * self.gradient(x, y))
			if self.loss(x, y) < 0.01:
				break
		return self.theta
