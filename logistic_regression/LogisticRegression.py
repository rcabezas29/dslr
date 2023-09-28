import numpy as np
from tqdm import tqdm

class LogisticRegression():
	def __init__(self, theta: np.ndarray, alpha=0.005, max_iter=4500, stochastic=False, batch_size=32):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.stochastic = stochastic
		self.batch_size = batch_size
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

	def	fit(self, x: np.ndarray, y: np.ndarray):
		x = self.normalize_data(x)
		m, _ = x.shape
		
		for i in tqdm(range(self.max_iter)):
			if (self.stochastic == True):
				indices = np.random.permutation(m)
				x = x[indices]
				y = y[indices]

				for i in range(0, m, self.batch_size):
					x_batch = x[i:i+self.batch_size]
					y_batch = y[i:i+self.batch_size]

					self.theta -= self.alpha * self.gradient(x_batch, y_batch)

			else:
				self.theta -= (self.alpha * self.gradient(x, y))
			
			if self.loss(x, y) < 0.01:
				break
		return self.theta
