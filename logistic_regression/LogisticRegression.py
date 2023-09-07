import numpy as np

class LogisticRegression():
	def __init__(self, theta, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta

	def	predict_(self, x):
		x_ = np.hstack((np.ones((x.shape[0], 1)), x))
		print(f"x: {-x_}\ntheta: {self.theta}")
		return 1 / (1 + np.exp(-x_ @ self.theta))

	def	loss_elem_(self, y, y_hat):
		eps = 1e-15
		return - (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

	def	loss_(self, x, y):
		y_hat = self.predict_(x)
		return np.mean(self.loss_elem_(y, y_hat))

	def	gradient_(self, x, y):
		m, _ = x.shape
		x_p = np.hstack((np.ones((m, 1)), x))
		return x_p.T @ (self.predict_(x) - y) / m

	def	fit_(self, x: np.array, y: np.array):
		for _ in range(self.max_iter):
			self.theta -= (self.alpha * self.gradient_(x, y))
		return self.theta
