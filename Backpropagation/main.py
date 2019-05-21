import numpy as np
from collections import OrderedDict
from common import *


class TwoLayer:
	def __init__(self):
		input_num = 2
		hidden_num = 5
		output_num = 2

		self.params = {}
		self.params['W1'] = 0.01 * np.random.randn(input_num, hidden_num)
		self.params['b1'] = np.zeros(hidden_num)
		self.params['W2'] = 0.01 * np.random.randn(hidden_num, hidden_num)
		self.params['b2'] = np.zeros(hidden_num)
		self.params['W3'] = 0.01 * np.random.randn(hidden_num, output_num)
		self.params['b3'] = np.zeros(output_num)

		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['ReLU'] = ReLU()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.layers['ReLU'] = ReLU()
		self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

		self.lastLayer = SoftmaxWithLoss()

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def calcGrads(self, x, t):
		# forward
		self.loss(x, t)

		#backward
		dout = 1
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)

		grads = {}
		grads['W1'], grads['b1'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
		grads['W2'], grads['b2'] = self.layers['Affine2'].dw, self.layers['Affine2'].db
		grads['W3'], grads['b3'] = self.layers['Affine3'].dw, self.layers['Affine3'].db
		return grads

	def train(self, x, t, iters):
		lr = 0.05  # 学習率
		for i in range(iters):
			grad = self.calcGrads(x, t)
			for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
				self.params[key] -= lr * grad[key]
			if i % 1000 == 0:
				print("Loss: ", self.loss(x, t))


def main():
	print("Train start!")
	tl = TwoLayer()
	x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
	t = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # XOR: non liner
	#t = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # AND: liner
	#t = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])  # NAND: liner

	tl.train(x, t, 50000)
	print("Train finish\n")
	print("x1 x2")
	for a in x:
		print(a, " => ", end="")
		print(np.argmax(tl.predict(a)))


if __name__ == "__main__":
	main()
