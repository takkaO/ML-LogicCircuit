import numpy as np

class TweLayer:
	def __init__(self, debug = False):
		self.debug = debug
		input_num = 2
		hidden_num = 3
		output_num = 2
		self.params = {}
		self.params['W1'] = np.random.randn(input_num, hidden_num)
		self.params['b1'] = np.zeros(hidden_num)
		self.params['W2'] = np.random.randn(hidden_num, output_num)
		self.params['b2'] = np.zeros(output_num)

		if self.debug:
			for key, value in self.params.items():
				print(key)
				print(value, end = "\n\n")

	def ReLU(self, x):
		return np.maximum(0, x)
	
	def predict(self, numin):
		a1 = np.dot(numin, self.params['W1']) + self.params['b1']
		z1 = self.ReLU(a1)

		a2 = np.dot(z1, self.params['W2']) + self.params['b2']
		y = softmax(a2)
		if self.debug:
			print("y = ", end = "")
			print(y)
		return y
	
	def crossEntropy(self, result, teacher):
		delta = 1e-7	# logが発散しないために微小値を入れておく
		if result.ndim == 1:
			result = result.reshape(1, result.size)
			teacher = teacher.reshape(1, teacher.size)
		return -np.sum(teacher * np.log(result + delta)) / result.shape[0]

	def loss(self, x, teacher):
		y = self.predict(x)
		return self.crossEntropy(y, teacher)
	
	def calcGrads(self, x, teacher):
		loss = lambda W: self.loss(x, teacher)

		grads = {}
		grads['W1'] = numerical_gradient(loss, self.params['W1'])
		grads['b1'] = numerical_gradient(loss, self.params['b1'])
		grads['W2'] = numerical_gradient(loss, self.params['W2'])
		grads['b2'] = numerical_gradient(loss, self.params['b2'])

		if self.debug:
			for key, value in grads.items():
				print(key)
				print(value, end="\n\n")
		return grads
	
	def train(self, x, t, iters):
		
		for i in range(iters):
			grad = self.calcGrads(x, t)

			for key in ('W1', 'b1', 'W2', 'b2'):
				self.params[key] -= 0.01 * grad[key]
			
			if i % 1000 == 0:
				print("Loss: ", self.loss(x, t))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def numerical_gradient(func, x):
	h = 1e-4
	grad = np.zeros_like(x)

	# ループ用にイテレータを作成
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = func(x)  # f(x+h)

		x[idx] = tmp_val - h
		fxh2 = func(x)  # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)

		x[idx] = tmp_val  # 値を元に戻す
		it.iternext()

	return grad

def main():
	print("Train start!")
	x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
	t = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]) 	# XOR: non liner
	#t = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # AND: liner
	#t = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])  # NAND: liner
	tl = TweLayer()
	tl.train(x, t, 10000)
	print("Train finish\n")
	print("x1 x2")
	for a in x:
		print(a, " => ", end="")
		print(np.argmax(tl.predict(a)))

if __name__ == "__main__":
	main()
