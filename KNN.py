import numpy as np

class KNN:
	def __init__(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def distance(self, p1, p2):
		return np.sum(np.square(p1-p2), axis = -1)

	def predict(self, x_test, k=5, show_progress = False):
		res = []
		for x in x_test:
			dis = self.distance(self.x_train, x)
			indice = np.argpartition(dis, k)[:k]
			(v, c) = np.unique(self.y_train[indice], return_counts = True)
			predicted_label = v[np.argmax(c)]
			res.append(int(predicted_label))
			if show_progress:
				print(predicted_label)
		return res

	def eval(self, x_test, y_test, k = 5, show_progress = False):
		pred = self.predict(x_test, k = k, show_progress = show_progress)
		acc = np.sum(np.array(pred) == np.array(y_test)) / len(y_test)
		return acc

