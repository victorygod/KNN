import numpy as np

def load_data(filename):
	x, y = [], []
	with open(filename, "r") as f:
		lines = f.readlines()
		for line in lines[1:]:
			a = line.split(',')
			x.append([float(aa) for aa in a[:-1]])
			y.append(float(a[-1]))
	return np.array(x), np.array(y)

def prepareData(filename):
	x, y = load_data(filename)
	l = len(y)
	indices = np.random.permutation(l)	
	train_size = int(0.8*l)
	val_size = int(0.1*l)
	x_train, y_train = x[indices[:train_size]], y[indices[:train_size]]
	x_val, y_val = x[indices[train_size:train_size+val_size]], y[indices[train_size:train_size+val_size]]
	x_test, y_test = x[indices[train_size+val_size:]], y[indices[train_size+val_size:]]
	return x_train, y_train, x_test, y_test, x_val, y_val