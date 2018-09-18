from KNN import KNN
import numpy as np
import data


if __name__ == "__main__":
	print("start to load data.")
	x, y = data.load_data_mnist("train.csv")
	x_train, y_train, x_test, y_test, x_val, y_val = data.prepareData(x, y)
	print("data loading finished.")
	model = KNN(x_train, y_train)
	kk, temp_acc = 0, 0
	for k in range(1,50):
		acc = model.eval(x_val, y_val, k = k, show_progress = True)
		if acc>temp_acc:
			kk = k
			temp_acc = acc
	print("validation accuracy: ", temp_acc, " k =", kk)
	acc = model.eval(x_test, y_test, k = kk)
	print("test accuracy: ", acc)