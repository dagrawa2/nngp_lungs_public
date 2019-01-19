import numpy as np
import scipy as sp

# numpy functions

def PD_inverse(K):
	L = np.linalg.cholesky(K)
	A = sp.linalg.solve_triangular(L, np.eye(K.shape[0]), lower=True)
	A = sp.linalg.solve_triangular(L.T, A, lower=False)
	return A

def np_sym(A):
	return (A + A.T)/2

def np_combine(a, b):
	return np.concatenate((a, b), axis=0)

def outer_add(A, B):
	dim_A = A.ndim
	dim_B = B.ndim
	A = A.reshape(np_combine(A.shape, np.ones((dim_B))).astype(np.int32))
	B = B.reshape(np_combine(np.ones((dim_A)), B.shape).astype(np.int32))
	return A+B


# metrics

def MSE(Y_test, Y_pred_mean, Y_pred_var=None):
	if Y_pred_var is None:
		return np.mean(np.sum((Y_pred_mean-Y_test)**2, axis=1))
	return np.mean(np.sum((Y_pred_mean-Y_test)**2, axis=1) + np.sum(Y_pred_var, axis=1))

def accuracy(Y_test, Y_pred_mean):
	act = np.argmax(Y_test, axis=1)
	pred = np.argmax(Y_pred_mean, axis=1)
	sum = 0
	for i in range(len(act)):
		if act[i] == pred[i]: sum += 1
	return sum/len(act)

def binary_accuracy(Y_test, Y_pred_mean):
	act = np.sign(Y_test)
	pred = np.sign(Y_pred_mean)
	sum = 0
	for i in range(len(act)):
		if act[i] == pred[i]: sum += 1
	return sum/len(act)


# miscellaneous

def flatten(L):
	return [item for sublist in L for item in sublist]
