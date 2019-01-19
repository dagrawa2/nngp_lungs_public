import gc
import numpy as np
import scipy as sp
from scipy import optimize as sp_opt
from . import utils

def make_batch_indices(n_points, batch_size):
	if batch_size is None:
		return [0, None]
	n_batches = n_points//batch_size
	if n_points%batch_size > 0:
		n_batches += 1
	batch_indices = [i*batch_size for i in range(n_batches)] + [None]
	return batch_indices


def batchit_gp(f):
	def wrapper(*args, **kwargs):
		self = args[0]
		if self.batch_size is None:
			return f(*args, **kwargs)
		if f.__name__ == "K":
			Q = args[1]

			K = []
			batch_indices = make_batch_indices(Q.shape[0], self.batch_size)
			n_batches = (len(batch_indices)-1)**2
			print("---")
			print("Calling method K")
			print("Number of batches: ", n_batches)
			batch_counter = 0
			for i, j in zip(batch_indices[:-1], batch_indices[1:]):
				blocks = []
				for i2, j2 in zip(batch_indices[:-1], batch_indices[1:]):
					batch_counter += 1
					print("Batch ", batch_counter, "/", n_batches, " . . . ")
					blocks.append(f(self, Q[i:j,i2:j2]))
				K.append(blocks)
			K = np.block(K)
			print("---")
			return K
		if f.__name__ == "Kdiag":
			Q = args[1]
			K = []
			batch_indices = make_batch_indices(Q_X.shape[0], self.batch_size)
			n_batches = len(batch_indices)-1
			print("---")
			print("Calling method Kdiag")
			print("Number of batches: ", n_batches)
			batch_counter = 0
			for i, j in zip(batch_indices[:-1], batch_indices[1:]):
				batch_counter += 1
				print("Batch ", batch_counter, "/", n_batches, " . . . ")
				K.append(f(self, Q[i:j,i:j]))
			K = np.block(K)
		print("---")
		return K
	return wrapper


def batchit_nngp(f):
	def wrapper(*args, **kwargs):
		self = args[0]
		if self.batch_size is None:
			return f(*args, **kwargs)
		if f.__name__ == "K":
			Q = args[1]
			Q_X = args[2] if len(args) == 4 else None
			Q_X2 = args[3] if len(args) == 4 else None
			if Q_X is not None:
				K = []
				batch_indices = make_batch_indices(Q.shape[0], self.batch_size)
				batch_indices2 = make_batch_indices(Q.shape[1], self.batch_size)
				n_batches = (len(batch_indices)-1)*(len(batch_indices2)-1)
				print("---")
				print("Calling method K with Q_X and Q_X2 not None")
				print("Number of batches: ", n_batches)
				batch_counter = 0
				for i, j in zip(batch_indices[:-1], batch_indices[1:]):
					blocks = []
					for i2, j2 in zip(batch_indices2[:-1], batch_indices2[1:]):
						batch_counter += 1
						print("Batch ", batch_counter, "/", n_batches, " . . . ")
						blocks.append(f(self, Q[i:j,i2:j2], Q_X[i:j], Q_X2[i2:j2]))
					K.append(blocks)
				K = np.block(K)
				print("---")
				return K
			if Q_X is None:
				K = []
				batch_indices = make_batch_indices(Q.shape[0], self.batch_size)
				n_batches = (len(batch_indices)-1)**2
				print("---")
				print("Calling method K with Q_X and Q_X2 None")
				print("Number of batches: ", n_batches)
				batch_counter = 0
				for i, j in zip(batch_indices[:-1], batch_indices[1:]):
					blocks = []
					for i2, j2 in zip(batch_indices[:-1], batch_indices[1:]):
						batch_counter += 1
						print("Batch ", batch_counter, "/", n_batches, " . . . ")
						if i != i2 or j != j2:
							blocks.append(f(self, Q[i:j,i2:j2], np.diagonal(Q)[i:j], np.diagonal(Q)[i2:j2]))
						else:
							blocks.append(f(self, Q[i:j,i:j]))
					K.append(blocks)
				K = np.block(K)
				print("---")
				return K
		if f.__name__ == "Kdiag":
			Q_X = args[1]
			K = []
			batch_indices = make_batch_indices(Q_X.shape[0], self.batch_size)
			n_batches = len(batch_indices)-1
			print("---")
			print("Calling method Kdiag")
			print("Number of batches: ", n_batches)
			batch_counter = 0
			for i, j in zip(batch_indices[:-1], batch_indices[1:]):
				batch_counter += 1
				print("Batch ", batch_counter, "/", n_batches, " . . . ")
				K.append(f(self, Q_X[i:j]))
			K = np.block(K)
		print("---")
		return K
	return wrapper


class RBFGP(object):

	def __init__(self, A, B, C, Y, c=1.0, l=1.0, v_n = 0.01, batch_size=None):
		self.A = A
		self.B = B
		self.C = C
		self.Y = Y
		self.c = c
		self.l = l
		self.v_n = v_n
		self.batch_size = batch_size

	@batchit_gp
	def K(self, Q):
		return self.c*np.exp(-1/2*Q/self.l**2)

	@batchit_gp
	def Kdiag(self, Q):
		return self.c*np.ones(Q.shape[0])

	def predict_y(self, full_cov=False):
		Kx = self.K(self.B)
		K = self.K(self.A) + np.eye(self.A.shape[0]) * self.v_n
		L = np.linalg.cholesky(K)
		A = sp.linalg.solve_triangular(L, Kx, lower=True)
		V = sp.linalg.solve_triangular(L, self.Y, lower=True)
		fmean = A.T.dot(V)
		if full_cov:
			fvar = self.K(self.C) - A.T.dot(A)
#			fvar = np.tile(np.expand_dims(fvar, 2), (1, 1, self.Y.shape[1]))
			return fmean, fvar + np.eye(self.C.shape[0])*self.v_n
		else:
			fvar = self.Kdiag(self.C) - np.sum(A**2, axis=0)
#			fvar = np.tile(fvar.reshape((-1, 1)), (1, self.Y.shape[1]))
		return fmean, fvar + self.v_n

	def NLE(self):
		N = self.A.shape[0]
		M = self.Y.shape[1]
		K = self.K(self.A) + self.v_n*np.eye(N)
		L = np.linalg.cholesky(K)
		A = sp.linalg.solve_triangular(L, self.Y, lower=True)
		nle = 1/2*np.sum(A**2) + M*np.sum(np.log(np.diagonal(L))) + M*N/2*np.log(2*np.pi)
		return nle

	def NLE_gradient(self):
		N = self.A.shape[0]
		M = self.Y.shape[1]
		DK_c = np.exp(-1/2*self.A/self.l**2)
		K = self.c*DK_c
		DK_l = K*self.A/self.l**3
		K = K + self.v_n*np.eye(N)
		K_inv = utils.PD_inverse(K)
		W = K_inv.dot(self.Y)
		NLE = 1/2*np.sum(self.Y*W) + M/2*list(np.linalg.slogdet(K))[1] + M*N/2*np.log(2*np.pi)
		DNLE = -1/2*W.dot(W.T) + M/2*K_inv
		D_v_n = np.trace(DNLE) *(1.0-np.exp(-self.v_n))
		D_c = np.sum(DNLE*DK_c) *(1.0-np.exp(-self.c))
		D_l = np.sum(DNLE*DK_l) *(1.0-np.exp(-self.l))
		self.NLE_value = np.sum(NLE)
		self.DNLE_value = np.array([D_v_n, D_c, D_l])
		return self.DNLE_value

	def optimize(self, max_iters=100, print_cb=False):
		self.NLE_gradient()
		x0 = np.array([self.v_n, self.c, self.l])
		x0 = np.log(np.exp(x0)-1.0)
		history = {"v_n":[self.v_n], "c":[self.c], "l":[self.l], "nle_train":[self.NLE_value]}
		def f(x):
			x = np.log(1.0+np.exp(x))
#			print("f:", np.round(x, 4))
			self.v_n = x[0]
			self.c = x[1]
			self.l = x[2]
			self.NLE_gradient()
			return self.NLE_value
		def df(x):
			x = np.log(1.0+np.exp(x))
#			print("df:", np.round(x, 4))
			return self.DNLE_value
		def cb(x):
			history["v_n"].append(self.v_n)
			history["c"].append(self.c)
			history["l"].append(self.l)
			history["nle_train"].append(self.NLE_value/self.A.shape[0])
			if print_cb: print("["+str(len(history["v_n"])-1)+"] "+str(np.round(self.NLE_value/self.A.shape[0], 5)))
		if print_cb: print("\nNLE loss per training point\n---")
		res = sp_opt.minimize(f, x0, jac=df, callback=cb, options={"maxiter":max_iters})
		if print_cb: print("---\n")
		x = np.log(1.0+np.exp(res.x))
		self.v_n = x[0]
		self.c = x[1]
		self.l = [2]
		return history


class NNGP(object):

	def __init__(self, A, B, C, Y, v_w=1.0, v_b=1.0, v_n = 0.01, depth=1, batch_size=None):
		self.A = A
		self.B = B
		self.C = C
		self.Y = Y
		self.v_b = v_b
		self.v_w = v_w
		self.v_n = v_n
		self.depth = depth
		self.batch_size = batch_size

	@batchit_nngp
	def K(self, K, K_X=None, K_X2=None):
		if K_X is None:
			K = self.v_b + self.v_w*K
			for d in range(self.depth):
				N = np.sqrt(np.outer(np.diagonal(K), np.diagonal(K)))
				R = np.maximum(np.minimum(K/N, 1.0), -1.0)
				K = self.v_b + self.v_w*N/(2*np.pi)*(np.sqrt(1-R**2) + (np.pi-np.arccos(R))*R)
			return K
		else:
			K = self.v_b + self.v_w*K
			K_X = self.v_b + self.v_w*K_X
			K_X2 = self.v_b + self.v_w*K_X2
			for d in range(self.depth):
				N = np.sqrt(np.outer(K_X, K_X2))
				R = np.maximum(np.minimum(K/N, 1.0), -1.0)
				K = self.v_b + self.v_w*N/(2*np.pi)*(np.sqrt(1-R**2) + (np.pi-np.arccos(R))*R)
				K_X = self.v_b + self.v_w*K_X/2.0
				K_X2 = self.v_b + self.v_w*K_X2/2.0
			return K

	@batchit_nngp
	def Kdiag(self, K_X):
		K_X = self.v_b + self.v_w*K_X
		for d in range(self.depth):
			K_X = self.v_b + self.v_w*K_X/2.0
		return K_X

	def predict_y(self, full_cov=False):
		Kx = self.K(self.B, np.diagonal(self.A), np.diagonal(self.C))
		K = self.K(self.A) + np.eye(self.A.shape[0]) * self.v_n
		L = np.linalg.cholesky(K)
		A = sp.linalg.solve_triangular(L, Kx, lower=True)
		V = sp.linalg.solve_triangular(L, self.Y, lower=True)
		fmean = A.T.dot(V)
		if full_cov:
			fvar = self.K(self.C) - A.T.dot(A)
#			fvar = np.tile(np.expand_dims(fvar, 2), (1, 1, self.Y.shape[1]))
			return fmean, fvar + np.eye(self.C.shape[0])*self.v_n
		else:
			fvar = self.Kdiag(np.diagonal(self.C)) - np.sum(A**2, axis=0)
#			fvar = np.tile(fvar.reshape((-1, 1)), (1, self.Y.shape[1]))
		return fmean, fvar + self.v_n

	def predict_y_memeff(self):
		A_diag = np.diagonal(self.A)
		C_diag = np.diagonal(self.C)
		del self.C; gc.collect()
		K = self.K(self.A) + np.eye(self.A.shape[0]) * self.v_n
		del self.A; gc.collect()
		Kx = self.K(self.B, A_diag, C_diag)
		del self.B; gc.collect()
		del A_diag; gc.collect()
		L = np.linalg.cholesky(K)
		A = sp.linalg.solve_triangular(L, Kx, lower=True)
		V = sp.linalg.solve_triangular(L, self.Y, lower=True)
		fmean = A.T.dot(V)
		fvar = self.Kdiag(C_diag) - np.sum(A**2, axis=0)
		return fmean, fvar + self.v_n

	def NLE(self):
		N = self.A.shape[0]
		M = self.Y.shape[1]
		K = self.K(self.A) + self.v_n*np.eye(N)
		L = np.linalg.cholesky(K)
		A = sp.linalg.solve_triangular(L, self.Y, lower=True)
		nle = 1/2*np.sum(A**2) + M*np.sum(np.log(np.diagonal(L))) + M*N/2*np.log(2*np.pi)
		return nle

	def NLE_gradient(self):
		N = self.A.shape[0]
		M = self.Y.shape[1]
		D_v_b = np.ones((N, N))
		D_v_w = self.A
		K = self.v_b + self.v_w*D_v_w
		for _ in range(self.depth):
			K_diag = np.diagonal(K)
			A = np.sqrt(np.outer(K_diag, K_diag))
			R = np.maximum(np.minimum(K/A, 1.0), -1.0)
			theta = np.arccos(R)
			F = A/(2*np.pi)*(np.sqrt(1-R**2) + (np.pi-theta)*R)
			DA = utils.np_sym(np.outer(K_diag, np.diagonal(D_v_b)))/A
			D_v_b = 1 + self.v_w*F/A*DA + self.v_w/(2*np.pi)*(np.pi-theta)*(D_v_b - R*DA)
			DA = utils.np_sym(np.outer(K_diag, np.diagonal(D_v_w)))/A
			D_v_w = F + self.v_w*F/A*DA + self.v_w/(2*np.pi)*(np.pi-theta)*(D_v_w - R*DA)
			K = self.v_b + self.v_w*F
		K = K + self.v_n*np.eye(N)
		K_inv = utils.PD_inverse(K)
		W = K_inv.dot(self.Y)
		NLE = 1/2*np.sum(self.Y*W) + M/2*list(np.linalg.slogdet(K))[1] + M*N/2*np.log(2*np.pi)
		DNLE = -1/2*W.dot(W.T) + M/2*K_inv
		D_v_n = np.trace(DNLE)
		D_v_b = np.sum(DNLE*D_v_b)
		D_v_w = np.sum(DNLE*D_v_w)
		self.NLE_value = np.sum(NLE)
		self.DNLE_value = np.array([D_v_n, D_v_b, D_v_w])
		return self.DNLE_value

	def optimize(self, max_iters=100, print_cb=False):
		x0 = np.array([self.v_n, self.v_b, self.v_w])
		bounds = [(0, None) for i in range(3)]
		history = {"v_n":[self.v_n], "v_b":[self.v_b], "v_w":[self.v_w], "nle_train":[self.NLE()/self.A.shape[0]]}
		def f(x):
			print("f:", np.round(x, 4))
			self.v_n = x[0]
			self.v_b = x[1]
			self.v_w = x[2]
			self.NLE_gradient()
			return self.NLE_value
		def df(x):
			print("df:", np.round(x, 4))
			return self.DNLE_value
		def cb(x):
			history["v_n"].append(self.v_n)
			history["v_b"].append(self.v_b)
			history["v_w"].append(self.v_w)
			history["nle_train"].append(self.NLE_value/self.A.shape[0])
			if print_cb: print("["+str(len(history["v_n"])-1)+"] "+str(np.round(self.NLE_value/self.A.shape[0], 5)))
		if print_cb: print("\nNLE loss per training point\n---\n[0] "+str(np.round(history["nle_train"][0], 5)))
		res = sp_opt.minimize(f, x0, jac=df, bounds=bounds, callback=cb, options={"maxiter":max_iters})
		if print_cb: print("---\n")
		self.v_n = res.x[0]
		self.v_b = res.x[1]
		self.v_w = res.x[2]
		return history


class NNGP2(object):

	def __init__(self, A, B, C, Y, v_n=0.01, v_b=0.01, v_w=1.0, depth=1, batch_size=None):
		self.A = A
		self.B = B
		self.C = C
		self.Y = Y
		self.v_n = v_n
		self.v_b = np.array([v_b for i in range(depth+1)])
		self.v_w = np.array([v_w for i in range(depth+1)])
		self.depth = depth
		self.batch_size = batch_size

	@batchit_nngp
	def K(self, K, K_X=None, K_X2=None):
		if K_X is None:
			K = self.v_b[0] + self.v_w[0]*K
			for v_b,v_w in zip(self.v_b[1:], self.v_w[1:]):
				N = np.sqrt(np.outer(np.diagonal(K), np.diagonal(K)))
				R = np.maximum(np.minimum(K/N, 1.0), -1.0)
				K = v_b + v_w*N/(2*np.pi)*(np.sqrt(1-R**2) + (np.pi-np.arccos(R))*R)
			return K
		else:
			v_b, v_w = self.v_b[0], self.v_w[0]
			K = v_b + v_w*K
			K_X = v_b + v_w*K_X
			K_X2 = v_b + v_w*K_X2
			for v_b,v_w in zip(self.v_b[1:], self.v_w[1:]):
				N = np.sqrt(np.outer(K_X, K_X2))
				R = np.maximum(np.minimum(K/N, 1.0), -1.0)
				K = v_b + v_w*N/(2*np.pi)*(np.sqrt(1-R**2) + (np.pi-np.arccos(R))*R)
				K_X = v_b + v_w*K_X/2.0
				K_X2 = v_b + v_w*K_X2/2.0
			return K

	@batchit_nngp
	def Kdiag(self, K_X):
		K_X = self.v_b[0] + self.v_w[0]*K_X
		for v_b,v_w in zip(self.v_b[1:], self.v_w[1:]):
			K_X = v_b + v_w*K_X/2.0
		return K_X

	def predict_y(self, full_cov=False):
		Kx = self.K(self.B, np.diagonal(self.A), np.diagonal(self.C))
		K = self.K(self.A) + np.eye(self.A.shape[0]) * self.v_n
		L = np.linalg.cholesky(K)
		A = sp.linalg.solve_triangular(L, Kx, lower=True)
		V = sp.linalg.solve_triangular(L, self.Y, lower=True)
		fmean = A.T.dot(V)
		if full_cov:
			fvar = self.K(self.C) - A.T.dot(A)
#			fvar = np.tile(np.expand_dims(fvar, 2), (1, 1, self.Y.shape[1]))
			return fmean, fvar + np.eye(self.C.shape[0])*self.v_n
		else:
			fvar = self.Kdiag(np.diagonal(self.C)) - np.sum(A**2, axis=0)
#			fvar = np.tile(fvar.reshape((-1, 1)), (1, self.Y.shape[1]))
		return fmean, fvar + self.v_n

	def predict_y_memeff(self):
		A_diag = np.diagonal(self.A)
		C_diag = np.diagonal(self.C)
		del self.C; gc.collect()
		K = self.K(self.A) + np.eye(self.A.shape[0]) * self.v_n
		del self.A; gc.collect()
		Kx = self.K(self.B, A_diag, C_diag)
		del self.B; gc.collect()
		del A_diag; gc.collect()
		L = np.linalg.cholesky(K)
		A = sp.linalg.solve_triangular(L, Kx, lower=True)
		V = sp.linalg.solve_triangular(L, self.Y, lower=True)
		fmean = A.T.dot(V)
		fvar = self.Kdiag(C_diag) - np.sum(A**2, axis=0)
		return fmean, fvar + self.v_n

	def NLE(self):
		N = self.A.shape[0]
		M = self.Y.shape[1]
		K = self.K(self.A) + self.v_n*np.eye(N)
		L = np.linalg.cholesky(K)
		A = sp.linalg.solve_triangular(L, self.Y, lower=True)
		nle = 1/2*np.sum(A**2) + M*np.sum(np.log(np.diagonal(L))) + M*N/2*np.log(2*np.pi)
		return nle

	def NLE_gradient(self):
		N = self.A.shape[0]
		M = self.Y.shape[1]
		D_v_b = [np.ones((N, N))]
		D_v_w = [self.A]
		K = self.v_b[0] + self.v_w[0]*D_v_w[0]
		for v_b,v_w in zip(self.v_b[1:], self.v_w[1:]):
			K_diag = np.diagonal(K)
			A = np.sqrt(np.outer(K_diag, K_diag))
			R = np.maximum(np.minimum(K/A, 1.0), -1.0)
			theta = np.arccos(R)
			F = A/(2*np.pi)*(np.sqrt(1-R**2) + (np.pi-theta)*R)
			for i in range(len(D_v_b)):
				DA = utils.np_sym(np.outer(K_diag, np.diagonal(D_v_b[i])))/A
				D_v_b[i] = v_w*F/A*DA + v_w/(2*np.pi)*(np.pi-theta)*(D_v_b[i] - R*DA)
				DA = utils.np_sym(np.outer(K_diag, np.diagonal(D_v_w[i])))/A
				D_v_w[i] = v_w*F/A*DA + v_w/(2*np.pi)*(np.pi-theta)*(D_v_w[i] - R*DA)
			D_v_b.append(np.ones((N, N)))
			D_v_w.append(F)
			K = v_b + v_w*F
		K = K + self.v_n*np.eye(N)
		K_inv = utils.PD_inverse(K)
		W = K_inv.dot(self.Y)
		NLE = 1/2*np.sum(self.Y*W) + M/2*list(np.linalg.slogdet(K))[1] + M*N/2*np.log(2*np.pi)
		DNLE = -1/2*W.dot(W.T) + M/2*K_inv
		D_v_n = np.array([np.trace(DNLE)])
		D_v_b = np.array([np.sum(DNLE*dvb) for dvb in D_v_b])
		D_v_w = np.array([np.sum(DNLE*dvw) for dvw in D_v_w])
		self.NLE_value = np.sum(NLE)
		self.DNLE_value = np.concatenate((D_v_n, D_v_b, D_v_w), axis=0)
		return self.DNLE_value

	def optimize(self, max_iters=100, print_cb=False):
		x0 = np.array([self.v_n]+list(self.v_b)+list(self.v_w))
		bounds = [(0, None) for i in range(len(x0))]
		history = {"v_n":[self.v_n], "v_b":[list(self.v_b)], "v_w":[list(self.v_w)], "nle_train":[self.NLE()/self.A.shape[0]]}
		def f(x):
#			print("f:", np.round(x, 4))
			self.v_n = x[0]
			self.v_b = x[1:self.depth+2]
			self.v_w = x[self.depth+2:]
			self.NLE_gradient()
			return self.NLE_value
		def df(x):
#			print("df:", np.round(x, 4))
			return self.DNLE_value
		def cb(x):
			history["v_n"].append(self.v_n)
			history["v_b"].append(list(self.v_b))
			history["v_w"].append(list(self.v_w))
			history["nle_train"].append(self.NLE_value/self.A.shape[0])
			if print_cb: print("["+str(len(history["v_n"])-1)+"] "+str(np.round(self.NLE_value/self.A.shape[0], 5)))
		if print_cb: print("\nNLE loss per training point\n---\n[0] "+str(np.round(history["nle_train"][0], 5)))
		res = sp_opt.minimize(f, x0, jac=df, bounds=bounds, callback=cb, options={"maxiter":max_iters})
		if print_cb: print("---\n")
		self.v_n = res.x[0]
		self.v_b = res.x[1:self.depth+2]
		self.v_w = res.x[self.depth+2:]
		return history
