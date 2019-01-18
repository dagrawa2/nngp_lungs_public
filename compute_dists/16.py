import time
import numpy as np
import h5py

def innerprods_to_dists(Q, Q_X=None, Q_X2=None):
	if Q_X is None:
		D = np.diagonal(Q).reshape((-1, 1))
		return D + D.T - 2*Q
	else:
		Q_X = Q_X.reshape((-1, 1))
		Q_X2 = Q_X2.reshape((-1, 1))
		return Q_X + Q_X2.T - 2*Q


time_file = open("/raid/scratch/devanshu/dist_times_16.txt", "w")

#skip = 100
print("loading data")
with h5py.File("/raid/scratch/devanshu/innerprods_16.h5", "r") as f:
	X_train_train = np.array(f["/train_train"]) # [::skip,::skip])
	X_train_test = np.array(f["/train_test"]) # [::skip,::skip])
	X_test_test = np.array(f["/test_test"]) # [::skip,::skip])

	with h5py.File("/raid/scratch/devanshu/dists_16.h5", "w") as f:
		print("Computing train_train")
		time_0 = time.time()
		tt = innerprods_to_dists(X_train_train)
		f.create_dataset("/train_train", data=tt)
		time_file.write("train_train: "+str(np.round(time.time()-time_0, 2))+" s\n")

		print("Computing train_test")
		time_0 = time.time()
		tt = innerprods_to_dists(X_train_test, np.diagonal(X_train_train), np.diagonal(X_test_test))
		f.create_dataset("/train_test", data=tt)
		time_file.write("train_test: "+str(np.round(time.time()-time_0, 2))+" s\n")

		print("Computing test_test")
		time_0 = time.time()
		tt = innerprods_to_dists(X_test_test)
		f.create_dataset("/test_test", data=tt)
		time_file.write("test_test: "+str(np.round(time.time()-time_0, 2))+" s\n")

print("Done")
