import h5py
import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA


np.random.seed(123)
time_0 = time.time()

stride = 1
print("Loading data")
with h5py.File('/raid/scratch/devanshu/innerprods_1024.h5', 'r') as f:
	K_train_train = np.array(f["/train_train"][::stride,::stride])
	K_train_test = np.array(f["/train_test"][::stride,::stride])

print("Rescaling . . . ")
with open("../scale.json", "r") as fp:
	scale = json.load(fp)["scale"]
K_train_train /= scale**2
K_train_test /= scale**2

n_components = 311  # explains 95% of variance
print("Fitting PCA model")
time_1 = time.time()
model = KernelPCA(n_components=n_components, kernel="precomputed", n_jobs=-1)
Z_train = model.fit_transform(K_train_train)

print("Transforming test set")
time_2 = time.time()
Z_test = model.transform(K_train_test.T)
time_3 = time.time()

print("Saving results")
with h5py.File("results/pca.h5", "w") as f:
	f.create_dataset("/train", data=Z_train)
	f.create_dataset("/test", data=Z_test)

info = {"stride": stride,
	"train_time": time_2-time_1,
	"test_time": time_3-time_2,
	"script_time": time.time()-time_0,
	"n_components": n_components
}
with open("results/pca_info.json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")