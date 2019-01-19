import h5py
import json
import time
import numpy as np
import pandas as pd

np.random.seed(123)
time_0 = time.time()

stride = 1
print("Loading data")
with h5py.File('/raid/scratch/devanshu/innerprods_1024.h5', 'r') as f:
	K_train_train = np.array(f["/train_train"][::stride,::stride])

print("Rescaling . . . ")
with open("../scale.json", "r") as fp:
	scale = json.load(fp)["scale"]
K_train_train /= scale**2

print("Computing eigenvalues")
eigs = np.linalg.eigvalsh(K_train_train)[::-1]

print("Computing ratios")
ratios = np.cumsum(eigs)/np.sum(eigs)
results = {"k": range(1, len(eigs)+1), "eig": eigs, "ratio": ratios}
results = pd.DataFrame.from_dict(results)[["k", "eig", "ratio"]]

print("Saving results")
results.to_csv("results/ratios.csv", index=False)

print("Done!")