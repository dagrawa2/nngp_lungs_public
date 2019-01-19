import h5py
import json
import time
import numpy as np

time_0 = time.time()

stride = 1
print("Loading data")
with h5py.File('/raid/scratch/devanshu/innerprods_1024.h5', 'r') as f:
	Q_train_train = np.array(f["/train_train"][::stride])

print("Computing scale . . . ")
scale = np.sqrt(np.trace(Q_train_train)/Q_train_train.shape[0])

print("Saving scale . . . ")
results = {"scale": scale, "time": time.time()-time_0}
with open("scale.json", "w") as fp:
	json.dump(results, fp, indent=2)

print("Done!")