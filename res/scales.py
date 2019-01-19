import h5py
import json
import time
import numpy as np

time_0 = time.time()

res_list = [16, 32, 64, 128, 256, 512, 1024]
scales = {}

stride = 1
for res in res_list:
	print("Resolution:", res)
	print("Loading data")
	with h5py.File('/raid/scratch/devanshu/innerprods_'+str(res)+'.h5', 'r') as f:
		Q_train_train = np.array(f["/train_train"][::stride])

	print("Computing scale . . . ")
	scale = np.sqrt(np.trace(Q_train_train)/Q_train_train.shape[0])
	scales.update({str(res): scale})

print("Saving scales . . . ")
scales.update({"time": time.time()-time_0})
with open("results/scales.json", "w") as fp:
	json.dump(scales, fp, indent=2)

print("Done!")