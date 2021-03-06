import gc
import h5py
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from my_module import models

np.random.seed(123)

time_0 = time.time()

stride = 5
print("Loading data")
with h5py.File('/raid/scratch/devanshu/dists_1024.h5', 'r') as f:
	Q_train_train = np.array(f["/train_train"][::stride,::stride])
	Q_train_test = None # np.array(f["/train_test"][::stride,:2])
	Q_test_test = None # np.array(f["/test_test"][:2,:2])

with h5py.File("/raid/ChestXRay14/chestxray14_1024.h5", "r") as hf:
	Y_train = 2*np.array(hf["labels/train"][::stride]).astype(np.int32) - 1

print("Rescaling . . . ")
with open("../scale.json", "r") as fp:
	scale = json.load(fp)["scale"]
Q_train_train /= scale**2
#Q_train_test /= scale**2
#Q_test_test /= scale**2

max_iters = 50
print("Training")
n_tasks = Y_train.shape[1]
model = models.RBFGP(Q_train_train, Q_train_test, Q_test_test, Y_train, c=1.0, l=1.0, v_n = 0.01, batch_size=None)
del Q_train_train; gc.collect()
del Q_train_test; gc.collect()
del Q_test_test; gc.collect()
del Y_train; gc.collect()
time_1 = time.time()
history = model.optimize(max_iters=max_iters, print_cb=True)
time_2 = time.time()

print("Saving results")
info = {"stride": stride,
	"train_time": time_2-time_1,
	"script_time": time.time()-time_0,
"max_iters": max_iters,
"hist": history
}
with open("results/rbf_ho.json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")