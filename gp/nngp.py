import gc
import h5py
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from my_module import models

parser = argparse.ArgumentParser(description="Predict with NNGP.")
parser.add_argument("--depth", dest="depth", type=int, default=0, help="Number of hidden layers in the NNGP.")
args = parser.parse_args()
depth = args.depth


np.random.seed(123)

time_0 = time.time()

stride = 1
print("Loading data")
with h5py.File('/raid/scratch/devanshu/innerprods_1024.h5', 'r') as f:
	Q_train_train = np.array(f["/train_train"][::stride,::stride])
	Q_train_test = np.array(f["/train_test"][::stride,::stride])
	Q_test_test = np.array(f["/test_test"][::stride,::stride])

with h5py.File("/raid/ChestXRay14/chestxray14_1024.h5", "r") as hf:
	Y_train = 2*np.array(hf["labels/train"][::stride]).astype(np.int32) - 1

print("Rescaling . . . ")
with open("../scale.json", "r") as fp:
	scale = json.load(fp)["scale"]
Q_train_train /= scale**2
Q_train_test /= scale**2
Q_test_test /= scale**2

batch_size = 20000
print("Loading optimized hyperparameters")
n_tasks = Y_train.shape[1]
model = models.NNGP(Q_train_train, Q_train_test, Q_test_test, Y_train, batch_size=batch_size)
del Q_train_train; gc.collect()
del Q_train_test; gc.collect()
del Q_test_test; gc.collect()
del Y_train; gc.collect()
with open("results/nngp_ho_depth"+str(depth)+".json", "r") as fp:
	hist = json.load(fp)["hist"]
model.v_n = hist["v_n"][-1]
model.v_b = hist["v_b"][-1]
model.v_w = hist["v_w"][-1]
model.depth = depth

print("Predicting")
time_1 = time.time()
Y_mean, Y_var = model.predict_y_memeff()
time_2 = time.time()

print("Saving results")
with h5py.File("results/nngp_pred_depth"+str(depth)+".h5", "w") as f:
	f.create_dataset("/mean", data=Y_mean)
	f.create_dataset("/var", data=Y_var)
info = {"stride": stride,
	"pred_time": time_2-time_1,
	"script_time": time.time()-time_0,
	"batch_size": batch_size,
	"depth": depth,
	"params": {"v_n": model.v_n,
		"v_b": model.v_b,
		"v_w": model.v_w
	}
}
with open("results/nngp_depth"+str(depth)+".json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")