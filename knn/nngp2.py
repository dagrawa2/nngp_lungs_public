import gc
import h5py
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from my_module import models

np.random.seed(123)

def KNN(kernel, Y, K=1):
	inds = np.argpartition(kernel, K, axis=0)[:K]
	vals = np.stack([kernel[inds[:,i],i] for i in range(kernel.shape[1])], axis=1)
	val_inds_sorted = np.argsort(vals, axis=0)
	inds = np.stack([inds[val_inds_sorted[:,i],i] for i in range(kernel.shape[1])], axis=1)
	return np.cumsum(Y[inds], axis=0)/(np.arange(1, K+1).reshape((-1, 1, 1)))

time_0 = time.time()

stride = 1
print("Loading data")
with h5py.File('/raid/scratch/devanshu/innerprods_1024.h5', 'r') as f:
	Q_train = np.array(f["/train_train"][::stride,::stride])
	Q_train_test = np.array(f["/train_test"][::stride,::stride])
	Q_test = np.array(f["/test_test"][::stride,::stride])

with h5py.File("/raid/ChestXRay14/chestxray14_1024.h5", "r") as hf:
	Y_train = np.array(hf["labels/train"][::stride]).astype(np.int32)
	Y_test = np.array(hf["labels/test"][::stride]).astype(np.int32)

print("Rescaling . . . ")
with open("../scale.json", "r") as fp:
	scale = json.load(fp)["scale"]
Q_train /= scale**2
Q_train_test /= scale**2
Q_test /= scale**2

Q_train = np.diagonal(Q_train)
Q_test = np.diagonal(Q_test)

print("Computing distances")
n_tasks = Y_train.shape[1]
with open("../gp/results/nngp2_best.json", "r") as fp:
	depth = json.load(fp)["depth"]
with open("../gp/results/nngp2_depth"+str(depth)+".json", "r") as fp:
	info = json.load(fp)
model = models.NNGP2(None, None, None, None, batch_size=info["batch_size"])
model.v_n = info["params"]["v_n"]
model.v_b = np.array(info["params"]["v_b"])
model.v_w = np.array(info["params"]["v_w"])
model.depth = depth

Q_train_test = model.K(Q_train_test, Q_train, Q_test)
Q_train = model.Kdiag(Q_train).reshape((-1, 1))
Q_test = model.Kdiag(Q_test).reshape((-1, 1))
Q_train_test = Q_train + Q_test.T - 2*Q_train_test
del Q_train; gc.collect()
del Q_test; gc.collect()
del model; gc.collect()

n_neighbors = 1000
print("Predicting")
n_tasks = Y_train.shape[1]
time_1 = time.time()
Y_pred = KNN(Q_train_test, Y_train, K=n_neighbors)
time_2 = time.time()

print("Computing metrics")
aucs = np.array([[roc_auc_score(Y_test[:,i], Y_pred[k,:,i]) for i in range(n_tasks)] for k in range(Y_pred.shape[0])])
aucs = np.concatenate((np.arange(1, n_neighbors+1).reshape((-1, 1)), aucs, np.mean(aucs, axis=1, keepdims=True)), axis=1)
Y_pred[Y_pred>0.5] = 1
Y_pred[Y_pred<0.5] = 0
Y_pred = np.where(Y_pred==0.5, np.random.randint(0, 2, size=Y_pred.shape), Y_pred)
accs = np.array([[accuracy_score(Y_test[:,i], Y_pred[k,:,i]) for i in range(n_tasks)] for k in range(Y_pred.shape[0])])
accs = np.concatenate((np.arange(1, n_neighbors+1).reshape((-1, 1)), accs, np.mean(accs, axis=1, keepdims=True)), axis=1)

print("Saving results")
with open("../headers.json", "r") as fp:
	headers = json.load(fp)
headers = ["K"] + headers + ["mean"]
aucs = pd.DataFrame(aucs, columns=headers)
accs = pd.DataFrame(accs, columns=headers)
aucs.to_csv("results/nngp2_aucs.csv", index=False)
accs.to_csv("results/nngp2_accs.csv", index=False)
best_index = aucs["mean"].values.reshape((-1)).argmax()
info = {"stride": stride,
	"pred_time": time_2-time_1,
	"script_time": time.time()-time_0,
	"n_neighbors": n_neighbors,
	"best": {"row_index": int(best_index),
		"n_neighbors": int(aucs["K"].values.reshape((-1))[best_index])
	}
}
with open("results/nngp2_info.json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")