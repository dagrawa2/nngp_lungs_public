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
parser.add_argument("--res", dest="res", type=int, default=1024, help="Resolution of images.")
args = parser.parse_args()
depth = args.depth
res = args.res


np.random.seed(123)

time_0 = time.time()

stride = 5
print("Resolution:", res)
print("Loading data")
with h5py.File('/raid/scratch/devanshu/innerprods_'+str(res)+'.h5', 'r') as f:
	Q_train_train = np.array(f["/train_train"][::stride,::stride])
	Q_train_test = None # np.array(f["/train_test"][::stride,:2])
	Q_test_test = None # np.array(f["/test_test"][:2,:2])

with h5py.File("/raid/ChestXRay14/chestxray14_1024.h5", "r") as hf:
	Y_train = 2*np.array(hf["labels/train"][::stride]).astype(np.int32) - 1

print("Rescaling . . . ")
with open("results/scales.json", "r") as fp:
	scale = json.load(fp)[str(res)]
Q_train_train /= scale**2
#Q_train_test /= scale**2
#Q_test_test /= scale**2

max_iters = 50
print("Training")
n_tasks = Y_train.shape[1]
model = models.NNGP(Q_train_train, Q_train_test, Q_test_test, Y_train, v_n=0.01, v_b=1.0, v_w=1.0, depth=depth, batch_size=None)
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
	"depth": depth,
	"res": res,
	"hist": history
}
with open("results/nngp_ho_depth"+str(depth)+"_res"+str(res)+".json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")