import gc
import argparse
import h5py
import json
import time
import numpy as np
import pandas as pd
from my_module import uq

parser = argparse.ArgumentParser(description="Predict with NNGP.")
parser.add_argument("--res", dest="res", type=int, default=1024, help="Resolution of images.")
args = parser.parse_args()
res = args.res

np.random.seed(123)
time_0 = time.time()

def zero_to_one(A):
	A[A==0] = 1
	return A

def fpr_tpr(Y_test, Y_pred, n_rejs, t):
	confs = np.where(Y_pred>=t, Y_pred, 1-Y_pred)
	Y_pred = np.where(Y_pred>=t, np.ones(Y_pred.shape), np.zeros(Y_pred.shape)).astype(np.bool)
	inds = np.argsort(confs, axis=0)
	del confs; gc.collect()
	Y_pred = np.stack([Y_pred[inds[:,i],i] for i in range(Y_test.shape[1])], axis=1)
	Y_test = np.stack([Y_test[inds[:,i],i] for i in range(Y_test.shape[1])], axis=1)
	del inds; gc.collect()
	TPR = zero_to_one(np.cumsum(np.logical_and(Y_pred, Y_test)[::-1], axis=0))/zero_to_one(np.cumsum(Y_test[::-1], axis=0))
	FPR = zero_to_one(np.cumsum(np.logical_and(Y_pred, np.logical_not(Y_test))[::-1], axis=0))/zero_to_one(np.cumsum(np.logical_not(Y_test)[::-1], axis=0))
	del Y_pred; gc.collect()
	del Y_test; gc.collect()
	stride = TPR.shape[0]//n_rejs
	TPR = TPR[::-stride]
	FPR = FPR[::-stride]
	return np.stack([FPR, TPR], axis=2)
	

stride = 1
print("Loading predictive mean and variance")
if res == 1024:
	with h5py.File("../gp/results/nngp_pred_depth5.h5", "r") as f:
		Y_mean = np.array(f["/mean"][::stride])
		Y_std = np.expand_dims(np.sqrt(np.array(f["/var"][::stride])), 1)
else:
	with h5py.File("../res/results/nngp_pred_depth5_res"+str(res)+".h5", "r") as f:
		Y_mean = np.array(f["/mean"][::stride])
		Y_std = np.expand_dims(np.sqrt(np.array(f["/var"][::stride])), 1)

print("Loading test labels")
with h5py.File("/raid/ChestXRay14/chestxray14_"+str(res)+".h5", "r") as hf:
	Y_test = np.array(hf["labels/test"][::stride]).astype(np.bool)

print("Computing predictive probabilities")
Y_pred = uq.Phi(Y_mean/Y_std)
del Y_mean; gc.collect()
del Y_std; gc.collect()
	
n_rejs = 1000
n_thresholds = 10000
rejs = np.arange(0, Y_test.shape[0], Y_test.shape[0]//n_rejs)/Y_test.shape[0]
thresholds = np.arange(n_thresholds+1)/n_thresholds

print("Computing rejection-AUC curves")
out = []
for i, t in enumerate(thresholds[::-1]):
	if i%100 == 0: print("  "+str(i)+"/"+str(n_thresholds)+"          ", end="\r")
	out.append( fpr_tpr(Y_test, Y_pred, n_rejs, t)  )

out = np.stack(out, axis=0)

aucs = np.sum((out[1:,:,:,0]-out[:-1,:,:,0])*(out[:-1,:,:,1]+out[1:,:,:,1]), axis=0)/2
aucs = np.concatenate((rejs.reshape((-1, 1)), aucs, np.mean(aucs, axis=1, keepdims=True)), axis=1)
del out; gc.collect()

print("Saving results")
with open("../headers.json", "r") as fp:
	headers = json.load(fp)

aucs = pd.DataFrame(aucs, columns=["rej"]+headers+["mean"])
aucs.to_csv("results/nngp_aucs_depth5_res"+str(res)+".csv", index=False)

info = {"stride": stride,
	"time": time.time()-time_0,
	"n_rejs": n_rejs,
	"n_thresholds": n_thresholds
}
with open("results/nngp_info_depth5_res"+str(res)+".json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")
