import gc
import h5py
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from my_module import uq

np.random.seed(123)

with open("results/rbf.json", "r") as fp:
	stride = json.load(fp)["stride"]

print("Loading predictive mean and variance")
with h5py.File("results/rbf_pred.h5", "r") as f:
	Y_mean = np.array(f["/mean"])
	Y_std = np.expand_dims(np.sqrt(np.array(f["/var"])), 1)

with h5py.File("/raid/ChestXRay14/chestxray14_1024.h5", "r") as hf:
	Y_test = np.array(hf["labels/test"][::stride]).astype(np.int32)

print("Computing metrics")
Y_pred_hard = np.heaviside(Y_mean, np.random.randint(0, 2, size=Y_mean.shape))
Y_pred_soft = uq.Phi(Y_mean/Y_std)
accs = np.mean(np.equal(Y_test, Y_pred_hard), axis=0)
aucs = np.array([roc_auc_score(Y_test[:,j], Y_pred_soft[:,j]) for j in range(Y_test.shape[1])])

print("Saving results")
with open("../headers.json", "r") as fp:
	headers = json.load(fp)
data = pd.DataFrame.from_dict({"condition": headers, "acc": accs, "auc": aucs})[["condition", "acc", "auc"]]
data.loc[-1] = ["mean", np.mean(accs), np.mean(aucs)]
data.to_csv("results/rbf_metrics.csv", index=False)

print("Done!")