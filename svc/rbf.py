import h5py
import json
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

np.random.seed(123)
time_0 = time.time()

stride = 1
print("Loading data")
with h5py.File('/raid/scratch/devanshu/dists_1024.h5', 'r') as f:
	K_train_train = np.array(f["/train_train"][::stride,::stride])
	K_train_test = np.array(f["/train_test"][::stride,::stride])

with h5py.File("/raid/ChestXRay14/chestxray14_1024.h5", "r") as hf:
	Y_train = np.array(hf["labels/train"][::stride]).astype(np.int32)
	Y_test = np.array(hf["labels/test"][::stride]).astype(np.int32)

print("Rescaling . . . ")
with open("../scale.json", "r") as fp:
	scale = json.load(fp)["scale"]
K_train_train /= scale**2
K_train_test /= scale**2

print("Computing RBF kernel . . . ")
with open("../gp/results/rbf.json", "r") as fp:
	lengthscale = json.load(fp)["params"]["l"]
K_train_train = np.exp(-K_train_train/2/lengthscale**2)
K_train_test = np.exp(-K_train_test/2/lengthscale**2)

C_range = [0.01, 0.1, 1.0, 10, 100]

n_tasks = Y_train.shape[1]
accs = []
aucs = []
pos = []
times = []

print("Training")
for C in C_range:
	print("C =", C)
	time_1 = time.time()
	#classifier = OneVsRestClassifier(LinearSVC())
	classifier = OneVsRestClassifier(SVC(kernel="precomputed", C=C, class_weight="balanced", probability=True), n_jobs=-1)
	classifier.fit(K_train_train, Y_train)
	time_2 = time.time()

	print("Predicting")
	pred = classifier.predict(K_train_test.T)
	acc_row = [accuracy_score(Y_test[:,i], pred[:,i]) for i in range(n_tasks)]
	accs.append([C] + acc_row + [np.mean(np.array(acc_row))])
	auc_row = [roc_auc_score(Y_test[:,i], pred[:,i]) for i in range(n_tasks)]
	aucs.append([C] + auc_row + [np.mean(np.array(auc_row))])
	pos_row = np.mean(pred, axis=0)
	pos.append([C] + list(pos_row) + [np.mean(pos_row)])
	times.append([C, time_2-time_1])


print("Saving results")
with open("../headers.json", "r") as fp:
	headers = ["C"] + json.load(fp) + ["mean"]
pd.DataFrame(np.array(accs), columns=headers).to_csv("results/rbf_acc.csv", index=False)
pd.DataFrame(np.array(aucs), columns=headers).to_csv("results/rbf_auc.csv", index=False)
pd.DataFrame(np.array(pos), columns=headers).to_csv("results/rbf_pos.csv", index=False)
pd.DataFrame(np.array(times), columns=["C", "time"]).to_csv("results/rbf_tim.csv", index=False)

aucs = pd.read_csv("results/rbf_auc.csv")
best_index = aucs["mean"].values.reshape((-1)).argmax()
info = {"stride": stride,
	"script_time": time.time()-time_0,
	"lengthscale": lengthscale,
	"best": {"row_index": int(best_index),
		"C": float(aucs["C"].values.reshape((-1))[best_index])
	}
}

with open("results/rbf_info.json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")