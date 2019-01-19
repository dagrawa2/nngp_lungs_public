import h5py
import itertools
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

np.random.seed(123)
time_0 = time.time()

stride = 1
print("Loading data")
with h5py.File('../pca/results/pca.h5', 'r') as f:
	Z_train = np.array(f["/train"][::stride])
	Z_test = np.array(f["/test"][::stride])

with h5py.File("/raid/ChestXRay14/chestxray14_1024.h5", "r") as hf:
	Y_train = np.array(hf["labels/train"][::stride]).astype(np.int32)
	Y_test = np.array(hf["labels/test"][::stride]).astype(np.int32)
n_tasks = Y_train.shape[1]

n_estimators_list = [100, 300, 500, 700]
max_depth_list = [10, 50, 100, 500]
min_samples_leaf_list = [50, 100, 500]


aucs = []
times = []
n_gridpoints = len(list(itertools.product(n_estimators_list, max_depth_list, min_samples_leaf_list)))

print("Performing grid search")
print("---")
for iter, (n_estimators, max_depth, min_samples_leaf) in enumerate(list(itertools.product(n_estimators_list, max_depth_list, min_samples_leaf_list))):
	print(str(iter+1)+"/"+str(n_gridpoints)+": (n_estimators="+str(n_estimators)+", max_depth="+str(max_depth)+", min_samples_leaf="+str(min_samples_leaf)+")")
	print("Training")
	time_1 = time.time()
	classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight="balanced", n_jobs=-1, random_state=123))
	classifier.fit(Z_train, Y_train)
	time_2 = time.time()

	print("Predicting")
	pred = classifier.predict(Z_test)
	time_3 = time.time()
	print("Calculating scores")
	auc_row = [roc_auc_score(Y_test[:,i], pred[:,i]) for i in range(n_tasks)]
	aucs.append([n_estimators, max_depth, min_samples_leaf] + auc_row + [np.mean(np.array(auc_row))])
	times.append([n_estimators, max_depth, min_samples_leaf, time_2-time_1, time_3-time_2, time_3-time_1])
	print("mean AUC: ", np.round(aucs[-1][-1], 3))
	print("time: ", np.round(times[-1][-1], 3))
	print("---")

print("Saving results")
with open("../headers.json", "r") as fp:
	headers = ["n_estimators", "max_depth", "min_samples_leaf"] + json.load(fp) + ["mean"]
pd.DataFrame(np.array(aucs), columns=headers).to_csv("results/aucs.csv", index=False)
pd.DataFrame(np.array(times), columns=["n_estimators", "max_depth", "min_samples_leaf", "train_time", "pred_time", "total_time"]).to_csv("results/times.csv", index=False)

aucs = pd.read_csv("results/aucs.csv")
best_index = aucs["mean"].values.reshape((-1)).argmax()
info = {"stride": stride,
	"script_time": time.time()-time_0,
	"grid": {"n_estimators": n_estimators_list,
		"max_depth": max_depth_list,
		"min_samples_leaf": min_samples_leaf_list
	},
	"best": {"row_index": int(best_index),
		"n_estimators": int(aucs["n_estimators"].values.reshape((-1))[best_index]),
		"max_depth": int(aucs["max_depth"].values.reshape((-1))[best_index]),
		"min_samples_leaf": int(aucs["min_samples_leaf"].values.reshape((-1))[best_index])
	}
}

with open("results/info.json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")