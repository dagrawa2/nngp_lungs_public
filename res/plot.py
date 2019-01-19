import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


with open("../headers.json", "r") as fp:
	headers = json.load(fp)

res_auc = pd.read_csv("results/nngp_auc.csv", usecols=["resolution", "Hernia", "mean"])
res = res_auc["resolution"].values.reshape((-1))
auc_hernia = res_auc["Hernia"].values.reshape((-1))
auc = res_auc["mean"].values.reshape((-1))

print("Plotting AUCs")
plt.figure()
plt.plot(np.arange(res.shape[0]), auc, color="blue", label="Mean")
plt.plot(np.arange(res.shape[0]), auc_hernia, color="red", label="Hernia")
plt.legend()
plt.xticks(np.arange(res.shape[0]), ["$"+str(r)+"$" for r in res])
plt.xlabel("Resolution")
plt.ylabel("AUC")
plt.title("Test ROC AUC of NNGP trained on various resolutions")
plt.savefig("plots/res_nngp_auc.png", box_inches="tight")


data = pd.read_csv("results/nngp_param.csv").values
res = data[:,0]
v_b = data[:,2]
v_w = data[:,3]

print("Plotting parameters")
plt.figure()
plt.plot(np.arange(res.shape[0]), v_b, color="red", label="$v_b$")
plt.plot(np.arange(res.shape[0]), v_w, color="blue", label="$v_w$")
plt.legend(title="Hyperparameter")
plt.xticks(np.arange(res.shape[0]), ["$"+str(r)+"$" for r in res])
plt.xlabel("Resolution")
plt.ylabel("Value")
plt.title("Optimal prior variances for an NNGP trained on various resolutions")
plt.savefig("plots/res_nngp_params.png", box_inches="tight")

print("Done!")