import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


with open("../headers.json", "r") as fp:
	headers = json.load(fp)

depth_auc = pd.read_csv("results/nngp_auc.csv", usecols=["depth", "mean"])
depth = depth_auc["depth"].values.reshape((-1))
auc = depth_auc["mean"].values.reshape((-1))
auc2 = pd.read_csv("results/nngp2_auc.csv", usecols=["mean"]).values.reshape((-1))

print("Plotting AUCs")
plt.figure()
plt.plot(depth, auc, color="red", label="Uniform")
plt.plot(depth, auc2, color="blue", label="Varying")
plt.legend(title="Hyperparameters")
plt.xlabel("Depth")
plt.ylabel("AUC")
plt.title("Test ROC AUC of NNGP of various depths")
plt.savefig("plots/auc.png", box_inches="tight")


data = pd.read_csv("results/nngp_param.csv").values
depth = data[:,0]
v_b = data[:,2]
v_w = data[:,3]

print("Plotting parameters")
plt.figure()
plt.plot(depth, v_b, color="red", label="$v_b$")
plt.plot(depth, v_w, color="blue", label="$v_w$")
plt.legend(title="Hyperparameter")
plt.xlabel("Depth")
plt.ylabel("Value")
plt.title("Optimal prior variances for an NNGP for various depths")
plt.savefig("plots/nngp_params.png", box_inches="tight")

drop_nan = lambda row: row[row!=np.nan]
colors = ["purple", "blue", "green", "orange", "pink", "red"]

data = pd.read_csv("results/nngp2_param_v_w.csv").values
depth = data[:,0]
param = data[:,1:]

print("Plotting v_w")
plt.figure()
plt.scatter(depth[:1], param[0,:1], color=colors[0], label=str(0))
for i in range(1, len(depth)):
	plt.plot(depth[:i+1], param[i,:i+1], color=colors[i], label=str(i))
plt.legend(title="Depth")
plt.xlabel("Layer")
plt.ylabel("$v_w$")
plt.title("Optimal $v_w$ at each layer of NNGP of various depths")
plt.savefig("plots/nngp2_param_v_w.png", box_inches="tight")

data = pd.read_csv("results/nngp2_param_v_b.csv").values
depth = data[:,0]
param = data[:,1:]

print("Plotting v_b")
plt.figure()
plt.scatter(depth[:1], param[0,:1], color=colors[0], label=str(0))
for i in range(1, len(depth)):
	plt.plot(depth[:i+1], param[i,:i+1], color=colors[i], label=str(i))
plt.legend(title="Depth")
plt.xlabel("Layer")
plt.ylabel("$v_b$")
plt.title("Optimal $v_b$ at each layer of NNGP of various depths")
plt.savefig("plots/nngp2_param_v_b.png", box_inches="tight")

print("Done!")