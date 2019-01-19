import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


with open("../headers.json", "r") as fp:
	headers = json.load(fp)
acc = pd.read_csv("results/nngp2_acc.csv", usecols=["mean"]).values.reshape((-1))
depth_auc = pd.read_csv("results/nngp2_auc.csv", usecols=["depth", "mean"])
depth = depth_auc["depth"].values.reshape((-1))
auc = depth_auc["mean"].values.reshape((-1))

print("Plotting accuracies")
plt.figure()
plt.plot(depth, acc, color="black")
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.title("Test accuracy of NNGP of various depths")
plt.savefig("plots/acc.png", box_inches="tight")

print("Plotting AUCs")
plt.figure()
plt.plot(depth, auc, color="black")
plt.xlabel("Depth")
plt.ylabel("AUC")
plt.title("Test ROC AUC of NNGP of various depths")
plt.savefig("plots/auc.png", box_inches="tight")


drop_nan = lambda row: row[row!=np.nan]
colors = ["black", "brown", "red", "orange", "green", "blue", "purple", "pink"]

data = pd.read_csv("results/nngp2_v_w.csv").values
depth = data[:,0]
param = [drop_nan(data[i]) for i in range(data.shape)]

print("Plotting v_w")
plt.figure()
for i in range(len(depth)):
	plt.plot(depth[:i+1], param[i], color=colors[i], label=str(i))
plt.legend(title="Depth")
plt.xlabel("Layer")
plt.ylabel("$v_w$")
plt.title("Optimal $v_w$ at each layer of NNGP of various depths")
plt.savefig("plots/nngp2_v_w.png", box_inches="tight")

data = pd.read_csv("results/nngp2_v_b.csv").values
depth = data[:,0]
param = [drop_nan(data[i]) for i in range(data.shape)]

print("Plotting v_b")
plt.figure()
for i in range(len(depth)):
	plt.plot(depth[:i+1], param[i], color=colors[i], label=str(i))
plt.legend(title="Depth")
plt.xlabel("Layer")
plt.ylabel("$v_b$")
plt.title("Optimal $v_b$ at each layer of NNGP of various depths")
plt.savefig("plots/nngp2_v_b.png", box_inches="tight")

print("Done!")