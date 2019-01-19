import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


with open("../headers.json", "r") as fp:
	headers = json.load(fp)
acc = pd.read_csv("results/nngp_acc.csv", usecols=["mean"]).values.reshape((-1))
depth_auc = pd.read_csv("results/nngp_auc.csv", usecols=["depth", "mean"])
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
plt.savefig("plots/nngp_v_w.png", box_inches="tight")

print("Done!")