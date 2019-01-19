import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

colors = ["purple", "blue", "green", "orange", "pink", "red", "brown"]

print("Plotting over depth")
plt.figure()
for d in range(6):
	data = pd.read_csv("results/nngp2_aucs_depth"+str(d)+".csv", usecols=["rej", "mean"]).values
	plt.plot(data[:,0], data[:,1], color=colors[d], label=str(d))
plt.legend(title="Depth")
plt.xlabel("Rejection rate")
plt.ylabel("AUC")
plt.title("AUC-rejection curves for NNGPs of various depths")
plt.savefig("plots/uq_nngp2_depth.png", box_inches="tight")

print("Plotting over resolution")
plt.figure()
for i, r in enumerate([16, 32, 64, 128, 256, 512, 1024]):
	data = pd.read_csv("results/nngp_aucs_depth5_res"+str(r)+".csv", usecols=["rej", "mean"]).values
	plt.plot(data[:,0], data[:,1], color=colors[i], label=str(r))
plt.legend(title="Resolution")
plt.xlabel("Rejection rate")
plt.ylabel("AUC")
plt.title("AUC-rejection curves for NNGPs trained at various resolutions")
plt.savefig("plots/uq_nngp_res.png", box_inches="tight")

print("Plotting against RBF")
plt.figure()
data = pd.read_csv("results/nngp2_aucs_depth5.csv", usecols=["rej", "mean"]).values
plt.plot(data[:,0], data[:,1], color="red", label="NNGP")
data = pd.read_csv("results/rbf_aucs.csv", usecols=["rej", "mean"]).values
plt.plot(data[:,0], data[:,1], color="blue", label="RBF")
plt.legend(title="Kernel")
plt.xlabel("Rejection rate")
plt.ylabel("AUC")
plt.title("AUC-rejection curves for NNGP vs. RBF kernels")
plt.savefig("plots/uq_nngp2_rbf.png", box_inches="tight")

print("Done!")