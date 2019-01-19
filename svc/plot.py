import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

print("Plotting mean accuracies")
data = pd.read_csv("results/rbf_acc.csv", usecols=["C", "mean"])
C = data["C"].values.reshape((-1))
accs = data["mean"].values.reshape((-1))

plt.figure()
plt.plot(C, accs, color="black")
plt.xlabel("$C$")
plt.xscale("log")
plt.ylabel("Accuracy")
plt.title("Mean test accuracy of RBF SVC")
plt.savefig("plots/rbf_acc.png", box_inches="tight")


print("Plotting mean AUCs")
data = pd.read_csv("results/rbf_auc.csv", usecols=["C", "mean"])
C = data["C"].values.reshape((-1))
aucs = data["mean"].values.reshape((-1))

plt.figure()
plt.plot(C, aucs, color="black")
plt.xlabel("$C$")
plt.xscale("log")
plt.ylabel("AUC")
plt.title("Mean test ROC AUC of RBF SVC")
plt.savefig("plots/rbf_auc.png", box_inches="tight")

print("Done!")