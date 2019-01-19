import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

print("Plotting mean accuracies")
cos_accs = pd.read_csv("results/cos_accs.csv", usecols=["mean"]).values.reshape((-1))
euc_accs = pd.read_csv("results/euc_accs.csv", usecols=["mean"]).values.reshape((-1))
K = np.arange(1, cos_accs.shape[0]+1)

cos_accs = cos_accs[::2]
euc_accs = euc_accs[::2]
K = K[::2]

plt.figure()
plt.plot(K, cos_accs, color="red", label="Cosine similarity")
plt.plot(K, euc_accs, color="blue", label="Euclidean distance")
plt.legend(title="Metric")
plt.xlabel("$K$")
plt.ylabel("Accuracy")
plt.title("Mean test accuracy of KNN")
plt.savefig("plots/accs.png", box_inches="tight")

print("Plotting mean AUCs")
cos_aucs = pd.read_csv("results/cos_aucs.csv", usecols=["mean"]).values.reshape((-1))
euc_aucs = pd.read_csv("results/euc_aucs.csv", usecols=["mean"]).values.reshape((-1))
K = np.arange(1, cos_aucs.shape[0]+1)

cos_aucs = cos_aucs[::2]
euc_aucs = euc_aucs[::2]
K = K[::2]

plt.figure()
plt.plot(K, cos_aucs, color="red", label="Cosine similarity")
plt.plot(K, euc_aucs, color="blue", label="Euclidean distance")
plt.legend(title="Metric")
plt.xlabel("$K$")
plt.ylabel("AUC")
plt.title("Mean test ROC AUC of KNN")
plt.savefig("plots/aucs.png", box_inches="tight")

print("Done!")