import json
import numpy as np
import pandas as pd

depths = range(7)
n_iters = []
times = []
times_per_iter = []

for d in depths:
	with open("results/nngp_ho_depth"+str(d)+".json", "r") as fp:
		info = json.load(fp)
	n_iters.append(len(info["hist"]["nle_train"]))
	times.append(info["script_time"])
	times_per_iter.append(times[-1]/n_iters[-1])

data = pd.DataFrame.from_dict({"depth": depths, "n_iter": n_iters, "time": times, "time_per_iter": times_per_iter})[["depth", "n_iter", "time", "time_per_iter"]]
data.to_csv("results/nngp_tim.csv", index=False)

print("Done!")