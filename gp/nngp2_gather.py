import json
import numpy as np
import pandas as pd

depths = range(1, 6)

with open("../headers.json", "r") as fp:
	headers = ["depth"] + json.load(fp) + ["mean"]

data = pd.read_csv("results/nngp_metrics_depth0.csv")
with open("results/nngp_depth0.json", "r") as fp:
	params = json.load(fp)["params"]
accs = [[0] + list(data["acc"].values.reshape((-1)))]
aucs = [[0] + list(data["auc"].values.reshape((-1)))]
params_v_b = [[0] + [params["v_b"]] + [np.nan]*depths[-1]]
params_v_w = [[0] + [params["v_w"]] + [np.nan]*depths[-1]]

for d in depths:
	data = pd.read_csv("results/nngp2_metrics_depth"+str(d)+".csv")
	acc = [d] + list(data["acc"].values.reshape((-1)))
	auc = [d] + list(data["auc"].values.reshape((-1)))
	accs.append(acc)
	aucs.append(auc)
	with open("results/nngp2_depth"+str(d)+".json", "r") as fp:
		params = json.load(fp)["params"]
	params_v_b.append([d] + params["v_b"] + [np.nan]*(depths[-1]-d))
	params_v_w.append([d] + params["v_w"] + [np.nan]*(depths[-1]-d))

accs = np.array(accs)
aucs = np.array(aucs)
params_v_b = np.array(params_v_b)
params_v_w = np.array(params_v_w)

accs = pd.DataFrame(accs, columns=headers)
aucs = pd.DataFrame(aucs, columns=headers)
params_v_b = pd.DataFrame(params_v_b, columns=["depth"]+["layer"+str(i) for i in range(depths[-1]+1)])
params_v_w = pd.DataFrame(params_v_w, columns=["depth"]+["layer"+str(i) for i in range(depths[-1]+1)])

accs.to_csv("results/nngp2_acc.csv", index=False)
aucs.to_csv("results/nngp2_auc.csv", index=False)
params_v_b.to_csv("results/nngp2_param_v_b.csv", index=False)
params_v_w.to_csv("results/nngp2_param_v_w.csv", index=False)

best_index = aucs["mean"].values.reshape((-1)).argmax()
info = {"row_index": int(best_index),
	"depth": int(aucs["depth"].values.reshape((-1))[best_index])
}
with open("results/nngp2_best.json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")