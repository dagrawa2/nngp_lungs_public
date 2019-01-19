import json
import numpy as np
import pandas as pd

depths = range(6)

with open("../headers.json", "r") as fp:
	headers = ["depth"] + json.load(fp) + ["mean"]

accs = []
aucs = []
params = []
for d in depths:
	data = pd.read_csv("results/nngp_metrics_depth"+str(d)+".csv")
	acc = [d] + list(data["acc"].values.reshape((-1)))
	auc = [d] + list(data["auc"].values.reshape((-1)))
	accs.append(acc)
	aucs.append(auc)
	with open("results/nngp_depth"+str(d)+".json", "r") as fp:
		param = json.load(fp)["params"]
	param = [d, param["v_n"], param["v_b"], param["v_w"]]
	params.append(param)

accs = np.array(accs)
aucs = np.array(aucs)
params = np.array(params)

accs = pd.DataFrame(accs, columns=headers)
aucs = pd.DataFrame(aucs, columns=headers)
params = pd.DataFrame(params, columns=["depth", "v_n", "v_b", "v_w"])

accs.to_csv("results/nngp_acc.csv", index=False)
aucs.to_csv("results/nngp_auc.csv", index=False)
params.to_csv("results/nngp_param.csv", index=False)

best_index = aucs["mean"].values.reshape((-1)).argmax()
info = {"row_index": int(best_index),
	"depth": int(aucs["depth"].values.reshape((-1))[best_index])
}
with open("results/nngp_best.json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")