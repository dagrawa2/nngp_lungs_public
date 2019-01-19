import json
import numpy as np
import pandas as pd

res = [16, 32, 64, 128, 256, 512]

with open("../headers.json", "r") as fp:
	headers = ["resolution"] + json.load(fp) + ["mean"]

accs = []
aucs = []
params = []
for r in res:
	data = pd.read_csv("results/nngp_metrics_depth5_res"+str(r)+".csv")
	acc = [r] + list(data["acc"].values.reshape((-1)))
	auc = [r] + list(data["auc"].values.reshape((-1)))
	accs.append(acc)
	aucs.append(auc)
	with open("results/nngp_depth5_res"+str(r)+".json", "r") as fp:
		param = json.load(fp)["params"]
	param = [r, param["v_n"], param["v_b"], param["v_w"]]
	params.append(param)

metrics_1024 = pd.read_csv("../gp/results/nngp_metrics_depth5.csv")
accs.append( [1024] + list(metrics_1024["acc"].values.reshape((-1))) )
aucs.append( [1024] + list(metrics_1024["auc"].values.reshape((-1))) )
with open("../gp/results/nngp_depth5.json", "r") as fp:
	params_1024 = json.load(fp)["params"]
params.append( [1024, params_1024["v_n"], params_1024["v_b"], params_1024["v_w"]] )

accs = np.array(accs)
aucs = np.array(aucs)
params = np.array(params)

accs = pd.DataFrame(accs, columns=headers)
aucs = pd.DataFrame(aucs, columns=headers)
params = pd.DataFrame(params, columns=["resolution", "v_n", "v_b", "v_w"])

accs.to_csv("results/nngp_acc.csv", index=False)
aucs.to_csv("results/nngp_auc.csv", index=False)
params.to_csv("results/nngp_param.csv", index=False)

best_index = aucs["mean"].values.reshape((-1)).argmax()
info = {"row_index": int(best_index),
	"res": int(aucs["resolution"].values.reshape((-1))[best_index])
}
with open("results/nngp_best.json", "w") as fp:
	json.dump(info, fp, indent=2)

print("Done!")