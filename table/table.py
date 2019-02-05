import json
import numpy as np
import pandas as pd

def load_json(path):
	with open(path, "r") as f:
		D = json.load(f)
	return D

headers = load_json("../headers.json") + ["mean"]
#nngp = np.round(pd.read_csv("../gp/results/nngp_auc.csv").values[load_json("../gp/results/nngp_best.json")["row_index"], 1:], 3)
nngp = np.round(pd.read_csv("../gp/results/nngp2_auc.csv").values[load_json("../gp/results/nngp2_best.json")["row_index"], 1:], 3)
rbfgp = np.round(pd.read_csv("../gp/results/rbf_metrics.csv", usecols=["auc"]).values.reshape((-1)), 3)
#knnnngp = np.round(pd.read_csv("../knn/results/nngp2_aucs.csv").values[load_json("../knn/results/nngp2_info.json")["best"]["row_index"], 1:], 3)
knneuc = np.round(pd.read_csv("../knn/results/euc_aucs.csv").values[load_json("../knn/results/euc_info.json")["best"]["row_index"], 1:], 3)
#knncos = np.round(pd.read_csv("../knn/results/cos_aucs.csv").values[load_json("../knn/results/cos_info.json")["best"]["row_index"], 1:], 3)
#nngpsvc = np.round(pd.read_csv("../svc/results/nngp2_auc.csv").values[load_json("../svc/results/nngp2_info.json")["best"]["row_index"], 1:], 3)
rbfsvc = np.round(pd.read_csv("../svc/results/rbf_auc.csv").values[load_json("../svc/results/rbf_info.json")["best"]["row_index"], 1:], 3)
rf = np.round(pd.read_csv("../rf/results/aucs.csv").values[load_json("../rf/results/info.json")["best"]["row_index"], 3:], 3)

sota = pd.read_csv("sota.csv")
#wang = sota["Wang"].values.reshape((-1))
wang = sota["Wang_correct"].values.reshape((-1))
wang = np.round(np.concatenate((wang, np.mean(wang, keepdims=True)), axis=0), 3)
yao = sota["Yao"].values.reshape((-1))
yao = np.round(np.concatenate((yao, np.mean(yao, keepdims=True)), axis=0), 3)

#data = pd.DataFrame.from_dict({"Condition": headers, "GP-NNGP": nngp, "GP-RBF": rbfgp, "KNN-NNGP": knnnngp, "KNN-EUC": knneuc, "SVC-NNGP": nngpsvc, "SVC-RBF": rbfsvc, "RF": rf, "Wang": wang, "Yao": yao})[["Condition", "GP-NNGP", "GP-RBF", "KNN-NNGP", "KNN-EUC", "SVC-NNGP", "SVC-RBF", "RF", "Wang", "Yao"]]
data = pd.DataFrame.from_dict({"Condition": headers, "GP-NNGP": nngp, "GP-RBF": rbfgp, "KNN-EUC": knneuc, "SVC-RBF": rbfsvc, "RF": rf, "Wang": wang, "Yao": yao})[["Condition", "GP-NNGP", "GP-RBF", "KNN-EUC", "SVC-RBF", "RF", "Wang", "Yao"]]
data.to_csv("table.csv", index=False)

print("Done!")