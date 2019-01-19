import gc
import json
import numpy as np
import pandas as pd

data = pd.read_csv("results/nngp2_aucs_depth5.csv").values
rejs = data[:,0]
aucs = data[:,1:]
del data; gc.collect()

sota = pd.read_csv("../table/sota.csv")
wang = sota["Wang"].values.reshape((-1))
wang = np.concatenate((wang, np.mean(wang, keepdims=True)))
yao = sota["Yao"].values.reshape((-1))
yao = np.concatenate((yao, np.mean(yao, keepdims=True)))
del sota; gc.collect()

wang_rejs = np.round(rejs[np.argmin(np.abs(aucs-wang.reshape((1, -1))), axis=0)], 3)
yao_rejs = np.round(rejs[np.argmin(np.abs(aucs-yao.reshape((1, -1))), axis=0)], 3)

with open("../headers.json", "r") as fp:
	headers = json.load(fp) + ["mean"]

data = pd.DataFrame.from_dict({"Condition": headers, "Wang": wang_rejs, "Yao": yao_rejs})[["Condition", "Wang", "Yao"]]
data.to_csv("tables/rejs.csv", index=False)

print("Done!")