import numpy as np
import pandas as pd

def append_zeros(x, n):
	try:
		temp = float(x)
		m = len(x.split(".")[-1])
		if m < n:
			return x + "0"*(n-m)
		else:
			return x
	except:
		return x

data = pd.read_csv("table.csv", dtype=str)
headers = list(data.columns)
values = data.values.tolist()
data = [headers] + values

data = [[append_zeros(str(d), 3) for d in row] for row in data]

file = open("table.tex", "w")
file.write("\\begin{tabular}{|"+"c|"*len(headers)+"} \\hline\n")

for row in data:
	for d in row[:-1]:
		file.write(d+" & ")
	file.write(row[-1]+" \\\\ \\hline\n")

file.write("\\end{tabular}")
file.close()

print("Done!")