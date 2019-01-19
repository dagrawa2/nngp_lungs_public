import numpy as np
import pandas as pd

data = pd.read_csv("tables/rejs.csv", dtype=str)
headers = list(data.columns)
values = data.values.tolist()
data = [headers] + values

data = [[str(d) for d in row] for row in data]

file = open("tables/rejs.tex", "w")
file.write("\\begin{tabular}{|"+"c|"*len(headers)+"} \\hline\n")

for row in data:
	for d in row[:-1]:
		file.write(d+" & ")
	file.write(row[-1]+" \\\\ \\hline\n")

file.write("\\end{tabular}")
file.close()

print("Done!")