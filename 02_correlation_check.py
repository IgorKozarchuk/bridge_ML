""" Correlation """

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


df = pd.read_csv("AESUM_clean.csv")
print(df.head())
print()

# Correlation matrix
print("Correlation:")
print(df.corr())
print()

plt.figure(figsize=(8, 6))
plt.title("Correlation matrix")
sns.heatmap(df.corr(), annot=True, cmap="magma_r")
plt.show()

# Print Pearson's correlation (sorted, not taking into account sign) - same result as df.corr()
def get_pearson_sorted(df):
	head = {"Feature": "Pearson correlation coefficient"}
	result = {}

	for x in df.columns[:-1]:
		p_corr = pearsonr(df[x], df[df.columns[-1]])[0] # current and last cols
		result[x] = p_corr
	
	# sorted dict
	sorted_result = {k: v for k, v in sorted(result.items(), key=lambda item: abs(item[1]), reverse=True)}

	# print
	print("Feature | Pearson's coefficient")
	for k, v in sorted_result.items():
		print(f"{k}: {v}")

	return sorted_result

get_pearson_sorted(df)
