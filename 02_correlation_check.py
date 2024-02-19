import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("AESUM_clean.csv")
print(df.head())

x1 = df[["Довжина"]]
x2 = df[["Категорія"]]
x3 = df[["Обл"]]
x4 = df[["ВікБуд"]]
x5 = df[["ВікРем"]]

y = df[["Стан"]]

plt.scatter(x1, y)
plt.show()
plt.scatter(x2, y)
plt.show()
plt.scatter(x3, y)
plt.show()
plt.scatter(x4, y)
plt.show()
plt.scatter(x5, y)
plt.show()
