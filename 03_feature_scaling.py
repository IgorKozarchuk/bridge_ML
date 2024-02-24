""" Scaling """

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("AESUM_clean.csv")
# print(df.describe().round(2))
# print()


# *** Standardization ***
X1 = df.iloc[:, :-1]

scaler_stand = StandardScaler()
scaled_X1 = scaler_stand.fit_transform(X1)

scaled_X1_df = pd.DataFrame(scaled_X1, columns=df.columns[:-1])

print(scaled_X1_df.head())
print()
# https://stackoverflow.com/questions/40347689/dataframe-describe-suppress-scientific-notation
print(scaled_X1_df.describe().apply(lambda s: s.apply("{0:.5f}".format)))
print()

df1 = pd.concat([scaled_X1_df, df[df.columns[-1]]], axis=1, join="inner")
print(df1.head())
print()

df1.to_csv("AESUM_stand.csv", encoding="utf-8-sig", index=False)


# *** Normalization [0...1] ***
X2 = df.iloc[:, :-1]

scaler_norm = MinMaxScaler()
scaled_X2 = scaler_norm.fit_transform(X2)

scaled_X2_df = pd.DataFrame(scaled_X2, columns=df.columns[:-1])

print(scaled_X2_df.head())
print()

print(scaled_X2_df.describe().apply(lambda s: s.apply("{0:.5f}".format)))
print()

df2 = pd.concat([scaled_X2_df, df[df.columns[-1]]], axis=1, join="inner")
print(df2.head())
print()

df2.to_csv("AESUM_norm.csv", encoding="utf-8-sig", index=False)
