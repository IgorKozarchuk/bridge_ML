""" Cleaning """

import pandas as pd
import numpy as np


df = pd.read_csv("AESUM.csv")
# print(df.info())
# print()


# Remove rows with not concrete bridges
def remove_not_concrete(df):
	for x in df.index:
		if df.loc[x, "Матеріал"] != "Залізобетон":
			df.drop(x, inplace=True)

remove_not_concrete(df)


# Remove unneeded columns
features = ["Будів", "Реконстр", "Ремонт", "Обстеж", "Довжина", "Категорія", "Обл", "Стан"]
df = df[features]


# Remove rows with NaN in selected columns
selected = ["Будів", "Обстеж", "Довжина", "Категорія", "Обл", "Стан"]

df.dropna(subset=selected, inplace=True)
print(df.info())
print()


# Remove rows with 0 in selected columns
df = df.loc[(df[selected] != 0).all(axis=1)]
print(df.info())
print()


# Remove duplicates
print(df.duplicated().any())
df.drop_duplicates(inplace=True)


# Create column "ВікБуд" (Обстження - Будів)
df["ВікБуд"] = df["Обстеж"] - df["Будів"]


# Create column "ВікРем" (Обстження - Реконстр OR Обстження - Реконстр OR Обстження - Будів)
df[["Реконстр", "Ремонт"]] = df[["Реконстр", "Ремонт"]].replace(["0", 0], np.nan)
df["ВікРем"] = np.nan

def add_repair_age_col(df):
	for x in df.index:
		if pd.isna(df.at[x, "Реконстр"]) and pd.isna(df.at[x, "Ремонт"]):
			df.at[x, "ВікРем"] = df.at[x, "ВікБуд"]
		else:
			survey_year = df.at[x, "Обстеж"]
			repair_year = max(df.at[x, "Реконстр"], df.at[x, "Ремонт"])
			if repair_year <= survey_year:
				df.at[x, "ВікРем"] = survey_year - repair_year
			else:
				df.at[x, "ВікРем"] = df.at[x, "ВікБуд"]

add_repair_age_col(df)


# Remove unneeded columns
df.drop(["Будів", "Обстеж", "Реконстр", "Ремонт"], axis=1, inplace=True)


# Replace "II категорія" to "2", etc.
mapping = {
	"Iа категорія": 1,
	"Iб категорія": 1,
	"II категорія": 2,
	"III категорія": 3,
	"IV категорія": 4,
	"V категорія": 5
}

df.replace({"Категорія": mapping}, inplace=True)

print(df.info())


# Rearrange columns (target feature in the end)
df = df[["Довжина", "Категорія", "Обл", "ВікБуд", "ВікРем", "Стан"]]


# Save clean dataset to a new file
df.to_csv("AESUM_clean.csv", encoding="utf-8-sig", index=False)
