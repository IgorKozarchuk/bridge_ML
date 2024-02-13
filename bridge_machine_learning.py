import datetime
import pandas as pd


df = pd.read_csv("AESUM.csv")
# print(df.head())
# print(df.info())
# print()


# *** 1. Data Preparation ***

# check for null values
print(df.isnull().sum())
print()

# remove rows with not concrete bridges
def remove_not_concrete(df):
	for x in df.index:
		if df.loc[x, "Матеріал"] != "Залізобетон":
			df.drop(x, inplace=True)

remove_not_concrete(df)
print(df.info())
print()

# selected features
target = "Стан"
features = ["Рейтинг", "Будів", "Довжина", "Категорія"]

# remove rows with null selected features
def remove_null_features(df):
	for x in df.index:
		for f in [target] + features:
			if df.isnull().loc[x, f]:
				df.drop(x, inplace=True)
				break

remove_null_features(df)
print(df.info())
print()

# add "Вік" column to the end of df
df = df.assign(Вік = datetime.datetime.now().year - df["Будів"])
print(df.head())
print()

df = df.drop("Будів", axis=1)

features.remove("Будів")
features.append("Вік")

print(df.head())
print(df.info())
print()
