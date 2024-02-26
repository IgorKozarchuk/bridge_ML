""" Cross validation """

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("AESUM_clean.csv")

X = df[["Довжина", "Категорія", "Обл", "ВікБуд"]]
y = df[["Стан"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.values.ravel()

# Calculate cross validation score for models
def get_cross_val_scores(X_train, y_train, folds_num=5, *models):
	results = {}

	def get_scores():
		cv_result = cross_val_score(model, X_train, y_train, cv=folds_num) # cv=5 by default
		results[model.__str__()] = {
			"scores": cv_result,
			"mean": cv_result.mean()
		}

	if isinstance(models[0], list): # if models is a list
		for model in models[0]:
			get_scores()
	else: # if models are arbitrary arguments
		for model in models:
			get_scores()

	return results


# Get model with the best score
def get_best_model(cross_val_results):
	max_key = None
	max_val = 0

	for key, value in cross_val_results.items():
		if value["mean"] > max_val:
			max_val = value["mean"]
			max_key = key

	return (max_key, max_val)


# Print model with the best score
def print_best_model(model_tuple):
	print("MODEL WITH THE BEST SCORE")
	print("(model name | mean score)")
	print(f"{model_tuple[0]} | {model_tuple[1]}")


# Print cross validation scores
def print_cross_val_scores(cross_val_results):
	for key, value in cross_val_results.items():
		print((f"{key}:\n"
				f"scores - {value['scores']}\n"
				f"mean - {value['mean']}\n"
				f"{'_'*30}"))


models = [
	LogisticRegression(max_iter=1000),
	tree.DecisionTreeClassifier(),
	SVC(),
	KNeighborsClassifier(n_neighbors=4),
	RandomForestClassifier()
]

results = get_cross_val_scores(X_train, y_train, 5, models)

print_cross_val_scores(results)
print()
print_best_model(get_best_model(results))
