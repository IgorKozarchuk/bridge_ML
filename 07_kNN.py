# K nearest neighbors
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score


df = pd.read_csv("AESUM_clean.csv")
# print(df.head())

# Features
X = df[["Довжина", "Категорія", "Обл", "ВікБуд"]]
# Target
y = df[["Стан"]]
# print(y.head())

# Split data into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# k-nearest neighbors classifier
# Find the best k
acc_scores = []

for k in range(1, 31): # range of k values
	knn = KNeighborsClassifier(n_neighbors=k)
	# using fit, predict and accuracy score
	knn.fit(X_train, y_train.values.ravel())
	y_pred = knn.predict(X_test)
	acc_scores.append(accuracy_score(y_test, y_pred))
	# OR using cross_val_score
	# score = cross_val_score(knn, X, y.values.ravel(), cv=5)
	# acc_scores.append(score.mean())

k = acc_scores.index(max(acc_scores))

plt.plot(range(1, 31), acc_scores)
plt.title("Accuracy vs K-value")
plt.xlabel("K")
plt.ylabel("Accuracy")
print("Max accuracy:", max(acc_scores), "at K =", k)
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=k) # k=3 by default

# Train the model
knn_model.fit(X_train, y_train.values.ravel())

# Score
print("Score:", knn_model.score(X_test, y_test))
print()

# Predict
y_pred = knn_model.predict(X_test)

# Metrics
print("Classification report:")
print(classification_report(y_test, y_pred))
print()

# Confusion matrix
c_matrix = confusion_matrix(y_test, y_pred)

print("Confusion matrix:")
print(c_matrix)
print()

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=knn_model.classes_)
disp.plot(cmap="magma_r")
disp.ax_.set_title("Confusion matrix")
plt.show()

# ROC AUC
knn_probs = knn_model.predict_proba(X_test)
knn_roc_auc = roc_auc_score(y_test, knn_probs, multi_class="ovr")

print("ROC AUC score:", knn_roc_auc)
