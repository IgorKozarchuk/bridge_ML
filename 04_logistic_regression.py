""" Logistic regression """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve


# Dataframe
df = pd.read_csv("AESUM_clean.csv")
# df = pd.read_csv("AESUM_stand.csv")
# df = pd.read_csv("AESUM_norm.csv")
# print(df.head())

# Features
X = df[["Довжина", "Категорія", "Обл", "ВікБуд"]]
# X = df[["Довжина", "Категорія", "Обл", "ВікРем"]]
# print(X.head())
# Target
y = df[["Стан"]]
# print(y.head())

# Split data into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic regression model
logistic_model = LogisticRegression(max_iter=1000)

# Train the model
logistic_model.fit(X_train, y_train.values.ravel())

# Score
print(logistic_model.score(X_test, y_test))
print()

# Predict
y_pred = logistic_model.predict(X_test)

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
# plt.figure(figsize=(8, 6))
# plt.title("Confusion matrix")
# sns.heatmap(c_matrix, annot=True, cmap="magma_r", fmt="g")
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=logistic_model.classes_)
disp.plot(cmap="magma_r")
disp.ax_.set_title("Confusion matrix")
plt.show()

# ROC AUC - Area Under the Receiver Operating Characteristic Curve
y_probs = logistic_model.predict_proba(X_test)
logr_roc_auc = roc_auc_score(y_test, y_probs, multi_class="ovr")
print("ROC AUC score:", logr_roc_auc)

# plot AUC curve (fpr, tpr - false positive and true positive rates)
# fpr, tpr, thresholds = roc_curve(y_test, y_probs) # !!! ValueError: multiclass format is not supported
# plt.plot(fpr, tpr)
# plt.title("Area under the ROC curve")
# plt.show()

# Export model
# https://scikit-learn.org/stable/model_persistence.html
# https://mljar.com/blog/save-load-scikit-learn-model/
file_path = "models/logistic_model.joblib"
# save model
joblib.dump(logistic_model, file_path, compress=3)
# load model
loaded_model = joblib.load(file_path)
