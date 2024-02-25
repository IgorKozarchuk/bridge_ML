""" Random forest (ensemble algorithm)"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve


# Dataframe
df = pd.read_csv("AESUM_clean.csv")
# Features
X = df[["Довжина", "Категорія", "Обл", "ВікБуд"]]
# Target
y = df[["Стан"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random forest model
rand_forest_model = RandomForestClassifier()

# Train
rand_forest_model.fit(X_train, np.ravel(y_train, order="C"))

# Score
print("Score:", rand_forest_model.score(X_test, y_test))
print()

# Predict
y_pred = rand_forest_model.predict(X_test)

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
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=rand_forest_model.classes_)
disp.plot(cmap="magma_r")
disp.ax_.set_title("Confusion matrix")
plt.show()

# ROC AUC - Area Under the Receiver Operating Characteristic Curve
y_probs = rand_forest_model.predict_proba(X_test)
rand_forest_roc_auc = roc_auc_score(y_test, y_probs, multi_class="ovr")
print("ROC AUC score:", rand_forest_roc_auc)

# Plot AUC curve (fpr, tpr - false positive and true positive rates)
# NOTE: this implementation is restricted to the binary classification task
# fpr, tpr, thresholds = roc_curve(y_test, y_probs) # !!! ValueError: multiclass format is not supported
# plt.plot(fpr, tpr)
# plt.title("Area under the ROC curve")
# plt.show()

# Export model
file_path = "models/random_forest_model.joblib"
# save model
joblib.dump(rand_forest_model, file_path, compress=3)
# load model
loaded_model = joblib.load(file_path)
