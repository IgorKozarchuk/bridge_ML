# Support Vector Machine
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("AESUM_clean.csv")
# print(df.head())

# Features
X = df[["Довжина", "Категорія", "Обл", "ВікБуд"]]
# Target
y = df[["Стан"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Support Vector Classifier
SVC_model = SVC(C=10)
"""
C - float, default=1.0
kernel - {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable, default='rbf'
gamma - {'scale', 'auto'} or float, default='scale'
"""

# Train
SVC_model.fit(X_train, y_train.values.ravel())

# Score 
print(SVC_model.score(X_test, y_test))
print()

# Predict
y_pred = SVC_model.predict(X_test)

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
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=SVC_model.classes_)
disp.plot(cmap="magma_r")
disp.ax_.set_title("Confusion matrix")
plt.show()
