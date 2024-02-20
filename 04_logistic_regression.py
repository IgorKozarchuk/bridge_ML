import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("AESUM_clean.csv")
# df = pd.read_csv("AESUM_stand.csv")
# df = pd.read_csv("AESUM_norm.csv")
# print(df.head())

# Features
X = df[["Довжина", "Категорія", "Обл", "ВікБуд"]]
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
