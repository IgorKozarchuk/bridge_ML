import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score


df = pd.read_csv("AESUM_clean.csv")
# print(df.head())

# Features
X = df[["Довжина", "Категорія", "Обл", "ВікБуд"]]
# Target
y = df[["Стан"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision tree model
dtree_model = tree.DecisionTreeClassifier()

# Train the model
dtree_model.fit(X_train, y_train)

# Score
print(dtree_model.score(X_test, y_test))
print()

# Predict
y_pred = dtree_model.predict(X_test)

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
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=dtree_model.classes_)
disp.plot(cmap="magma_r")
disp.ax_.set_title("Confusion matrix")
plt.show()

# ROC AUC - Area Under the Receiver Operating Characteristic Curve
dtree_probs = dtree_model.predict_proba(X_test)
dtree_roc_auc = roc_auc_score(y_test, dtree_probs, multi_class="ovr")
print("ROC AUC score:", dtree_roc_auc)
