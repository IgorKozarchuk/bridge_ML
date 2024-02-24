""" K-means clustering (unsupervised ML algorithm) """

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Dataframe
df = pd.read_csv("AESUM_clean.csv")

# Features
X = df[["Довжина", "Категорія", "Обл", "ВікБуд"]]

# Target
y = df[["Стан"]]

# K = number of clusters
# NOTE: for unsupervised algorithms, we don't have target values,
# so you we have to use elbow method to determine the best K
# Elbow method
inertias = [] # Within the sum of squares (WSS) is the sum of the squared distance between each member of the cluster and its centroid

for k in range(1, 11):
	kmeans_model = KMeans(n_clusters=k)
	kmeans_model.fit(X)
	inertias.append(kmeans_model.inertia_)

plt.plot(range(1, 11), inertias, marker="o")
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia (WSS)")
plt.show()

K = 4 # according to the plot

# NOTE: for our example, we have target values,
# so K = number of unique target values, i.d. 5 technical conditions of bridges
K = y.nunique().iloc[0] # 5 технічних станів, iloc - to get int value form the series

# Kmeans model with optimal K
kmeans_model = KMeans(n_clusters=K)
kmeans_model.fit(X)

y_pred = kmeans_model.labels_ + 1 # labels start from 0 but "Стан" starts from 1
print(y_pred[:21])
print()

# Metrics
print("Classification report:")
print(classification_report(y, y_pred))
print()

# Confusion matrix
c_matrix = confusion_matrix(y, y_pred)
print("Confusion matrix:")
print(c_matrix)
print()

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=range(1, K+1))
disp.plot(cmap="magma_r")
disp.ax_.set_title("Confusion matrix")
plt.show()

# Export model
# https://scikit-learn.org/stable/model_persistence.html
# https://mljar.com/blog/save-load-scikit-learn-model/
file_path = "models/Kmeans_model.joblib"
# save model
joblib.dump(kmeans_model, file_path, compress=3)
# load model
loaded_model = joblib.load(file_path)
