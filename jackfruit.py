from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

print("First 10 rows of features (X):")
print(X[:10])
print("First 10 labels (y):")
print(y[:10])

#0 = Setosa,1 = Versicolor,2 = Virginica

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model's predictions:", y_pred)
print("Actual answers: ", y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
