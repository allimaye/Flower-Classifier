from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

features = iris.data
labels = iris.target

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size= 0.3)

KNN_classifier = KNeighborsClassifier()

KNN_classifier.fit(x_train, y_train)

predictions = KNN_classifier.predict(x_test)
print accuracy_score(y_test, predictions)

