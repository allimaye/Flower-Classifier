from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import heapq

class KNNClassifier():

    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, k):
        if(k <= 0 or k is None):
            k = 1
        predictions = []
        for row in x_test:
            prediction = self.closest(row, k)
            predictions.append(prediction)
        return predictions

    def closest(self, row, k):
        # Use a max heap to store k closest points.
        max_heap = []
        for i in xrange(len(self.x_train)):
            dist = distance.euclidean(row, x_train[i])
            if (len(max_heap) < k):
                heapq.heappush(max_heap, (dist * -1, i))
                continue
            cmp_dist = max_heap[0][0] * -1
            if (dist < cmp_dist):
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, (dist * -1, i))

        # take majority vote from k closest points
        tally, most_popular, count = {}, None, 0
        for dist, index in max_heap:
            label = y_train[index]
            tally[label] = tally.get(label, 0) + 1
            if (tally[label] > count):
                count = tally[label]
                most_popular = label

        return most_popular



iris = datasets.load_iris()

features = iris.data
labels = iris.target

KNN_classifier = KNNClassifier()

k_accuracy = []
for k in xrange(1, 26):
    accuracy_scores = []
    for i in xrange(5):
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
        KNN_classifier.fit(x_train, y_train)
        predictions = KNN_classifier.predict(x_test, k)
        score = accuracy_score(y_test, predictions)
        accuracy_scores.append(score)

    avg_accuracy = sum(accuracy_scores)/float(len(accuracy_scores))
    k_accuracy.append(avg_accuracy)

best_accuracy, best_index = 0, 0

for index, acc in enumerate(k_accuracy):
    if(acc > best_accuracy):
        best_index, best_accuracy = index, acc


print "Optimal K: %s, Accuracy: %s" % (best_index + 1, best_accuracy)


