import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


def euclidean_distance(instance1, instance2):
    """
    Computes the Euclidean distance between two instances.

    Args:
        instance1 (numpy.ndarray): First instance.
        instance2 (numpy.ndarray): Second instance.

    Returns:
        float: The Euclidean distance between the two instances.

    """
    distance = np.sqrt(np.sum((instance1 - instance2) ** 2))
    return distance


def knn(test_instance, x_train, y_train, k):
    """
    Performs k-nearest neighbors classification for a test instance.

    Args:
        test_instance (numpy.ndarray): Test instance for classification.
        x_train (numpy.ndarray): Training data matrix of shape (m, n), where m is the number of training examples
                                  and n is the number of features.
        y_train (numpy.ndarray): Target values of the training examples, of shape (m,).
        k (int): Number of nearest neighbors to consider for classification.

    Returns:
        Any: The predicted label for the test instance.

    """
    # compute distances
    distances = [euclidean_distance(test_instance, train_instance) for train_instance in x_train]

    # get closest k
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]

    # voting
    most_common = Counter(k_nearest_labels).most_common()
    return most_common[0][0]


def predict_from_scratch(x_test, x_train, y_train, k):
    """
    Performs predictions for multiple test instances using the k-nearest neighbors algorithm.

    Args:
        x_test (numpy.ndarray): Test data matrix of shape (m, n), where m is the number of test instances
                                and n is the number of features.
        x_train (numpy.ndarray): Training data matrix of shape (m, n), where m_train is the number of training examples
                                  and n is the number of features.
        y_train (numpy.ndarray): Target values of the training examples, of shape (m,).
        k (int): Number of nearest neighbors to consider for classification.

    Returns:
        list: A list containing the predicted labels for the test instances.

    """
    predictions = [knn(x, x_train, y_train, k) for x in x_test]
    return predictions


# Main
df = pd.read_csv('BankNote_Authentication.csv')
print(df.head())

x = df.drop('class', axis=1)
y = df['class']

# Standard Scaler Normalizer
x_normalized = (x - x.mean()) / x.std()

# Convert data to numpy array
x_normalized = x_normalized.to_numpy()
y = y.to_numpy()

# Splitting Data
X_train, X_test, Y_train, Y_test = train_test_split(x_normalized, y, test_size=0.3)

for i in range(1, 10):
    # Trying different number of K
    correct = 0
    y_pred_scratch = predict_from_scratch(X_test, X_train, Y_train, i)
    accuracy = np.mean(y_pred_scratch == Y_test) * 100
    correct = sum(y_pred_scratch == Y_test)
    print('k value = ', i)
    print('Number of correctly classified instances', correct, 'Total number of instances', len(y_pred_scratch))
    print('Accuracy = ', accuracy)

print("===========================================================================")

print("Checking SKlearn")
for i in range(1, 10):
    # Trying different number of K
    correct = 0
    model = KNeighborsClassifier(n_neighbors=i)
    knn2 = model.fit(X_train, Y_train)
    y_pred = knn2.predict(X_test)
    accuracy = np.mean(y_pred == Y_test) * 100
    correct = sum(y_pred == Y_test)
    print('k value = ', i)
    print('Number of correctly classified instances', correct, 'Total number of instances', len(y_pred))
    print('Accuracy = ', accuracy)
