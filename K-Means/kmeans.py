import numpy as np
import pandas as pd


def initialize_centroids(data, k):
    """
    Initializes k centroids for the k-means algorithm using random sampling of the data.
    Args:
        data: the input dataset
        k: the number of clusters to create
    Returns:
        centroids: a numpy array containing the coordinates of the k centroids
    """
    centroids = np.zeros((k, data.shape[1] - 1))
    for i in range(k):
        centroids[i] = data.sample().iloc[:, 1:5].values
    return centroids


def manhattan_distance(x, y):
    """
    Calculates the Manhattan distance between two vectors.
    Args:
        x: the first vector
        y: the second vector
    Returns:
        the Manhattan distance between x and y
    """
    return np.sum(np.abs(x - y))


def assign_clusters(data, centroids):
    """
    Assigns each data point to its closest centroid.
    Args:
        data: the input dataset
        centroids: a numpy array containing the coordinates of the k centroids
    Returns:
        cluster_assignments: a numpy array containing the index of the closest centroid for each data point
    """
    distances = np.zeros((data.shape[0], centroids.shape[0]))
    for i in range(data.shape[0]):
        for j in range(centroids.shape[0]):
            distances[i, j] = manhattan_distance(data.iloc[i, 1:5], centroids[j])
    return np.argmin(distances, axis=1)


def update_centroids(data, cluster_assignments, k):
    """
    Updates the position of each centroid based on the mean position of its assigned data points.
    Args:
        data: the input dataset
        cluster_assignments: a numpy array containing the index of the closest centroid for each data point
        k: the number of clusters
    Returns:
        centroids: a numpy array containing the updated coordinates of the k centroids
    """
    centroids = np.zeros((k, data.shape[1] - 1))
    for j in range(k):
        centroids[j] = np.mean(data.iloc[:, 1:5][cluster_assignments == j], axis=0)
    return centroids


def detect_outliers(data, centroids, clusters, threshold):
    """
    Detects outliers in the dataset based on their distance from their assigned centroid.
    Args:
        data: the input dataset
        centroids: a numpy array containing the coordinates of the k centroids
        clusters: a numpy array containing the index of the closest centroid for each data point
        threshold: the maximum distance a data point can be from its centroid before it is considered an outlier
    Returns:
        outliers: a list containing the indices of the detected outliers in the dataset
    """
    outliers = []
    for i in range(len(clusters)):
        distances = manhattan_distance(data.iloc[i, 1:5], centroids[clusters[i]])
        if distances > threshold:
            outliers.append(i)
    return outliers


def kmeans(data, k, max_iter=100):
    """
    Performs the k-means clustering algorithm on the input dataset.
    Args:
        data: the input dataset
        k: the number of clusters to create
        max_iter: the maximum number of iterations to run the algorithm
    Returns:
        cluster_assignments: a numpy array containing the index of the closest centroid for each data point
        centroids: a numpy array containing the coordinates of the k centroids
    """
    centroids = initialize_centroids(data, k)
    cluster_assignments = np.zeros(data.shape[0])
    prev_cluster_assignments = None
    for iter in range(max_iter):
        cluster_assignments = assign_clusters(data, centroids)
        centroids = update_centroids(data, cluster_assignments, k)
        if prev_cluster_assignments is not None and np.array_equal(prev_cluster_assignments, cluster_assignments):
            break
        prev_cluster_assignments = np.copy(cluster_assignments)

    return cluster_assignments, centroids


# Main
data = pd.read_csv('crime_data.csv')
X = data[['Murder', 'Assault', 'Rape', 'UrbanPopulation']].values
k = int(input("Enter the number of clusters: "))
threshold = int(input("Enter the threshold to detect the outliers: "))
cluster_assignments, centroids = kmeans(data, k)

for j in range(k):
    print("Cluster ", j + 1, ": ", data.iloc[cluster_assignments == j, :])

outlier_indices_scratch = detect_outliers(data, centroids, cluster_assignments, threshold=threshold)
print("\nOutliers detected: ")
print(data.iloc[outlier_indices_scratch])
