import numpy as np
import pandas as pd
from collections import defaultdict


def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def centroids_plusplus(points, k):
    centroids = [points.sample(n=1).values.flatten().tolist()]
    for i in range(1, k):
        distances = points.apply(lambda x: min([euclidean_distance(x, centroid) for centroid in centroids]), axis=1)
        next_centroid = points.sample(weights=distances, n=1).values.flatten().tolist()
        centroids.append(next_centroid)
    return np.array(centroids)


def clusterize(points, centroids):
    clusters = defaultdict(list)
    for index, point in points.iterrows():
        closest_centroid = np.argmin([euclidean_distance(point, centroid) for centroid in centroids])
        clusters[closest_centroid].append(index)
    return clusters

def compute_new_centroids(points, clusters):
    new_centroids = []
    for cluster in clusters.values():
        new_centroids.append(points.loc[cluster].mean().values)
    return new_centroids

def convergence(old_centroids, centroids, threshold):
    total_movement = sum(euclidean_distance(old, new) for old, new in zip(old_centroids, centroids))
    return total_movement < threshold

def compute_centroid_distances(points, centroids):
    distances = defaultdict(list)
    for index, point in points.iterrows():
        centroid_distance = [euclidean_distance(point, centroid) for centroid in centroids]
        distances[index] = centroid_distance
    return distances

def remove_outliers(clusters, centroid_distances, threshold):
    new_clusters = defaultdict(list)
    outliers = []
    for cluster_id, members in clusters.items():
        for member in members:
            if min(centroid_distances[member]) > threshold:
                outliers.append(member)
            else:
                new_clusters[cluster_id].append(member)
    return new_clusters, outliers

def kmeans(points, k, method='random', threshold=0.001):
    centroids = centroids_plusplus(points, k)
    while True:
        old_centroids = centroids
        clusters = clusterize(points, centroids)
        centroids = compute_new_centroids(points, clusters)
        if convergence(old_centroids, centroids, threshold):
            break
    return clusters, centroids

def calculate_cluster_stats(points, cluster, centroid):
    distances = [euclidean_distance(points.iloc[i], centroid) for i in cluster]
    max_distance = max(distances)
    min_distance = min(distances)
    avg_distance = sum(distances) / len(distances)
    sse = sum([d**2 for d in distances])
    return len(cluster), max_distance, min_distance, avg_distance, sse


def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def main(filename, k):
    with open(filename, 'r') as file:
        first_line = file.readline().strip().split(',')
    use_columns = [i for i, flag in enumerate(first_line) if flag == '1']

    df0 = pd.read_csv(filename, skiprows=1, usecols=use_columns)
    df = normalize_dataframe(df0)

    points = df

    clusters, centroids = kmeans(points, k)
    centroid_distances = compute_centroid_distances(points, centroids)
    threshold = np.percentile([min(dist) for dist in centroid_distances.values()], 95)
    clusters, outliers = remove_outliers(clusters, centroid_distances, threshold)
    for i, cluster in enumerate(clusters.values()):
        cluster_size, max_dist, min_dist, avg_dist, sse = calculate_cluster_stats(df, cluster, centroids[i])
        print(f"Cluster {i+1}:")
        print(f"Number of points: {cluster_size}")
        print(f"Centroid coordinates: {centroids[i][0]}, {centroids[i][1]}")
        print(f"Maximum distance to centroid: {max_dist}")
        print(f"Minimum distance to centroid: {min_dist}")
        print(f"Average distance to centroid: {avg_dist}")
        print(f"Sum of Squared Errors: {sse}")
        print()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Too many inputs!")
        sys.exit(1)

    filename = sys.argv[1]
    k = int(sys.argv[2])

    main(filename, k)