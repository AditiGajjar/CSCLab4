import numpy as np
import pandas as pd
import json
from kmeans import calculate_cluster_stats

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def compute_distance_matrix(data):
    n = len(data)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance = euclidean_distance(data.iloc[i], data.iloc[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

def closest_clusters(distance_matrix):
    min_dist = np.inf
    pair = (-1, -1)
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            if distance_matrix[i][j] < min_dist:
                min_dist = distance_matrix[i][j]
                pair = (i, j)
    return pair, min_dist

def agglomerative_clustering(data, distance_matrix):
    n = len(data)
    clusters = [{"type": "leaf", "height": 0.0, "data": [data.iloc[i].tolist()]} for i in range(n)]
    dendrogram = []

    while len(clusters) > 1:
        (i, j), dist = closest_clusters(distance_matrix)

        new_cluster = {"type": "node", "height": dist, "nodes": [clusters[i], clusters[j]]}
        dendrogram.append(new_cluster)
        clusters.append(new_cluster)

        # Update distance matrix for the new cluster
        new_distances = np.full((len(distance_matrix) + 1,), np.inf)
        for k in range(len(distance_matrix)):
            if k not in [i, j]:
                new_distances[k] = min(distance_matrix[k, i], distance_matrix[k, j])
        new_distances[-1] = 0  # Distance of the new cluster to itself

        # Expand the distance matrix and set the new distances
        distance_matrix = np.vstack((distance_matrix, new_distances[:-1]))
        new_distances_column = np.append(new_distances[:-1], [0])
        distance_matrix = np.column_stack((distance_matrix, new_distances_column))

        # Remove the merged clusters from the distance matrix and clusters list
        distance_matrix = np.delete(distance_matrix, [i, j], axis=0)
        distance_matrix = np.delete(distance_matrix, [i, j], axis=1)
        del clusters[max(i, j)]  # Order of deletion matters
        del clusters[min(i, j)]

    return dendrogram[-1] if dendrogram else None  

def cut_dendrogram_at_threshold(node, threshold):
    if node['type'] == 'leaf':
        return [[node['data'][0]]]
    elif node['type'] in ['node', 'root']:
        if node['height'] <= threshold:
            return [get_all_points(node)] 
        else:
            return cut_dendrogram_at_threshold(node['nodes'][0], threshold) + \
                   cut_dendrogram_at_threshold(node['nodes'][1], threshold)
    else:
        return []

def get_all_points(node):
    if node['type'] == 'leaf':
        return node['data']
    else:
        return get_all_points(node['nodes'][0]) + get_all_points(node['nodes'][1])


def main(filename, threshold=None):
    with open(filename, 'r') as file:
        first_line = file.readline().strip().split(',')
    use_columns = [i for i, flag in enumerate(first_line) if flag == '1']
    data = pd.read_csv(filename, skiprows=1, usecols=use_columns)
    distance_matrix = compute_distance_matrix(data)

    root_dendrogram = agglomerative_clustering(data, distance_matrix)
    if root_dendrogram:
        root_dendrogram["type"] = "root" 
        dendrogram_json = json.dumps(root_dendrogram, indent=4)
        with open('dendrogram.json', 'w') as f:
            json.dump(root_dendrogram, f, indent=4)
    if threshold is not None:
            clusters_at_threshold = cut_dendrogram_at_threshold(root_dendrogram, threshold)
            i = 1
            for cluster in clusters_at_threshold:
                cluster_df = pd.DataFrame(cluster)
                centroid= list(cluster_df.mean())
                cluster_size, max_dist, min_dist, avg_dist, sse  = calculate_cluster_stats(data, cluster_df.index.tolist(), centroid)
                print(f"Cluster {i}:")
                print(f"Number of points: {cluster_size}")
                print(f"Maximum distance to centroid: {max_dist}")
                print(f"Minimum distance to centroid: {min_dist}")
                print(f"Average distance to centroid: {avg_dist}")
                print(f"Sum of Squared Errors: {sse}")
                print()
                i+= 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Too many inputs!")
        sys.exit(1)

    filename = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None

    main(filename, threshold)
