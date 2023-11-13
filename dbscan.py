import sys
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

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

# find core points
def dbscan(data, epsilon, min_pts):

    distances = compute_distance_matrix(data)
    n = distances.shape[0]
    # dictionary of core points
    core_points = {}
    # initializing a list of -1 the length of the dataset 
    cluster_assignment = [0] * data.shape[0] 

    for i in range(n):
        neighbors = []
        for j in range(n):
            if i != j and distances[i][j] <= epsilon:
                neighbors.append(j)

        #checking if the point is a core point 
        if len(neighbors) >= min_pts:
            core_points[i] = neighbors
            #added the core point as the key and set neighbors as the values

    # initialize current cluster label
    current_cluster = 0 
    # list of keys
    core_points_keys = list(core_points.keys())

    for d in core_points:
        if cluster_assignment[d] == 0:
            current_cluster += 1 # start a new cluster
            cluster_assignment[d] = current_cluster
            #finding all density connected points
            cluster_assignment = density_connected(d, core_points_keys, core_points, current_cluster, cluster_assignment)
        
    clusters = {}
    for k in range(1,current_cluster+1): 
        clusters[k] = [i for i, label in enumerate(cluster_assignment) if label == k]
    
    #outliers
    noise = [i for i, label in enumerate(cluster_assignment) if label == 0]
    border = [i for i in range(data.shape[0]) if i not in core_points and i not in noise]
    
    return clusters, core_points, border, noise


#find density connected points
def density_connected(point, core_points_keys, core_points, cluster_id, cluster_assignment):
    if len(core_points_keys) == 0:
        return cluster_assignment

    for d in core_points[point]:
        cluster_assignment[d] = cluster_id
        #if cluster_assignment[d] == 0:
        if d in core_points_keys:
            core_points_keys.remove(d)
            cluster_assignment = density_connected(d, core_points_keys, core_points, cluster_id, cluster_assignment)
    return cluster_assignment

def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


def main(filename, epsilon, min_pts):
    with open(filename, 'r') as file:
        first_line = file.readline().strip().split(',')
    use_columns = [i for i, flag in enumerate(first_line) if flag == '1']

    data0 = pd.read_csv(filename, skiprows=1, usecols=use_columns)
    data = normalize_dataframe(data0)
    distance_matrix = compute_distance_matrix(data)
    clusters, core_points, border, noise = dbscan(data, epsilon, min_pts)

    total_sse = 0
    for cluster, points in clusters.items():
        center = data.iloc[points].mean(axis=0).to_list()
        distance = [euclidean_distance(data.iloc[x], center) for x in points]
        cluster_sse = np.sum(np.array(distance)**2)  # Calculate SSE for the current cluster
        total_sse += cluster_sse
        print(f"Cluster: {cluster}")
        print(f"Num of Points: {len(points)}")
        print("Center:", ", ".join(str(i) for i in center))
        print(f"Max Dist. to Center: {np.max(distance)}")
        print(f"Min Dist. to Center: {np.min(distance)}")
        print(f"Avg Dist. to Center: {np.mean(distance)}")
        print(f"SSE for Cluster: {cluster_sse} \n")
    print("Total SSE:", total_sse)
    print("Core Points:", core_points)
    print("Percentage of Core Points:", len(core_points) / data.shape[0] * 100, "%")
    print()
    print("Border Points:", border)
    print("Percentage of Border Points:", len(border) / data.shape[0] * 100, "%")
    print()
    print("Noise Points (outliers):", noise)
    print("Percentage of Noise:", len(noise) / data.shape[0] * 100, "%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Too many inputs!")
        sys.exit(1)

    filename = sys.argv[1]
    epsilon = float(sys.argv[2])
    min_pts = int(sys.argv[3])

    main(filename, epsilon, min_pts)

# python3 dbscan.py 4clusters.csv 0.14 4
# python3 dbscan.py mammal_milk.csv 0.15 1
# python3 dbscan.py planets.csv 0.3 3
# python3 dbscan.py iris.csv 0.15 2
# python3 dbscan.py AccidentsSet03.csv 0.4 3