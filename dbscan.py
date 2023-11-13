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

#find neighbors
# def find_neighbors(data, point, epsilon):
#     neighbors = []
#     for i, neighbor in enumerate(data):
#         if euclidean_distance(point, neighbor) <= epsilon:
#             neighbors.append(i)
#     return neighbors
      

# find core points
def dbscan(data, epsilon, min_pts):
    n = data.shape[0]
    distances = compute_distance_matrix(data)
    # dictionary of core points
    core_points = {}
    # initializing a list of -1 the length of the dataset 
    cluster_assignment = [-1] * data.shape[0] 

    for i in range(n):
       # neighbors = find_neighbors(data, data.iloc[i], epsilon)
        neighbors = []
        for j in range(n):
            if i != j and euclidean_distance(i, j) <= epsilon:
                neighbors.append(j)
        #print(f'Neighbors: {neighbors}')

        #checking if the point is a core point 
        if len(neighbors) >= min_pts:
            core_points[i] = neighbors
            #added the core point as the key and set neighbors as the values
        #print(f'Core Points: {core_points}')

    # initialize current cluster label
    current_cluster = -1 
    # list of keys
    core_points_keys = list(core_points.keys())
    for d in core_points:
        if cluster_assignment[d] == -1:
            current_cluster += 1 # start a new cluster
            cluster_assignment[d] = current_cluster
            #finding all density connected points
            cluster_assignment = density_connected(d, core_points_keys, core_points, current_cluster, cluster_assignment)
            #cluster_assignme
        
    clusters = {}
    for k in range(current_cluster+1): 
        clusters[k] = [i for i, label in enumerate(cluster_assignment) if label == k]
    
    #outliers
    noise = [i for i, label in enumerate(cluster_assignment) if label == -1]
    border = [i for i in range(data.shape[0]) if i not in core_points and i not in noise]
    #outliers = [i for i in range(data.shape[0]) if i not in core_points and cluster_assignment[i] == -1]
    return clusters, core_points, border, noise


#find density connected points
def density_connected(point, core_points_keys, core_points, cluster_id, cluster_assignment):
    if len(core_points_keys) == 0:
        return cluster_assignment

    for d in core_points[point]:
        cluster_assignment[d] = cluster_id
        #if cluster_assignment[d] == -1:
        if d in core_points_keys:
            core_points_keys.remove(d)
            cluster_assignment = density_connected(d, core_points_keys, core_points, cluster_id, cluster_assignment)
    return cluster_assignment

# def density_connected(data, point, core_points, cluster_id, epsilon, min_samples, distances, cluster_assignment):
#     for d in core_points:
#         if cluster_assignment[d] == -1 and distances[point, d] <= epsilon:
#             #cluster_assignment[d] = cluster_id
#             density_connected(data, d, core_points, cluster_id, epsilon, min_samples, distances, cluster_assignment)
# def dbscan(data, epsilon, minpts):
#     return

# at core points see what points it hits, if non core point then add to cluster but dont continue chain reactions

def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def min_max_scaling(col):
    min_val = col.min()
    max_val = col.max()
    scaled_column = (col - min_val) / (max_val - min_val)
    return scaled_column

def main(filename, epsilon, min_pts):
    with open(filename, 'r') as file:
        first_line = file.readline().strip().split(',')
    use_columns = [i for i, flag in enumerate(first_line) if flag == '1']

    data0 = pd.read_csv(filename, skiprows=1, usecols=use_columns)
    data = normalize_dataframe(data0)
    distance_matrix = compute_distance_matrix(data)
    clusters, core_points, border, noise = dbscan(data, epsilon, min_pts)
    print(data)
    print("Clusters:", clusters)
    print("Core Points:", core_points)
    print("Percentage of Core Points:", len(core_points) / data.shape[0] * 100, "%")
    print("Border Points:", border)
    print("Percentage of Border Points:", len(border) / data.shape[0] * 100, "%")
    print("Noise Points (outliers):", noise)
    #print("Outliers Count:", len(outliers))
    print("Percentage of Noise:", len(noise) / data.shape[0] * 100, "%")


#testing 

filename = '/Users/anaghasikha/Desktop/CSC_466/CSCLab4/mammal_milk.csv'
# with open(filename, 'r') as file:
#         first_line = file.readline().strip().split(',')
# use_columns = [i for i, flag in enumerate(first_line) if flag == '1']
# data0 = pd.read_csv(filename, skiprows=1, usecols=use_columns)
# data = normalize_dataframe(data0)
#df = df.rename(columns={'1':'sepalLength','1.1':'sepalWidth','1.2':'petalLength','1.3':'petalWidth','0':'species'})

# for i in range(data.shape[0]):
#     neighbors = find_neighbors(data, data.iloc[i], epsilon=3)
#     print(neighbors)
main(filename, epsilon=3, min_pts = 4)
#print(compute_distance_matrix(data))


if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 4:
    #     print("Too many inputs!")
    #     sys.exit(1)

    # filename = sys.argv[1]
    # epsilon = float(sys.argv[2])
    # num_points = int(sys.argv[3])

    # main(filename, epsilon=3,min_pts=3)

    # filename = '/Users/anaghasikha/Desktop/CSC_466/CSCLab4/iris.csv'
    # data = pd.read_csv(filename)
    # data_numeric = data.select_dtypes(include=[np.number]).to_numpy()
    # epsilon = 2.0
    # min_samples = 2
    # core_points = find_core_points(data, epsilon, min_samples)
    # print("Core Points:", core_points)

