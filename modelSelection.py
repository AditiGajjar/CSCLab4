import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans, calculate_cluster_stats, normalize_dataframe
from hclustering import agglomerative_clustering, compute_distance_matrix, cut_dendrogram_at_threshold
import seaborn as sns
from collections import defaultdict



# Computing accuracy
def calculate_total_sse(points, clusters, centroids):
    total_sse = 0
    for i, cluster in clusters.items():
        centroid = centroids[i]
        sse = calculate_cluster_stats(points, cluster, centroid)[4]
        total_sse += sse
    return(total_sse)

# k-means tuning
# if we see a high dip (an elbow), we have found the optimal k
def tune_k(df, max_k):
    sse_values = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        clusters, centroids = kmeans(df, k)
        sse = calculate_total_sse(df, clusters, centroids)
        sse_values.append(sse)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse_values)
    plt.xlabel('Number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.title('SSE vs. Number of Clusters')
    plt.show()

    return sse_values



def tune_height(data, max_threshold):
    sse_values = []
    threshold_values = np.arange(0, max_threshold, 0.01)

    distance_matrix = compute_distance_matrix(data)
    root_dendrogram = agglomerative_clustering(data, distance_matrix)

    for threshold in threshold_values:
        clusters = cut_dendrogram_at_threshold(root_dendrogram, threshold)
        
        # Create a temporary dictionary for clusters
        cluster_dict = {}
        centroids = []

        for i, cluster in enumerate(clusters):
            if not cluster:
                continue


            cluster_indices = [data[(data == point).all(axis=1)].index[0] for point in cluster]
            cluster_dict[i] = cluster_indices

            centroid = np.mean(np.array(cluster), axis=0)
            centroids.append(centroid)

        sse = calculate_total_sse(data, cluster_dict, centroids)
        sse_values.append(sse)

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_values, sse_values)
    plt.xlabel('Threshold')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('SSE vs. Threshold')
    plt.show()

    return threshold_values, sse_values

def plot_clusters(data, clusters, centroids):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # Extend this list if more colors are needed
    if data.shape[1] == 2:
        for i, cluster in clusters.items():
            cluster_points = data.iloc[cluster]
            plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], color=colors[i % len(colors)], label=f'Cluster {i}')
        plt.scatter(centroids[:, 0], centroids[:, 1], color='gold', marker='*', s=200, label='Centroids')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        plt.title("2D Scatter Plot of Clusters")
        plt.legend()
    elif data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, cluster in clusters.items():
            cluster_points = data.iloc[cluster]
            ax.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], cluster_points.iloc[:, 2], color=colors[i % len(colors)], label=f'Cluster {i}')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='gold', marker='*', s=200, label='Centroids')
        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        ax.set_zlabel(data.columns[2])
        plt.title("3D Scatter Plot of Clusters")
        plt.legend()
    else:
        print("Data has more than three features, cannot plot.")
    plt.show()




# Plotting low dimension data
def plot_data(data):
    if data.shape[1] == 2:
        # 2D plot for datasets with two features
        sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1])
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        plt.title("2D Scatter Plot of the Data")
    elif data.shape[1] == 3:
        # 3D plot for datasets with three features
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        ax.set_zlabel(data.columns[2])
        plt.title("3D Scatter Plot of the Data")
    else:
        print("Data has more than three features, cannot plot.")
        return

    plt.show()

def main(filename, model_type,k = None, threshold = None):
    with open(filename, 'r') as file:
        first_line = file.readline().strip().split(',')
    use_columns = [i for i, flag in enumerate(first_line) if flag == '1']
    data0 = pd.read_csv(filename, skiprows=1, usecols=use_columns)
    data = normalize_dataframe(data0)

    plot_data(data)

    # Run the appropriate model
    if model_type.lower() == 'kmeans':
        tune_k(data, k)
        k = int(input("Best k: "))
        print(model_type, ":")
        clusters, centroids = kmeans(data, k)
        plot_clusters(data, clusters, centroids)  # Plotting the clusters
        print(calculate_total_sse(data, clusters, centroids))
        print()

    elif model_type.lower() == 'hclustering':
        tune_height(data, threshold)
        thresh = float(input("Best threshold: "))
        print(model_type, ":")
        distance_matrix = compute_distance_matrix(data)
        root_dendrogram = agglomerative_clustering(data, distance_matrix)
        raw_clusters = cut_dendrogram_at_threshold(root_dendrogram, thresh)
        print("Number of clusters at this threshold:", len(raw_clusters))
        clusters = defaultdict(list)
        centroids = []

        for i, cluster in enumerate(raw_clusters):
            cluster_indices = [data[(data == point).all(axis=1)].index[0] for point in cluster]
            clusters[i] = cluster_indices
            centroid = np.mean(np.array(cluster), axis=0)
            centroids.append(centroid)

        print(calculate_total_sse(data, clusters, centroids))
        print()
    
    # DBSCAN
    else:
        pass

if __name__ == "__main__":
    import sys
    k = None
    threshold = None

    if len(sys.argv) < 2:
        print("Too many inputs!")
        sys.exit(1)

    filename = sys.argv[1]
    model_type = str(sys.argv[2])

    if model_type.lower() == 'kmeans':
        k = int(sys.argv[3])
    elif model_type.lower() == 'hclustering':
        threshold = float(sys.argv[3])

    main(filename, model_type, k, threshold)
