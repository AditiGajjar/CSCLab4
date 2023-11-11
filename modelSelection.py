import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import euclidean_distance, kmeans, calculate_cluster_stats
from hclustering import agglomerative_clustering, compute_distance_matrix, cut_dendrogram_at_threshold
import seaborn as sns
from sklearn.preprocessing import StandardScaler




# HCLUSTERING

def hypertune_threshold(data, max_height, step):
    heights = range(0, max_height, step)
    dbi_scores = []

    # Perform hierarchical clustering
    distance_matrix = compute_distance_matrix(data)
    dendrogram_data = agglomerative_clustering(data, distance_matrix)

    for height in heights:
        clusters = cut_dendrogram_at_threshold(dendrogram_data, height)
        dbi_score = calculate_davies_bouldin_index(clusters)
        dbi_scores.append(dbi_score)

    # Plotting DBI scores
    plt.plot(heights, dbi_scores, 'bx-')
    plt.xlabel('Height Threshold')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Hypertuning Threshold in Hierarchical Clustering')
    plt.show()

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


# Plotting low dimension data
def plot_data_for_cluster_analysis(data):
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
    data = pd.read_csv(filename, skiprows=1, usecols=use_columns)

    plot_data_for_cluster_analysis(data)

    # Run the appropriate model
    if model_type.lower() == 'kmeans':
        tune_k(data, k)
        k = int(input("Best k: "))
        print(model_type, ":")
        clusters, centroids = kmeans(data, k)
        print(calculate_total_sse(data, clusters, centroids))
        print()

    elif model_type.lower() == 'hclustering':
        hypertune_threshold(data, threshold, 0.5)

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
        threshold = int(sys.argv[3])

    main(filename, model_type, k, threshold)
