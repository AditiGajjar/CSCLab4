import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import euclidean_distance, kmeans

def compute_ssd(clusters):
    ssd = 0
    for centroid, points in clusters.items():
        for point in points:
            ssd += euclidean_distance(point, centroid) ** 2
    return ssd

def find_optimal_clusters(data, max_k):
    ssd = []
    K = range(1, max_k + 1)

    for k in K:
        clusters, centroids = kmeans(data, k)
        ssd.append(compute_ssd(clusters))

    plt.figure(figsize=(10, 6))
    plt.plot(K, ssd, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def main(filename, model_type,max_k = None, threshold = None):
    with open(filename, 'r') as file:
        first_line = file.readline().strip().split(',')
    use_columns = [i for i, flag in enumerate(first_line) if flag == '1']
    data = pd.read_csv(filename, skiprows=1, usecols=use_columns)

    # Run the appropriate model
    if model_type.lower() == 'kmeans':
        find_optimal_clusters(data, max_k)

    elif model_type.lower() == 'hclustering':
        pass

if __name__ == "__main__":
    import sys
    max_k = None
    threshold = None

    if len(sys.argv) < 2:
        print("Too many inputs!")
        sys.exit(1)

    filename = sys.argv[1]
    model_type = str(sys.argv[2])

    if model_type.lower() == 'kmeans':
        max_k = int(sys.argv[3])
    elif model_type.lower() == 'hclustering':
        threshold = int(sys.argv[3])

    main(filename, model_type, max_k, threshold)
