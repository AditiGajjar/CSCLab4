import sys
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def dbscan(data, eps, minpts):
    return

# find core points
def find_neighbors(data, point, eps):
    neighbors = []
    for i, neighbor in enumerate(data):
        if euclidean_distance(point, neighbor) <= eps:
            neighbors.append(i)
    return neighbors







# at core points see what points it hits, if non core point then add to cluseter but dont continue chain reactions














def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def main(filename, k):
    with open(filename, 'r') as file:
        first_line = file.readline().strip().split(',')
    use_columns = [i for i, flag in enumerate(first_line) if flag == '1']

    data0 = pd.read_csv(filename, skiprows=1, usecols=use_columns)
    data = normalize_dataframe(data0)
    distance_matrix = compute_distance_matrix(data)



df = pd.read_csv('/Users/anaghasikha/Desktop/CSC_466/CSCLab4/iris.csv')
df = df.rename(columns={'1':'sepalLength','1.1':'sepalWidth','1.2':'petalLength','1.3':'petalWidth','0':'species'})
sl, pl = df.sepalLength, df.petalLength
plt.scatter(sl,pl)



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Too many inputs!")
        sys.exit(1)

    filename = sys.argv[1]
    e = int(sys.argv[2])
    numPoints = int(sys.argv[3])

    main(filename, k)

