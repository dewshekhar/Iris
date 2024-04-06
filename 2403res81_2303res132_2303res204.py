import pandas as pd  # For data handling
import matplotlib.pyplot as plt  # For plotting
import random  # For random number generation
import numpy as np  # For numerical operations

# Load the dataset from the provided URL
url = "https://raw.githubusercontent.com/dewshekhar/ML-C/main/Iris.csv"
df = pd.read_csv(url)

# Select the features for clustering
selected_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Extract selected features
X = df[selected_features]

# Standardize the data manually
mean = X.mean()
std = X.std()
X_std = (X - mean) / std

# Choose the number of clusters (k=3)
num_clusters = 3

# Initialize cluster medoids randomly
def initialize_medoids(X, k):
    medoids_indices = random.sample(range(len(X)), k)
    medoids = [X.iloc[i].values for i in medoids_indices]
    return medoids

# Initialize medoids
medoids = initialize_medoids(X_std, num_clusters)

# Define a function to calculate Manhattan distance between two points
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

# Define a function to assign data points to the nearest medoid
def assign_to_clusters(X, medoids):
    clusters = {}
    for i in range(len(X)):
        point = X.iloc[i].values
        closest_medoid_index = min(range(len(medoids)), key=lambda j: manhattan_distance(point, medoids[j]))
        if closest_medoid_index in clusters:
            clusters[closest_medoid_index].append(point)
        else:
            clusters[closest_medoid_index] = [point]
    return clusters

# Define a function to update cluster medoids
def update_medoids(clusters):
    medoids = []
    for cluster in clusters.values():
        cluster_distances = [sum(manhattan_distance(point1, point2) for point2 in cluster) for point1 in cluster]
        medoid_index = cluster_distances.index(min(cluster_distances))
        medoids.append(cluster[medoid_index])
    return medoids

# Perform K-Medoids clustering
max_iterations = 100
tolerance = 1e-4  # Convergence threshold

for iteration in range(max_iterations):
    clusters = assign_to_clusters(X_std, medoids)
    new_medoids = update_medoids(clusters)

    # Calculate the change in medoid positions
    medoid_distance = sum(sum(manhattan_distance(new, old) for old in cluster) for new, cluster in zip(new_medoids, clusters.values()))

    if medoid_distance < tolerance:
        break

    medoids = new_medoids

# Assign each data point to a cluster
df['cluster_kmedoids'] = -1  # Initialize with a placeholder value
for cluster_label, points in enumerate(clusters.values()):
    for point in points:
        df.loc[(X_std[selected_features[0]] == point[0]) & 
               (X_std[selected_features[1]] == point[1]), 'cluster_kmedoids'] = cluster_label

# Count data points in each cluster for K-Medoids
count_kmedoids = df['cluster_kmedoids'].value_counts().sort_index()

# Calculate the Sum of Squared Errors (SSE) for K-Medoids
sse_kmedoids = 0
for cluster_label, medoid in enumerate(medoids):
    cluster_points = clusters[cluster_label]
    for point in cluster_points:
        sse_kmedoids += np.sum((point - medoid) ** 2)

# Initialize cluster centroids randomly for K-Means
def initialize_centroids(X, k):
    centroids_indices = random.sample(range(len(X)), k)
    centroids = [X.iloc[i].values for i in centroids_indices]
    return centroids

# Initialize centroids for K-Means
centroids = initialize_centroids(X_std, num_clusters)

# Define a function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Define a function to assign data points to the nearest centroid
def assign_to_clusters(X, centroids):
    clusters = {}
    for i in range(len(X)):
        point = X.iloc[i].values
        closest_centroid_index = min(range(len(centroids)), key=lambda j: euclidean_distance(point, centroids[j]))
        if closest_centroid_index in clusters:
            clusters[closest_centroid_index].append(point)
        else:
            clusters[closest_centroid_index] = [point]
    return clusters

# Define a function to update cluster centroids for K-Means
def update_centroids(clusters):
    centroids = []
    for cluster in clusters.values():
        cluster_mean = np.mean(cluster, axis=0)
        centroids.append(cluster_mean)
    return centroids

# Perform K-Means clustering
for iteration in range(max_iterations):
    clusters = assign_to_clusters(X_std, centroids)
    new_centroids = update_centroids(clusters)

    # Calculate the Euclidean distance between the old and new centroids
    centroid_distance = sum(euclidean_distance(new, old) for new, old in zip(new_centroids, centroids))

    if centroid_distance < tolerance:
        break

    centroids = new_centroids

# Assign each data point to a cluster for K-Means
df['cluster_kmeans'] = -1  # Initialize with a placeholder value
for cluster_label, points in enumerate(clusters.values()):
    for point in points:
        df.loc[(X_std[selected_features[0]] == point[0]) & 
               (X_std[selected_features[1]] == point[1]), 'cluster_kmeans'] = cluster_label

# Count data points in each cluster for K-Means
count_kmeans = df['cluster_kmeans'].value_counts().sort_index()

# Calculate the Sum of Squared Errors (SSE) for K-Means
sse_kmeans = 0
for cluster_label, centroid in enumerate(centroids):
    cluster_data = clusters[cluster_label]
    for point in cluster_data:
        sse_kmeans += np.sum((point - centroid) ** 2)

# Plot K-Medoids clusters for different feature combinations
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
for cluster_label, color in zip(range(num_clusters), ['b', 'g', 'r']):
    cluster_data = df[df['cluster_kmedoids'] == cluster_label]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['SepalWidthCm'], c=color, label=f'Cluster {cluster_label} (K-Medoids)')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.title(f'K-Medoids Clustering (k={num_clusters}) - SSE: {sse_kmedoids:.2f}, Counts: {count_kmedoids.values}')
plt.legend()

plt.subplot(1, 3, 2)
for cluster_label, color in zip(range(num_clusters), ['b', 'g', 'r']):
    cluster_data = df[df['cluster_kmedoids'] == cluster_label]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['PetalLengthCm'], c=color, label=f'Cluster {cluster_label} (K-Medoids)')
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.title(f'K-Medoids Clustering (k={num_clusters}) - SSE: {sse_kmedoids:.2f}, Counts: {count_kmedoids.values}')
plt.legend()

plt.subplot(1, 3, 3)
for cluster_label, color in zip(range(num_clusters), ['b', 'g', 'r']):
    cluster_data = df[df['cluster_kmedoids'] == cluster_label]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['PetalWidthCm'], c=color, label=f'Cluster {cluster_label} (K-Medoids)')
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalWidthCm')
plt.title(f'K-Medoids Clustering (k={num_clusters}) - SSE: {sse_kmedoids:.2f}, Counts: {count_kmedoids.values}')
plt.legend()

plt.tight_layout()
plt.show()

# Plot K-Means clusters for different feature combinations
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
for cluster_label, color in zip(range(num_clusters), ['b', 'g', 'r']):
    cluster_data = df[df['cluster_kmeans'] == cluster_label]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['SepalWidthCm'], c=color, label=f'Cluster {cluster_label} (K-Means)')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.title(f'K-Means Clustering (k={num_clusters}) - SSE: {sse_kmeans:.2f}, Counts: {count_kmeans.values}')
plt.legend()

plt.subplot(1, 3, 2)
for cluster_label, color in zip(range(num_clusters), ['b', 'g', 'r']):
    cluster_data = df[df['cluster_kmeans'] == cluster_label]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['PetalLengthCm'], c=color, label=f'Cluster {cluster_label} (K-Means)')
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.title(f'K-Means Clustering (k={num_clusters}) - SSE: {sse_kmeans:.2f}, Counts: {count_kmeans.values}')
plt.legend()

plt.subplot(1, 3, 3)
for cluster_label, color in zip(range(num_clusters), ['b', 'g', 'r']):
    cluster_data = df[df['cluster_kmeans'] == cluster_label]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['PetalWidthCm'], c=color, label=f'Cluster {cluster_label} (K-Means)')
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalWidthCm')
plt.title(f'K-Means Clustering (k={num_clusters}) - SSE: {sse_kmeans:.2f}, Counts: {count_kmeans.values}')
plt.legend()

plt.tight_layout()
plt.show()
