import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # For feature scaling
from sklearn.metrics import silhouette_score # For evaluating clustering performance

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        # Randomly select n_clusters data points as initial centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[indices]

    def _assign_to_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X):
        self._initialize_centroids(X)

        for i in range(self.max_iter):
            old_centroids = np.copy(self.centroids)
            self.labels = self._assign_to_clusters(X)
            self.centroids = self._update_centroids(X, self.labels)

            # Check for convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged after {i+1} iterations.")
                break
        else:
            print(f"Max iterations ({self.max_iter}) reached without convergence.")

    def predict(self, X):
        return self._assign_to_clusters(X)

# --- Generate Sample Customer Data ---
np.random.seed(42) # for reproducibility

# Cluster 1: High Spenders, High Frequency
cluster1_spending = np.random.normal(loc=1500, scale=300, size=(100, 1))
cluster1_frequency = np.random.normal(loc=20, scale=5, size=(100, 1))
cluster1_data = np.hstack((cluster1_spending, cluster1_frequency))

# Cluster 2: Medium Spenders, Medium Frequency
cluster2_spending = np.random.normal(loc=500, scale=100, size=(100, 1))
cluster2_frequency = np.random.normal(loc=8, scale=2, size=(100, 1))
cluster2_data = np.hstack((cluster2_spending, cluster2_frequency))

# Cluster 3: Low Spenders, Low Frequency
cluster3_spending = np.random.normal(loc=100, scale=50, size=(100, 1))
cluster3_frequency = np.random.normal(loc=2, scale=1, size=(100, 1))
cluster3_data = np.hstack((cluster3_spending, cluster3_frequency))

customer_data = np.vstack((cluster1_data, cluster2_data, cluster3_data))

# Add some noise/outliers
noise_spending = np.random.uniform(low=50, high=2000, size=(20, 1))
noise_frequency = np.random.uniform(low=1, high=25, size=(20, 1))
noise_data = np.hstack((noise_spending, noise_frequency))

customer_data = np.vstack((customer_data, noise_data))

print(f"Shape of raw customer data: {customer_data.shape}")

# --- Feature Scaling ---
# It's crucial to scale features for K-Means, especially when they have different ranges.
scaler = StandardScaler()
scaled_customer_data = scaler.fit_transform(customer_data)
print(f"Shape of scaled customer data: {scaled_customer_data.shape}")

# --- Apply K-Means Clustering ---
k = 3 # We know there are 3 inherent groups in our synthetic data
kmeans = KMeans(n_clusters=k, max_iter=300, random_state=42)
kmeans.fit(scaled_customer_data)

# Get the cluster assignments
customer_segments = kmeans.labels
centroids = kmeans.centroids

print(f"\nCustomer Segments (first 10): {customer_segments[:10]}")
print(f"Centroids (scaled):\n{centroids}")

# Inverse transform centroids to original scale for better interpretation
original_scale_centroids = scaler.inverse_transform(centroids)
print(f"\nCentroids (original scale):\n{original_scale_centroids}")

# --- Visualize the Clusters ---
plt.figure(figsize=(10, 7))
scatter = plt.scatter(customer_data[:, 0], customer_data[:, 1], c=customer_segments, cmap='viridis', s=50, alpha=0.8)
plt.scatter(original_scale_centroids[:, 0], original_scale_centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Customer Segments based on Purchase History (K-Means Clustering)')
plt.xlabel('Total Spending')
plt.ylabel('Frequency of Visits')
plt.colorbar(scatter, label='Cluster ID')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Evaluate Clustering (Optional, for real-world data where you don't know k) ---
# Silhouette Score: A measure of how similar an object is to its own cluster (cohesion)
# compared to other clusters (separation). Higher score means better defined clusters.
try:
    score = silhouette_score(scaled_customer_data, customer_segments)
    print(f"\nSilhouette Score: {score:.3f}")
except Exception as e:
    print(f"\nCould not calculate Silhouette Score: {e}")
    print("Silhouette score requires at least 2 clusters and each cluster to have at least 2 points.")


# --- How to determine the optimal K (Elbow Method) ---
# This is crucial in real-world scenarios where 'k' is unknown.
wcss = [] # Within-Cluster Sum of Squares
max_k = 10 # Test up to 10 clusters
for i in range(1, max_k + 1):
    kmeans_elbow = KMeans(n_clusters=i, max_iter=300, random_state=42)
    kmeans_elbow.fit(scaled_customer_data)
    # Calculate WCSS
    current_wcss = 0
    for j in range(i):
        cluster_points = scaled_customer_data[kmeans_elbow.labels == j]
        if len(cluster_points) > 0:
            current_wcss += np.sum((cluster_points - kmeans_elbow.centroids[j])**2)
    wcss.append(current_wcss)

plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1, max_k + 1))
plt.show()

print("\nInterpretation of Elbow Method:")
print("Look for the 'elbow' point in the graph where the rate of decrease in WCSS slows down significantly.")
print("This point is often considered a good estimate for the optimal number of clusters (K).")
print("In this example, the elbow is clearly visible at K=3, which matches our synthetic data generation.")
