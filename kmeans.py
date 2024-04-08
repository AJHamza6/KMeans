import numpy as np
from random import sample

class KMeans:
    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
    
    def initialize_centroids(self, data):
        self.centroids = sample(list(data), self.k)
    
    def assign_clusters(self, data):
        clusters = [[] for _ in range(self.k)]
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)
        return clusters
    
    def update_centroids(self, clusters):
        for i in range(self.k):
            if clusters[i]:  # Avoid division by zero if a cluster is empty
                self.centroids[i] = np.mean(clusters[i], axis=0)
    
    def fit(self, data):
        self.initialize_centroids(data)
        
        for _ in range(self.max_iterations):
            clusters = self.assign_clusters(data)
            prev_centroids = self.centroids.copy()
            self.update_centroids(clusters)
            
            if all(np.array_equal(prev, curr) for prev, curr in zip(prev_centroids, self.centroids)):
                break  # Exit loop if centroids do not change
    
    def predict(self, data):
        predictions = []
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            closest_centroid = distances.index(min(distances))
            predictions.append(closest_centroid)
        return predictions
