"""
Code Clustering Module

Clusters code snippets by structural features and/or
semantic embeddings to discover programming patterns.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    KMEANS_N_CLUSTERS_RANGE,
    KMEANS_DEFAULT_CLUSTERS,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    HIERARCHICAL_LINKAGE,
    PCA_COMPONENTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    labels: np.ndarray
    n_clusters: int
    algorithm: str
    metrics: Dict[str, float]
    cluster_centers: Optional[np.ndarray] = None
    
    def get_cluster_indices(self, cluster_id: int) -> np.ndarray:
        """Get indices of samples in a specific cluster."""
        return np.where(self.labels == cluster_id)[0]
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class CodeClusterer:
    """
    Clustering engine for code analysis.
    
    Supports multiple algorithms:
    - K-Means: Fast, requires specifying K
    - DBSCAN: Density-based, handles noise, finds K automatically
    - Hierarchical: Good for visualization, produces dendrograms
    """
    
    def __init__(self, normalize: bool = True, reduce_dims: bool = True):
        """
        Initialize clusterer.
        
        Args:
            normalize: Whether to standardize features before clustering
            reduce_dims: Whether to reduce dimensionality for high-dim embeddings
        """
        self.normalize = normalize
        self.reduce_dims = reduce_dims
        self.scaler = StandardScaler() if normalize else None
        self.pca = None
        self._fitted_features = None
    
    def _preprocess(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features for clustering."""
        X = features.copy()
        
        # Normalize
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Reduce dimensions if needed (for embeddings)
        if self.reduce_dims and X.shape[1] > PCA_COMPONENTS:
            n_components = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
            self.pca = PCA(n_components=n_components)
            X = self.pca.fit_transform(X)
            logger.info(f"Reduced dimensions: {features.shape[1]} -> {X.shape[1]}")
        
        self._fitted_features = X
        return X
    
    def kmeans(
        self,
        features: np.ndarray,
        n_clusters: int = KMEANS_DEFAULT_CLUSTERS,
        **kwargs
    ) -> ClusterResult:
        """
        Cluster using K-Means algorithm.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
            **kwargs: Additional KMeans parameters
            
        Returns:
            ClusterResult with labels and metrics
        """
        X = self._preprocess(features)
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=kwargs.get('max_iter', KMEANS_MAX_ITER),
            n_init=kwargs.get('n_init', KMEANS_N_INIT),
            random_state=42,
        )
        
        labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        metrics = self._calculate_metrics(X, labels)
        metrics['inertia'] = kmeans.inertia_
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            algorithm='kmeans',
            metrics=metrics,
            cluster_centers=kmeans.cluster_centers_,
        )
    
    def dbscan(
        self,
        features: np.ndarray,
        eps: float = DBSCAN_EPS,
        min_samples: int = DBSCAN_MIN_SAMPLES,
    ) -> ClusterResult:
        """
        Cluster using DBSCAN algorithm.
        
        DBSCAN finds clusters of varying shapes and identifies outliers.
        Labels of -1 indicate noise/outliers.
        
        Args:
            features: Feature matrix
            eps: Maximum distance between samples in same neighborhood
            min_samples: Minimum samples in a neighborhood for core point
            
        Returns:
            ClusterResult with labels and metrics
        """
        X = self._preprocess(features)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Number of clusters (excluding noise labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calculate metrics (only if we have valid clusters)
        if n_clusters >= 2:
            # Filter out noise for metrics
            mask = labels != -1
            if mask.sum() > n_clusters:
                metrics = self._calculate_metrics(X[mask], labels[mask])
            else:
                metrics = {}
        else:
            metrics = {}
        
        metrics['n_noise'] = np.sum(labels == -1)
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            algorithm='dbscan',
            metrics=metrics,
        )
    
    def hierarchical(
        self,
        features: np.ndarray,
        n_clusters: int = KMEANS_DEFAULT_CLUSTERS,
        linkage: str = HIERARCHICAL_LINKAGE,
    ) -> ClusterResult:
        """
        Cluster using Agglomerative Hierarchical Clustering.
        
        Good for visualization with dendrograms.
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            
        Returns:
            ClusterResult with labels and metrics
        """
        X = self._preprocess(features)
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        
        labels = hierarchical.fit_predict(X)
        metrics = self._calculate_metrics(X, labels)
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            algorithm='hierarchical',
            metrics=metrics,
        )
    
    def _calculate_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters >= 2 and n_clusters < len(X):
            try:
                metrics['silhouette'] = silhouette_score(X, labels)
            except:
                pass
            
            try:
                metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            except:
                pass
        
        return metrics
    
    def find_optimal_k(
        self,
        features: np.ndarray,
        k_range: Tuple[int, int] = KMEANS_N_CLUSTERS_RANGE,
    ) -> Tuple[int, List[Dict]]:
        """
        Find optimal number of clusters using elbow method and silhouette.
        
        Args:
            features: Feature matrix
            k_range: Range of k values to try (min, max)
            
        Returns:
            Tuple of (optimal_k, list of results for each k)
        """
        X = self._preprocess(features)
        
        results = []
        best_k = k_range[0]
        best_silhouette = -1
        
        for k in range(k_range[0], min(k_range[1] + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            result = {
                'k': k,
                'inertia': kmeans.inertia_,
            }
            
            if k >= 2:
                try:
                    sil = silhouette_score(X, labels)
                    result['silhouette'] = sil
                    if sil > best_silhouette:
                        best_silhouette = sil
                        best_k = k
                except:
                    pass
            
            results.append(result)
        
        logger.info(f"Optimal k = {best_k} (silhouette = {best_silhouette:.4f})")
        return best_k, results


def cluster_codes(
    features: np.ndarray,
    algorithm: str = 'kmeans',
    n_clusters: Optional[int] = None,
    auto_k: bool = True,
    **kwargs
) -> ClusterResult:
    """
    Convenience function to cluster code features.
    
    Args:
        features: Feature matrix (structural features or embeddings)
        algorithm: 'kmeans', 'dbscan', or 'hierarchical'
        n_clusters: Number of clusters (ignored for DBSCAN)
        auto_k: Automatically find optimal K for K-Means
        **kwargs: Additional algorithm parameters
        
    Returns:
        ClusterResult
    """
    clusterer = CodeClusterer()
    
    if algorithm == 'dbscan':
        return clusterer.dbscan(features, **kwargs)
    
    if algorithm == 'hierarchical':
        n = n_clusters or KMEANS_DEFAULT_CLUSTERS
        return clusterer.hierarchical(features, n_clusters=n, **kwargs)
    
    # K-Means (default)
    if n_clusters is None and auto_k:
        optimal_k, _ = clusterer.find_optimal_k(features)
        n_clusters = optimal_k
    elif n_clusters is None:
        n_clusters = KMEANS_DEFAULT_CLUSTERS
    
    return clusterer.kmeans(features, n_clusters=n_clusters, **kwargs)


if __name__ == "__main__":
    # Test clustering
    np.random.seed(42)
    
    # Generate synthetic code features
    n_samples = 100
    n_features = 25
    
    # Create 3 distinct clusters
    cluster1 = np.random.randn(30, n_features) + np.array([2, 0] + [0] * (n_features - 2))
    cluster2 = np.random.randn(40, n_features) + np.array([-2, 2] + [0] * (n_features - 2))
    cluster3 = np.random.randn(30, n_features) + np.array([0, -2] + [0] * (n_features - 2))
    
    features = np.vstack([cluster1, cluster2, cluster3])
    
    print("Testing K-Means...")
    result = cluster_codes(features, algorithm='kmeans', auto_k=True)
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Sizes: {result.get_cluster_sizes()}")
    print(f"  Silhouette: {result.metrics.get('silhouette', 'N/A'):.4f}")
    
    print("\nTesting DBSCAN...")
    result = cluster_codes(features, algorithm='dbscan', eps=1.5)
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Sizes: {result.get_cluster_sizes()}")
    print(f"  Noise points: {result.metrics.get('n_noise', 0)}")
    
    print("\nTesting Hierarchical...")
    result = cluster_codes(features, algorithm='hierarchical', n_clusters=3)
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Sizes: {result.get_cluster_sizes()}")
