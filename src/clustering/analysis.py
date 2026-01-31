"""
Cluster Analysis Module

Provides tools to interpret and summarize clustering results,
including representative samples, cluster characteristics, and comparisons.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .clusterer import ClusterResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    """Summary information about a single cluster."""
    cluster_id: int
    size: int
    percentage: float
    representative_indices: List[int]
    feature_means: Optional[Dict[str, float]] = None
    feature_stds: Optional[Dict[str, float]] = None
    description: str = ""


def analyze_clusters(
    result: ClusterResult,
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_representatives: int = 3,
) -> List[ClusterSummary]:
    """
    Analyze clustering results and generate summaries for each cluster.
    
    Args:
        result: ClusterResult from clustering
        features: Original feature matrix
        feature_names: Names of features
        n_representatives: Number of representative samples per cluster
        
    Returns:
        List of ClusterSummary objects
    """
    summaries = []
    total_samples = len(result.labels)
    
    # Get unique cluster IDs (excluding noise -1)
    cluster_ids = sorted(set(result.labels))
    if -1 in cluster_ids:
        cluster_ids.remove(-1)
    
    for cluster_id in cluster_ids:
        indices = result.get_cluster_indices(cluster_id)
        cluster_features = features[indices]
        
        # Calculate statistics
        means = np.mean(cluster_features, axis=0)
        stds = np.std(cluster_features, axis=0)
        
        # Create feature dictionaries
        if feature_names:
            feature_means = dict(zip(feature_names, means.tolist()))
            feature_stds = dict(zip(feature_names, stds.tolist()))
        else:
            feature_means = {f"f{i}": v for i, v in enumerate(means)}
            feature_stds = {f"f{i}": v for i, v in enumerate(stds)}
        
        # Find representative samples (closest to centroid)
        # Note: cluster_centers may be in different space (e.g., PCA-reduced)
        # than the features passed for labeling. Handle gracefully.
        if result.cluster_centers is not None and result.cluster_centers.shape[1] == cluster_features.shape[1]:
            centroid = result.cluster_centers[cluster_id]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
        else:
            # Use cluster mean as centroid if official centers don't match dimensions
            centroid = means
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
        
        closest_in_cluster = np.argsort(distances)[:n_representatives]
        representative_indices = indices[closest_in_cluster].tolist()
        
        # Generate description
        description = _generate_cluster_description(
            cluster_id, 
            feature_means, 
            feature_stds,
            feature_names
        )
        
        summary = ClusterSummary(
            cluster_id=cluster_id,
            size=len(indices),
            percentage=len(indices) / total_samples * 100,
            representative_indices=representative_indices,
            feature_means=feature_means,
            feature_stds=feature_stds,
            description=description,
        )
        summaries.append(summary)
    
    return summaries


def _generate_cluster_description(
    cluster_id: int,
    means: Dict[str, float],
    stds: Dict[str, float],
    feature_names: Optional[List[str]] = None,
) -> str:
    """Generate a human-readable description of a cluster."""
    descriptions = []
    
    # Key features to highlight
    key_features = {
        'loop_count': ('loops', 'many loops', 'few loops'),
        'max_nesting_depth': ('nesting', 'deeply nested', 'shallow'),
        'has_recursion': ('recursion', 'recursive', 'iterative'),
        'try_except_count': ('exception handling', 'heavy error handling', 'minimal error handling'),
        'function_count': ('functions', 'many functions', 'few functions'),
        'cyclomatic_complexity': ('complexity', 'complex', 'simple'),
        'comprehension_count': ('comprehensions', 'pythonic', 'traditional'),
    }
    
    for feature, (name, high_desc, low_desc) in key_features.items():
        if feature in means:
            value = means[feature]
            if feature == 'has_recursion':
                if value > 0.5:
                    descriptions.append('recursive patterns')
            elif feature in ['loop_count', 'function_count']:
                if value > 5:
                    descriptions.append(high_desc)
                elif value < 2:
                    descriptions.append(low_desc)
            elif feature == 'max_nesting_depth':
                if value > 4:
                    descriptions.append(high_desc)
                elif value < 2:
                    descriptions.append(low_desc)
            elif feature == 'cyclomatic_complexity':
                if value > 8:
                    descriptions.append('high complexity')
                elif value < 3:
                    descriptions.append('low complexity')
    
    if descriptions:
        return f"Cluster {cluster_id}: " + ", ".join(descriptions)
    return f"Cluster {cluster_id}: general code patterns"


def get_cluster_summary(
    result: ClusterResult,
    features: np.ndarray,
    cluster_id: int,
    feature_names: Optional[List[str]] = None,
) -> Optional[ClusterSummary]:
    """
    Get summary for a specific cluster.
    
    Args:
        result: ClusterResult from clustering
        features: Original feature matrix
        cluster_id: Which cluster to summarize
        feature_names: Names of features
        
    Returns:
        ClusterSummary or None if cluster doesn't exist
    """
    if cluster_id not in result.labels:
        return None
    
    summaries = analyze_clusters(result, features, feature_names)
    for summary in summaries:
        if summary.cluster_id == cluster_id:
            return summary
    return None


def compare_clusters(
    summaries: List[ClusterSummary],
    feature_name: str,
) -> Dict[int, float]:
    """
    Compare clusters by a specific feature.
    
    Args:
        summaries: List of ClusterSummary objects
        feature_name: Feature to compare
        
    Returns:
        Dictionary mapping cluster_id to feature mean
    """
    comparison = {}
    for summary in summaries:
        if summary.feature_means and feature_name in summary.feature_means:
            comparison[summary.cluster_id] = summary.feature_means[feature_name]
    return comparison


def identify_outlier_cluster(
    summaries: List[ClusterSummary],
    threshold_features: Optional[Dict[str, float]] = None,
) -> List[int]:
    """
    Identify clusters that might contain problematic code.
    
    Args:
        summaries: List of ClusterSummary objects
        threshold_features: Dict of feature -> threshold for "problematic"
        
    Returns:
        List of cluster IDs that might contain issues
    """
    if threshold_features is None:
        threshold_features = {
            'max_nesting_depth': 5,
            'bare_except_count': 0.5,
            'cyclomatic_complexity': 10,
        }
    
    problematic = []
    
    for summary in summaries:
        if summary.feature_means is None:
            continue
        
        is_problematic = False
        for feature, threshold in threshold_features.items():
            if feature in summary.feature_means:
                if summary.feature_means[feature] > threshold:
                    is_problematic = True
                    break
        
        if is_problematic:
            problematic.append(summary.cluster_id)
    
    return problematic


def print_cluster_report(summaries: List[ClusterSummary]):
    """Print a formatted cluster analysis report."""
    print("\n" + "=" * 60)
    print("CLUSTER ANALYSIS REPORT")
    print("=" * 60)
    
    for summary in summaries:
        print(f"\n{summary.description}")
        print(f"  Size: {summary.size} samples ({summary.percentage:.1f}%)")
        
        if summary.feature_means:
            print("  Key features:")
            key_features = ['loop_count', 'max_nesting_depth', 'function_count', 
                          'cyclomatic_complexity', 'has_recursion']
            for feat in key_features:
                if feat in summary.feature_means:
                    print(f"    - {feat}: {summary.feature_means[feat]:.2f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test with mock data
    from .clusterer import ClusterResult
    
    np.random.seed(42)
    
    # Mock cluster result
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    features = np.random.randn(10, 5)
    feature_names = ['loop_count', 'max_nesting_depth', 'function_count', 
                    'cyclomatic_complexity', 'has_recursion']
    
    # Modify features to create distinct clusters
    features[:3, 0] = 10  # Cluster 0: many loops
    features[3:7, 1] = 8  # Cluster 1: deep nesting
    features[7:, 4] = 1   # Cluster 2: recursive
    
    result = ClusterResult(
        labels=labels,
        n_clusters=3,
        algorithm='test',
        metrics={'silhouette': 0.5},
        cluster_centers=np.array([
            features[:3].mean(axis=0),
            features[3:7].mean(axis=0),
            features[7:].mean(axis=0),
        ])
    )
    
    summaries = analyze_clusters(result, features, feature_names)
    print_cluster_report(summaries)
