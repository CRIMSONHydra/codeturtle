"""
Visualization Module

Creates beautiful plots for cluster analysis, risk distribution,
and feature exploration.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    PLOT_STYLE,
    FIGURE_DPI,
    FIGURE_SIZE,
    TSNE_PERPLEXITY,
    TSNE_N_ITER,
    PCA_COMPONENTS,
    OUTPUTS_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
try:
    plt.style.use(PLOT_STYLE)
except:
    plt.style.use('seaborn-v0_8-darkgrid')


def reduce_dimensions(
    features: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce feature dimensions for visualization.
    
    Args:
        features: High-dimensional feature matrix
        method: 'tsne' or 'pca'
        n_components: Target dimensions (usually 2)
        
    Returns:
        Reduced feature matrix
    """
    if features.shape[1] <= n_components:
        return features
    
    # First reduce with PCA if needed (t-SNE is slow on high dims)
    if features.shape[1] > PCA_COMPONENTS and method == 'tsne':
        pca = PCA(n_components=min(PCA_COMPONENTS, features.shape[0] - 1))
        features = pca.fit_transform(features)
    
    if method == 'tsne':
        perplexity = min(TSNE_PERPLEXITY, len(features) - 1)
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=TSNE_N_ITER,  # Renamed from n_iter in sklearn 1.5+
            random_state=42,
        )
        reduced = tsne.fit_transform(features)
    else:  # PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(features)
    
    return reduced


def plot_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "Code Clusters",
    method: str = 'tsne',
    file_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot cluster visualization using dimensionality reduction.
    
    Args:
        features: Feature matrix
        labels: Cluster labels
        title: Plot title
        method: Reduction method ('tsne' or 'pca')
        file_names: Optional file names for hover info
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    # Reduce dimensions
    reduced = reduce_dimensions(features, method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    
    # Get unique labels and colors
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    
    # Color palette
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        
        if label == -1:
            # Noise points
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c='gray',
                marker='x',
                s=50,
                alpha=0.5,
                label='Noise',
            )
        else:
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colors[label % len(colors)]],
                s=80,
                alpha=0.7,
                label=f'Cluster {label} ({mask.sum()})',
                edgecolors='white',
                linewidths=0.5,
            )
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved cluster plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_risk_distribution(
    risk_scores: np.ndarray,
    file_names: Optional[List[str]] = None,
    threshold: float = 50.0,
    title: str = "Code Risk Distribution",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot distribution of risk scores.
    
    Args:
        risk_scores: Array of risk scores (0-100)
        file_names: Optional file names
        threshold: Risk threshold line
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=FIGURE_DPI)
    
    # Histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(
        risk_scores, bins=20, edgecolor='white', alpha=0.7
    )
    
    # Color by risk level
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 30:
            patch.set_facecolor('#2ecc71')  # Green - low risk
        elif bin_center < 60:
            patch.set_facecolor('#f39c12')  # Orange - medium
        else:
            patch.set_facecolor('#e74c3c')  # Red - high risk
    
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Risk Score', fontsize=12)
    ax1.set_ylabel('Number of Files', fontsize=12)
    ax1.set_title('Risk Score Distribution', fontsize=14)
    ax1.legend()
    
    # Sorted bar chart
    ax2 = axes[1]
    sorted_indices = np.argsort(risk_scores)[::-1][:20]  # Top 20 riskiest
    sorted_scores = risk_scores[sorted_indices]
    
    if file_names:
        sorted_names = [Path(file_names[i]).name[:20] for i in sorted_indices]
    else:
        sorted_names = [f"File {i}" for i in sorted_indices]
    
    colors = ['#e74c3c' if s >= 60 else '#f39c12' if s >= 30 else '#2ecc71' for s in sorted_scores]
    
    bars = ax2.barh(range(len(sorted_scores)), sorted_scores, color=colors, edgecolor='white')
    ax2.set_yticks(range(len(sorted_scores)))
    ax2.set_yticklabels(sorted_names, fontsize=9)
    ax2.set_xlabel('Risk Score', fontsize=12)
    ax2.set_title('Top 20 Riskiest Files', fontsize=14)
    ax2.invert_yaxis()
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved risk plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_feature_importance(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importance by Cluster",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot feature importance/differences between clusters.
    
    Args:
        features: Feature matrix
        labels: Cluster labels
        feature_names: Names of features
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    unique_labels = sorted([l for l in set(labels) if l != -1])
    n_features = min(len(feature_names), 15)  # Limit to top 15
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=FIGURE_DPI)
    
    # Calculate mean features per cluster
    cluster_means = []
    for label in unique_labels:
        mask = labels == label
        cluster_means.append(features[mask].mean(axis=0))
    
    cluster_means = np.array(cluster_means)
    
    # Find most variable features
    feature_variance = cluster_means.var(axis=0)
    top_feature_indices = np.argsort(feature_variance)[-n_features:][::-1]
    
    # Plot heatmap
    data = cluster_means[:, top_feature_indices].T
    top_names = [feature_names[i] for i in top_feature_indices]
    
    sns.heatmap(
        data,
        xticklabels=[f'Cluster {l}' for l in unique_labels],
        yticklabels=top_names,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=data.mean(),
        ax=ax,
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_elbow_curve(
    inertias: List[float],
    k_range: range,
    optimal_k: Optional[int] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot elbow curve for K-Means K selection.
    
    Args:
        inertias: List of inertia values
        k_range: Range of K values tested
        optimal_k: Highlight optimal K
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)
    
    ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    
    if optimal_k is not None:
        idx = list(k_range).index(optimal_k)
        ax.plot(optimal_k, inertias[idx], 'ro', markersize=15, label=f'Optimal K={optimal_k}')
        ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_summary_dashboard(
    cluster_labels: np.ndarray,
    risk_scores: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    file_names: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """
    Create a complete visualization dashboard.
    
    Args:
        cluster_labels: Cluster assignments
        risk_scores: Risk scores per file
        features: Feature matrix
        feature_names: Names of features
        file_names: Optional file names
        save_dir: Directory to save all figures
        
    Returns:
        Dictionary of figure names to Figure objects
    """
    if save_dir is None:
        save_dir = OUTPUTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Cluster visualization
    figures['clusters'] = plot_clusters(
        features, cluster_labels,
        title="Code Pattern Clusters",
        save_path=save_dir / "clusters.png",
        show=False,
    )
    
    # Risk distribution
    figures['risk'] = plot_risk_distribution(
        risk_scores, file_names,
        title="Code Risk Analysis",
        save_path=save_dir / "risk_distribution.png",
        show=False,
    )
    
    # Feature importance
    figures['features'] = plot_feature_importance(
        features, cluster_labels, feature_names,
        title="Feature Patterns by Cluster",
        save_path=save_dir / "feature_importance.png",
        show=False,
    )
    
    logger.info(f"Created dashboard with {len(figures)} visualizations in {save_dir}")
    return figures


if __name__ == "__main__":
    # Test visualizations
    np.random.seed(42)
    
    # Generate test data
    n_samples = 100
    n_features = 10
    
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 4, n_samples)
    risk_scores = np.random.uniform(0, 100, n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    print("Testing cluster plot...")
    plot_clusters(features, labels, method='pca', show=False)
    
    print("Testing risk distribution...")
    plot_risk_distribution(risk_scores, show=False)
    
    print("Testing feature importance...")
    plot_feature_importance(features, labels, feature_names, show=False)
    
    print("All visualization tests passed!")
