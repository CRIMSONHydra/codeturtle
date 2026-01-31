"""
Anomaly Detection for Code Analysis

Uses machine learning to detect unusual/outlier code patterns
that might indicate problems not caught by static rules.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import ISOLATION_FOREST_CONTAMINATION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single sample."""
    is_anomaly: bool
    anomaly_score: float  # Higher = more anomalous
    percentile: float  # Position in score distribution


@dataclass
class AnomalyReport:
    """Complete anomaly detection report."""
    results: List[AnomalyResult]
    anomaly_indices: np.ndarray
    normal_indices: np.ndarray
    threshold: float
    algorithm: str
    stats: Dict[str, float]


class AnomalyDetector:
    """
    Anomaly detector for code features.
    
    Uses unsupervised ML to identify code that deviates
    significantly from the norm.
    """
    
    def __init__(
        self,
        algorithm: str = 'isolation_forest',
        contamination: float = ISOLATION_FOREST_CONTAMINATION,
    ):
        """
        Initialize anomaly detector.
        
        Args:
            algorithm: 'isolation_forest' or 'one_class_svm'
            contamination: Expected proportion of outliers
        """
        self.algorithm = algorithm
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, features: np.ndarray):
        """
        Fit the anomaly detector on "normal" code samples.
        
        Args:
            features: Feature matrix (n_samples, n_features)
        """
        # Normalize features
        X = self.scaler.fit_transform(features)
        
        if self.algorithm == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
        elif self.algorithm == 'one_class_svm':
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto',
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.model.fit(X)
        self._is_fitted = True
        logger.info(f"Fitted {self.algorithm} on {len(features)} samples")
    
    def predict(self, features: np.ndarray) -> AnomalyReport:
        """
        Predict anomalies in new samples.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            AnomalyReport with all predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        X = self.scaler.transform(features)
        
        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X)
        
        # Get anomaly scores (lower = more anomalous for IsolationForest)
        if hasattr(self.model, 'score_samples'):
            raw_scores = self.model.score_samples(X)
            # Invert so higher = more anomalous
            scores = -raw_scores
        else:
            # Fallback for SVM
            scores = -self.model.decision_function(X)
        
        # Normalize scores to 0-100 range
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            normalized_scores = (scores - score_min) / (score_max - score_min) * 100
        else:
            normalized_scores = np.zeros_like(scores)
        
        # Calculate percentiles
        percentiles = np.array([
            (scores <= s).mean() * 100 for s in scores
        ])
        
        # Build results
        results = []
        for i in range(len(features)):
            results.append(AnomalyResult(
                is_anomaly=(predictions[i] == -1),
                anomaly_score=normalized_scores[i],
                percentile=percentiles[i],
            ))
        
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        normal_indices = np.where(~anomaly_mask)[0]
        
        # Calculate threshold (score at which samples become anomalies)
        if len(anomaly_indices) > 0:
            threshold = normalized_scores[anomaly_mask].min()
        else:
            threshold = 100.0
        
        stats = {
            'n_samples': len(features),
            'n_anomalies': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(features) * 100,
            'mean_score': normalized_scores.mean(),
            'std_score': normalized_scores.std(),
        }
        
        return AnomalyReport(
            results=results,
            anomaly_indices=anomaly_indices,
            normal_indices=normal_indices,
            threshold=threshold,
            algorithm=self.algorithm,
            stats=stats,
        )
    
    def fit_predict(self, features: np.ndarray) -> AnomalyReport:
        """
        Fit and predict in one step.
        
        Args:
            features: Feature matrix
            
        Returns:
            AnomalyReport
        """
        self.fit(features)
        return self.predict(features)


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple algorithms.
    
    Uses voting from Isolation Forest and Local Outlier Factor
    for more robust anomaly detection.
    """
    
    def __init__(
        self,
        contamination: float = ISOLATION_FOREST_CONTAMINATION,
        voting: str = 'soft',  # 'soft' (average scores) or 'hard' (majority vote)
    ):
        """
        Initialize ensemble detector.
        
        Args:
            contamination: Expected proportion of outliers
            voting: 'soft' for averaged scores, 'hard' for majority voting
        """
        self.contamination = contamination
        self.voting = voting
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.lof = None  # Instantiated in fit_predict based on data size
        
        # Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
    
    def fit_predict(self, features: np.ndarray) -> AnomalyReport:
        """
        Fit ensemble and predict anomalies.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            AnomalyReport with ensemble predictions
        """
        X = self.scaler.fit_transform(features)
        n_samples = X.shape[0]
        
        if n_samples < 2:
            raise ValueError(f"Ensemble detection requires at least 2 samples, got {n_samples}")
            
        # Initialize LOF with dynamic neighbors
        n_neighbors = min(20, n_samples - 1)
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False,
        )
        
        # Get predictions from each model
        if_pred = self.isolation_forest.fit_predict(X)  # -1 = anomaly, 1 = normal
        lof_pred = self.lof.fit_predict(X)  # -1 = anomaly, 1 = normal
        
        # Get scores
        if_scores = -self.isolation_forest.score_samples(X)
        lof_scores = -self.lof.negative_outlier_factor_
        
        # Normalize scores to 0-100
        def normalize_scores(scores):
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                return (scores - s_min) / (s_max - s_min) * 100
            return np.zeros_like(scores)
        
        if_scores_norm = normalize_scores(if_scores)
        lof_scores_norm = normalize_scores(lof_scores)
        
        if self.voting == 'soft':
            # Average scores
            combined_scores = (if_scores_norm + lof_scores_norm) / 2
            # Threshold at mean + 1.5*std for anomaly classification
            threshold = combined_scores.mean() + 1.5 * combined_scores.std()
            predictions = np.where(combined_scores >= threshold, -1, 1)
        else:
            # Hard voting - anomaly if both agree or either is very confident
            # Anomaly if at least one model says anomaly
            predictions = np.where((if_pred == -1) | (lof_pred == -1), -1, 1)
            combined_scores = np.maximum(if_scores_norm, lof_scores_norm)
        
        # Calculate percentiles
        percentiles = np.array([(combined_scores <= s).mean() * 100 for s in combined_scores])
        
        # Build results
        results = []
        for i in range(len(features)):
            results.append(AnomalyResult(
                is_anomaly=(predictions[i] == -1),
                anomaly_score=combined_scores[i],
                percentile=percentiles[i],
            ))
        
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        normal_indices = np.where(~anomaly_mask)[0]
        
        threshold = combined_scores[anomaly_mask].min() if len(anomaly_indices) > 0 else 100.0
        
        stats = {
            'n_samples': len(features),
            'n_anomalies': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(features) * 100,
            'mean_score': combined_scores.mean(),
            'std_score': combined_scores.std(),
            'if_anomalies': (if_pred == -1).sum(),
            'lof_anomalies': (lof_pred == -1).sum(),
        }
        
        self._is_fitted = True
        logger.info(f"Ensemble detected {len(anomaly_indices)} anomalies (IF: {stats['if_anomalies']}, LOF: {stats['lof_anomalies']})")
        
        return AnomalyReport(
            results=results,
            anomaly_indices=anomaly_indices,
            normal_indices=normal_indices,
            threshold=threshold,
            algorithm=f'ensemble_{self.voting}',
            stats=stats,
        )

def detect_anomalies(
    features: np.ndarray,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    algorithm: str = 'isolation_forest',
) -> AnomalyReport:
    """
    Convenience function for anomaly detection.
    
    Args:
        features: Feature matrix
        contamination: Expected proportion of outliers
        algorithm: Detection algorithm ('isolation_forest', 'one_class_svm', 'ensemble')
        
    Returns:
        AnomalyReport
    """
    if algorithm == 'ensemble':
        detector = EnsembleAnomalyDetector(
            contamination=contamination,
            voting='soft',
        )
    else:
        detector = AnomalyDetector(
            algorithm=algorithm,
            contamination=contamination,
        )
    return detector.fit_predict(features)


def get_risk_scores(
    features: np.ndarray,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
) -> np.ndarray:
    """
    Get risk scores (0-100) for each code sample.
    
    Higher scores indicate more unusual/risky code.
    
    Args:
        features: Feature matrix
        contamination: Expected outlier proportion
        
    Returns:
        Array of risk scores
    """
    report = detect_anomalies(features, contamination)
    return np.array([r.anomaly_score for r in report.results])


def combine_risk_scores(
    structural_features: np.ndarray,
    rule_scores: np.ndarray,
    anomaly_weight: float = 0.4,
) -> np.ndarray:
    """
    Combine rule-based and ML-based risk scores.
    
    Args:
        structural_features: Structural feature matrix
        rule_scores: Risk scores from rule-based detection (0-100)
        anomaly_weight: Weight for anomaly scores (0-1)
        
    Returns:
        Combined risk scores (0-100)
    """
    anomaly_scores = get_risk_scores(structural_features)
    
    # Weighted combination
    combined = (
        anomaly_weight * anomaly_scores +
        (1 - anomaly_weight) * rule_scores
    )
    
    return np.clip(combined, 0, 100)


def print_anomaly_report(report: AnomalyReport, file_names: Optional[List[str]] = None):
    """Print a formatted anomaly detection report."""
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION REPORT")
    print("=" * 60)
    print(f"\nAlgorithm: {report.algorithm}")
    print(f"Total samples: {report.stats['n_samples']}")
    print(f"Anomalies detected: {report.stats['n_anomalies']} ({report.stats['anomaly_rate']:.1f}%)")
    print(f"Threshold: {report.threshold:.1f}")
    
    if len(report.anomaly_indices) > 0:
        print("\nAnomalous samples:")
        for idx in report.anomaly_indices[:10]:  # Limit output
            result = report.results[idx]
            name = file_names[idx] if file_names else f"Sample {idx}"
            print(f"  - {name}: score={result.anomaly_score:.1f}, percentile={result.percentile:.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test anomaly detection
    np.random.seed(42)
    
    # Generate "normal" code features
    n_normal = 90
    normal_features = np.random.randn(n_normal, 10) * 2 + 5
    
    # Add some anomalies (very different patterns)
    n_anomalies = 10
    anomaly_features = np.random.randn(n_anomalies, 10) * 2 + 20  # Different distribution
    
    all_features = np.vstack([normal_features, anomaly_features])
    file_names = [f"normal_{i}.py" for i in range(n_normal)] + \
                 [f"suspicious_{i}.py" for i in range(n_anomalies)]
    
    print("Testing Isolation Forest...")
    report = detect_anomalies(all_features, contamination=0.1)
    print_anomaly_report(report, file_names)
    
    # Check if we caught the anomalies
    detected_suspicious = sum(1 for idx in report.anomaly_indices if idx >= n_normal)
    print(f"\nCorrectly detected {detected_suspicious}/{n_anomalies} injected anomalies")
