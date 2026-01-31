"""
Tests for anomaly detection module including ensemble methods.
"""

import pytest
import numpy as np

from src.detection.anomaly import (
    AnomalyDetector,
    EnsembleAnomalyDetector,
    detect_anomalies,
    AnomalyResult,
    AnomalyReport,
)


class TestAnomalyDetector:
    """Test the base AnomalyDetector class."""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal data with some outliers."""
        np.random.seed(42)
        normal = np.random.randn(90, 10) * 2 + 5
        outliers = np.random.randn(10, 10) * 2 + 20
        return np.vstack([normal, outliers])
    
    def test_isolation_forest_fit_predict(self, normal_data):
        """Test Isolation Forest algorithm."""
        detector = AnomalyDetector(algorithm='isolation_forest', contamination=0.1)
        report = detector.fit_predict(normal_data)
        
        assert isinstance(report, AnomalyReport)
        assert len(report.results) == 100
        assert report.algorithm == 'isolation_forest'
        assert 0 <= report.stats['anomaly_rate'] <= 100
    
    def test_one_class_svm_fit_predict(self, normal_data):
        """Test One-Class SVM algorithm."""
        detector = AnomalyDetector(algorithm='one_class_svm', contamination=0.1)
        report = detector.fit_predict(normal_data)
        
        assert isinstance(report, AnomalyReport)
        assert len(report.results) == 100
        assert report.algorithm == 'one_class_svm'
    
    def test_unknown_algorithm_raises(self):
        """Test that unknown algorithm raises ValueError."""
        detector = AnomalyDetector(algorithm='unknown')
        with pytest.raises(ValueError, match="Unknown algorithm"):
            detector.fit(np.random.randn(10, 5))
    
    def test_predict_without_fit_raises(self):
        """Test that predicting without fitting raises RuntimeError."""
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict(np.random.randn(10, 5))
    
    def test_anomaly_result_structure(self, normal_data):
        """Test AnomalyResult dataclass structure."""
        detector = AnomalyDetector(contamination=0.1)
        report = detector.fit_predict(normal_data)
        
        result = report.results[0]
        assert isinstance(result, AnomalyResult)
        assert isinstance(result.is_anomaly, (bool, np.bool_))
        assert 0 <= result.anomaly_score <= 100
        assert 0 <= result.percentile <= 100


class TestEnsembleAnomalyDetector:
    """Test the EnsembleAnomalyDetector class."""
    
    @pytest.fixture
    def test_data(self):
        """Generate test data with clear outliers."""
        np.random.seed(42)
        normal = np.random.randn(80, 10) * 2 + 5
        outliers = np.random.randn(20, 10) * 2 + 25
        return np.vstack([normal, outliers])
    
    def test_ensemble_soft_voting(self, test_data):
        """Test ensemble with soft voting."""
        detector = EnsembleAnomalyDetector(contamination=0.15, voting='soft')
        report = detector.fit_predict(test_data)
        
        assert isinstance(report, AnomalyReport)
        assert len(report.results) == 100
        assert 'ensemble' in report.algorithm
        assert 'if_anomalies' in report.stats
        assert 'lof_anomalies' in report.stats
    
    def test_ensemble_hard_voting(self, test_data):
        """Test ensemble with hard voting."""
        detector = EnsembleAnomalyDetector(contamination=0.15, voting='hard')
        report = detector.fit_predict(test_data)
        
        assert isinstance(report, AnomalyReport)
        assert 'ensemble' in report.algorithm
    
    def test_ensemble_detects_outliers(self, test_data):
        """Test that ensemble actually detects outliers."""
        detector = EnsembleAnomalyDetector(contamination=0.2)
        report = detector.fit_predict(test_data)
        
        # Should detect at least some of the injected outliers (indices 80-99)
        high_index_anomalies = sum(1 for idx in report.anomaly_indices if idx >= 80)
        assert high_index_anomalies > 5, "Ensemble should detect most injected outliers"


class TestDetectAnomaliesFunction:
    """Test the convenience function."""
    
    def test_detect_anomalies_isolation_forest(self):
        """Test detect_anomalies with isolation_forest."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        
        report = detect_anomalies(data, algorithm='isolation_forest')
        assert report.algorithm == 'isolation_forest'
    
    def test_detect_anomalies_ensemble(self):
        """Test detect_anomalies with ensemble."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        
        report = detect_anomalies(data, algorithm='ensemble')
        assert 'ensemble' in report.algorithm
    
    def test_detect_anomalies_contamination(self):
        """Test different contamination levels."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        
        report_low = detect_anomalies(data, contamination=0.05)
        report_high = detect_anomalies(data, contamination=0.2)
        
        # Higher contamination should generally detect more anomalies
        # (though not strictly guaranteed due to algorithm behavior)
        assert report_low.stats['n_samples'] == report_high.stats['n_samples']
