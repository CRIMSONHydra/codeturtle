"""
Tests for HTML report generation.
"""

import pytest
from pathlib import Path
import tempfile
import pandas as pd

from src.visualization.html_report import generate_html_report


class TestHTMLReportGeneration:
    """Test the HTML report generator."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'filename': ['test.py', 'main.py', 'utils.py', 'app.py', 'config.py'],
            'filepath': ['/path/test.py', '/path/main.py', '/path/utils.py', '/path/app.py', '/path/config.py'],
            'risk_score': [85.5, 42.0, 15.0, 67.2, 5.0],
            'cluster': [0, 1, 2, 0, 2],
            'is_anomaly': [True, False, False, True, False],
            'anomaly_score': [78.0, 35.0, 10.0, 82.0, 5.0],
        })
    
    @pytest.fixture
    def sample_cluster_info(self):
        """Create sample cluster info for testing."""
        return [
            {'cluster_id': 0, 'description': 'High complexity', 'size': 10, 'percentage': 33.3},
            {'cluster_id': 1, 'description': 'Medium complexity', 'size': 12, 'percentage': 40.0},
            {'cluster_id': 2, 'description': 'Simple code', 'size': 8, 'percentage': 26.7},
        ]
    
    def test_generate_report_creates_file(self, sample_df, sample_cluster_info):
        """Test that generate_html_report creates an HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            report_path = generate_html_report(sample_df, sample_cluster_info, output_path)
            
            assert report_path.exists()
            assert report_path.name == 'report.html'
            assert report_path.stat().st_size > 0
    
    def test_report_contains_statistics(self, sample_df, sample_cluster_info):
        """Test that report contains key statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            report_path = generate_html_report(sample_df, sample_cluster_info, output_path)
            
            content = report_path.read_text()
            
            # Check for key statistics
            assert '5' in content  # Total files
            assert '3' in content  # Number of clusters
            assert 'High complexity' in content
            assert 'Medium complexity' in content
    
    def test_report_contains_risk_table(self, sample_df, sample_cluster_info):
        """Test that report contains risk table with file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            report_path = generate_html_report(sample_df, sample_cluster_info, output_path)
            
            content = report_path.read_text()
            
            # Check for file names in risk table
            assert 'test.py' in content
            assert 'app.py' in content
    
    def test_report_with_custom_title(self, sample_df, sample_cluster_info):
        """Test report generation with custom title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            custom_title = "My Custom Analysis"
            report_path = generate_html_report(
                sample_df, sample_cluster_info, output_path, 
                title=custom_title
            )
            
            content = report_path.read_text()
            assert custom_title in content
    
    def test_report_html_structure(self, sample_df, sample_cluster_info):
        """Test that report has valid HTML structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            report_path = generate_html_report(sample_df, sample_cluster_info, output_path)
            
            content = report_path.read_text()
            
            # Basic HTML structure checks
            assert '<!DOCTYPE html>' in content
            assert '<html' in content
            assert '</html>' in content
            assert '<head>' in content
            assert '</head>' in content
            assert '<body>' in content
            assert '</body>' in content
    
    def test_report_empty_df(self, sample_cluster_info):
        """Test report generation with empty DataFrame."""
        empty_df = pd.DataFrame({
            'filename': [],
            'risk_score': [],
            'cluster': [],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            report_path = generate_html_report(empty_df, sample_cluster_info, output_path)
            
            assert report_path.exists()
    
    def test_report_missing_columns(self):
        """Test report generation with minimal columns."""
        minimal_df = pd.DataFrame({
            'filename': ['test.py'],
        })
        cluster_info = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            report_path = generate_html_report(minimal_df, cluster_info, output_path)
            
            assert report_path.exists()
            content = report_path.read_text()
            assert '1' in content  # Total files
