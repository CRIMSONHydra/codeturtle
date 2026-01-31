"""
Tests for project configuration system.
"""

import pytest
from pathlib import Path
import tempfile

from config.project_config import (
    ProjectConfig,
    load_config,
    save_default_config,
    get_preset,
    PRESETS,
)


class TestProjectConfig:
    """Test the ProjectConfig dataclass."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = ProjectConfig()
        
        assert config.batch_size == 32
        assert config.parallel_workers == 0
        assert config.use_onnx is True
        assert config.use_gnn is False
        assert config.algorithm == 'kmeans'
        assert config.anomaly_algorithm == 'ensemble'
        assert config.contamination == 0.1
        assert config.generate_html is True
    
    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = ProjectConfig(
            batch_size=64,
            parallel_workers=-1,
            use_gnn=True,
            anomaly_algorithm='isolation_forest',
        )
        
        assert config.batch_size == 64
        assert config.parallel_workers == -1
        assert config.use_gnn is True
        assert config.anomaly_algorithm == 'isolation_forest'
    
    def test_risk_weights_default(self):
        """Test default risk weights."""
        config = ProjectConfig()
        
        assert 'structural' in config.risk_weights
        assert 'anomaly' in config.risk_weights
        assert 'rule_based' in config.risk_weights
        assert sum(config.risk_weights.values()) == pytest.approx(1.0)


class TestLoadConfig:
    """Test configuration loading."""
    
    def test_load_config_no_file(self):
        """Test loading config when no file exists returns defaults."""
        config = load_config(Path('/nonexistent/path/config.yaml'))
        
        assert isinstance(config, ProjectConfig)
        assert config.batch_size == 32  # Default value
    
    def test_load_config_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
extraction:
  batch_size: 64
  parallel_workers: 4
  use_gnn: true

analysis:
  algorithm: dbscan
  anomaly_algorithm: isolation_forest
  contamination: 0.15

output:
  html_report: false
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            
            assert config.batch_size == 64
            assert config.parallel_workers == 4
            assert config.use_gnn is True
            assert config.algorithm == 'dbscan'
            assert config.anomaly_algorithm == 'isolation_forest'
            assert config.contamination == 0.15
            assert config.generate_html is False
        finally:
            config_path.unlink()
    
    def test_load_config_partial_yaml(self):
        """Test loading config with only some values specified."""
        yaml_content = """
analysis:
  algorithm: hierarchical
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            
            # Specified value
            assert config.algorithm == 'hierarchical'
            # Default values
            assert config.batch_size == 32
            assert config.use_onnx is True
        finally:
            config_path.unlink()


class TestSaveDefaultConfig:
    """Test saving default configuration."""
    
    def test_save_creates_file(self):
        """Test that save_default_config creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_config.yaml'
            result_path = save_default_config(output_path)
            
            assert result_path.exists()
            assert result_path.stat().st_size > 0
    
    def test_saved_config_has_all_sections(self):
        """Test that saved config has all required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_config.yaml'
            save_default_config(output_path)
            
            content = output_path.read_text()
            
            assert 'extraction:' in content
            assert 'analysis:' in content
            assert 'output:' in content
            assert 'excluded_patterns:' in content
    
    def test_saved_config_is_loadable(self):
        """Test that saved config can be loaded back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_config.yaml'
            save_default_config(output_path)
            
            config = load_config(output_path)
            
            assert isinstance(config, ProjectConfig)
            assert config.batch_size == 32


class TestPresets:
    """Test configuration presets."""
    
    def test_presets_exist(self):
        """Test that expected presets exist."""
        assert 'strict' in PRESETS
        assert 'balanced' in PRESETS
        assert 'permissive' in PRESETS
    
    def test_get_preset_strict(self):
        """Test getting strict preset."""
        config = get_preset('strict')
        
        assert config.contamination == 0.15
        assert config.anomaly_algorithm == 'ensemble'
    
    def test_get_preset_balanced(self):
        """Test getting balanced preset."""
        config = get_preset('balanced')
        
        assert config.contamination == 0.1
    
    def test_get_preset_permissive(self):
        """Test getting permissive preset."""
        config = get_preset('permissive')
        
        assert config.contamination == 0.05
    
    def test_get_preset_unknown_raises(self):
        """Test that unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset('unknown_preset')
