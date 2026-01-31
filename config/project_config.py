"""
Configuration File Support for CodeTurtle

Allows project-specific configuration via codeturtle.yaml.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """Project-specific configuration."""
    
    # Feature extraction
    batch_size: int = 32
    parallel_workers: int = 0  # 0=sequential, -1=auto
    use_onnx: bool = True
    use_gnn: bool = False
    use_cache: bool = True
    clean_code: bool = True
    
    # Analysis
    algorithm: str = "kmeans"
    n_clusters: Optional[int] = None  # Auto-detect if None
    anomaly_algorithm: str = "ensemble"
    contamination: float = 0.1
    
    # Risk scoring
    risk_weights: Dict[str, float] = field(default_factory=lambda: {
        "structural": 0.4,
        "anomaly": 0.3,
        "rule_based": 0.3,
    })
    
    # Output
    output_dir: str = "outputs"
    generate_html: bool = True
    generate_plots: bool = True
    
    # Exclusions
    excluded_patterns: list = field(default_factory=lambda: [
        "test_*.py",
        "*_test.py",
        "setup.py",
        "conftest.py",
        "__init__.py",
    ])


def load_config(config_path: Optional[Path] = None) -> ProjectConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, searches for codeturtle.yaml
                    in current directory and parent directories.
                    
    Returns:
        ProjectConfig with loaded or default values
    """
    if config_path is None:
        # Search for config file in current and parent directories
        current_dir = Path.cwd()
        # Traverse up to root
        search_dirs = [current_dir] + list(current_dir.parents)
        
        for directory in search_dirs:
            # Check for standard config names
            search_paths = [
                directory / "codeturtle.yaml",
                directory / ".codeturtle.yaml",
                directory / "codeturtle.yml",
            ]
            
            for path in search_paths:
                if path.exists():
                    config_path = path
                    break
            
            if config_path:
                break
    
    if config_path is None or not config_path.exists():
        logger.debug("No config file found, using defaults")
        return ProjectConfig()
    
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, using default config")
        return ProjectConfig()
    
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        
        logger.info(f"Loaded config from {config_path}")
        
        # Flatten nested config
        extraction = data.get('extraction', {})
        analysis = data.get('analysis', {})
        output = data.get('output', {})
        
        return ProjectConfig(
            # Extraction
            batch_size=extraction.get('batch_size', 32),
            parallel_workers=extraction.get('parallel_workers', 0),
            use_onnx=extraction.get('use_onnx', True),
            use_gnn=extraction.get('use_gnn', False),
            use_cache=extraction.get('use_cache', True),
            clean_code=extraction.get('clean_code', True),
            
            # Analysis
            algorithm=analysis.get('algorithm', 'kmeans'),
            n_clusters=analysis.get('n_clusters'),
            anomaly_algorithm=analysis.get('anomaly_algorithm', 'ensemble'),
            contamination=analysis.get('contamination', 0.1),
            
            # Risk
            risk_weights=analysis.get('risk_weights', {
                "structural": 0.4,
                "anomaly": 0.3,
                "rule_based": 0.3,
            }),
            
            # Output
            output_dir=output.get('directory', 'outputs'),
            generate_html=output.get('html_report', True),
            generate_plots=output.get('plots', True),
            
            # Exclusions
            excluded_patterns=data.get('excluded_patterns', [
                "test_*.py", "*_test.py", "setup.py", "conftest.py", "__init__.py"
            ]),
        )
        
    except (OSError, TypeError, ValueError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        return ProjectConfig()


def save_default_config(output_path: Path) -> Path:
    """
    Save a default configuration file as a template.
    
    Args:
        output_path: Where to save the config
        
    Returns:
        Path to saved config file
    """
    default_config = """# CodeTurtle Configuration
# Copy to your project root as codeturtle.yaml

# Feature extraction settings
extraction:
  batch_size: 32
  parallel_workers: 0     # 0 = sequential (default), -1 = auto-detect
  use_onnx: true          # Use ONNX for faster embeddings
  use_gnn: false          # Enable GNN structural embeddings
  use_cache: true         # Cache embeddings for incremental updates
  clean_code: true        # Clean comments/docstrings before analysis

# Analysis settings
analysis:
  algorithm: kmeans       # kmeans, dbscan, or hierarchical
  n_clusters: null        # null = auto-detect optimal k
  anomaly_algorithm: ensemble  # isolation_forest, one_class_svm, or ensemble
  contamination: 0.1      # Expected proportion of anomalies
  
  # Risk score weights (should sum to 1.0)
  risk_weights:
    structural: 0.4
    anomaly: 0.3
    rule_based: 0.3

# Output settings
output:
  directory: outputs
  html_report: true
  plots: true

# Files to exclude from analysis
excluded_patterns:
  - "test_*.py"
  - "*_test.py"
  - "setup.py"
  - "conftest.py"
  - "__init__.py"
"""
    
    output_path = Path(output_path)
    output_path.write_text(default_config)
    return output_path


# Preset configurations
PRESETS = {
    'strict': ProjectConfig(
        anomaly_algorithm='ensemble',
        contamination=0.15,
        risk_weights={'structural': 0.3, 'anomaly': 0.4, 'rule_based': 0.3},
    ),
    'balanced': ProjectConfig(
        anomaly_algorithm='isolation_forest',
        contamination=0.1,
        risk_weights={'structural': 0.4, 'anomaly': 0.3, 'rule_based': 0.3},
    ),
    'permissive': ProjectConfig(
        anomaly_algorithm='isolation_forest',
        contamination=0.05,
        risk_weights={'structural': 0.5, 'anomaly': 0.2, 'rule_based': 0.3},
    ),
}


def get_preset(name: str) -> ProjectConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
