"""
CodeTurtle Configuration Settings
Central configuration for all modules.
"""

import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GitHub Configuration
# =============================================================================
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", None)

# Target repositories for analysis (mix of good and bad code)
TARGET_REPOS = [
    # Good code - clean algorithms
    "TheAlgorithms/Python",
    
    # Bad code examples - code smells
    "ZikaZaki/code-smells-python",
    "ArjanCodes/2021-code-smells",
    "sunnysid3up/bad-python-code",
    "sobolevn/python-code-disasters",
    
    # Beginner code - mixed quality
    "zhiwehu/Python-programming-exercises",
]

# File filtering
EXCLUDED_PATTERNS = [
    "**/test_*.py",
    "**/*_test.py",
    "**/tests/**",
    "**/test/**",
    "**/__pycache__/**",
    "**/setup.py",
    "**/conftest.py",
    "**/.git/**",
]

MIN_FILE_SIZE = 50  # bytes - skip empty/tiny files
MAX_FILE_SIZE = 100_000  # bytes - skip huge files

# =============================================================================
# Feature Extraction Configuration
# =============================================================================
# AST-based structural features
STRUCTURAL_FEATURES = [
    "loop_count",
    "max_nesting_depth",
    "has_recursion",
    "try_except_count",
    "global_var_count",
    "function_count",
    "class_count",
    "import_count",
    "avg_function_length",
    "max_function_length",
    "cyclomatic_complexity",
    "assertion_count",
    "lambda_count",
    "comprehension_count",
]

# =============================================================================
# CodeBERT Embedding Configuration
# =============================================================================
# Model choices (in order of preference)
EMBEDDING_MODELS = [
    "microsoft/unixcoder-base",      # Best for code understanding
    "microsoft/codebert-base",       # Good alternative
    "microsoft/graphcodebert-base",  # Graph-aware
]

DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS[0]
EMBEDDING_DIMENSION = 768
MAX_CODE_LENGTH = 512  # tokens
EMBEDDING_BATCH_SIZE = 8  # Adjust based on GPU memory (8GB VRAM)

# GPU Configuration
USE_GPU = True
GPU_DEVICE = "cuda:0"

# =============================================================================
# Clustering Configuration
# =============================================================================
# K-Means
KMEANS_N_CLUSTERS_RANGE = (3, 15)  # Range to test
KMEANS_DEFAULT_CLUSTERS = 7
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10

# DBSCAN
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 3

# Hierarchical
HIERARCHICAL_LINKAGE = "ward"

# =============================================================================
# Risk Detection Configuration
# =============================================================================
# Rule-based thresholds
RISK_THRESHOLDS = {
    "max_nesting_depth": 5,
    "max_function_length": 50,
    "max_parameters": 7,
    "max_cyclomatic_complexity": 10,
}

# Risk severity weights (0-1)
RISK_WEIGHTS = {
    "deep_nesting": 0.7,
    "bare_except": 0.9,
    "global_mutable": 0.6,
    "no_base_case": 0.95,
    "long_function": 0.4,
    "too_many_params": 0.5,
}

# Anomaly detection
ISOLATION_FOREST_CONTAMINATION = 0.1  # Expected outlier ratio

# =============================================================================
# Visualization Configuration
# =============================================================================
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 150
FIGURE_SIZE = (12, 8)

# t-SNE parameters
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# PCA for dimensionality reduction before t-SNE
PCA_COMPONENTS = 50
