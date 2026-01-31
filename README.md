# 🐢 CodeTurtle

[![CI](https://github.com/Start-Sandeep/codeturtle/actions/workflows/ci.yml/badge.svg)](https://github.com/Start-Sandeep/codeturtle/actions/workflows/ci.yml)

**Discover Hidden Programming Patterns in Open-Source Code**

CodeTurtle is an ML-powered system that analyzes GitHub Python code to discover hidden programming patterns, cluster similar coding styles, and detect risky or inefficient code.

## ✨ Features

- **⚡ Optimized Processing**: Generators and batch processing for constant RAM usage
- **🚀 ONNX Acceleration**: 3x faster inference using optimized ONNX Runtime
- **💾 Smart Caching**: Integrated ChromaDB vector store skips analysis of unchanged files
- **📊 Pattern Discovery**: Cluster similar code patterns using K-Means, DBSCAN, or Hierarchical clustering
- **🔍 Risk Detection**: Rule-based static analysis + ML anomaly detection
- **🧠 Code Embeddings**: GPU-accelerated CodeBERT/UniXcoder embeddings
- **📈 Visualizations**: t-SNE/PCA cluster plots, risk heatmaps, feature importance
- **🖥️ Interactive Dashboard**: Streamlit web interface for exploration

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd /path/to/codeturtle

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### GPU Support (Optional)

For GPU-accelerated embeddings with your RTX 4070:

```bash
# Install PyTorch with CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Basic Usage

```bash
# 1. Collect data from GitHub
python scripts/collect_data.py --limit 3

# 2. Extract features
# 2. Extract features (with ONNX acceleration & caching)
python scripts/extract_features.py --clean --onnx --cache

# 3. Run analysis
python scripts/run_analysis.py --visualize --report

# 4. Launch dashboard
streamlit run src/visualization/dashboard.py
```

## 📁 Project Structure

```
codeturtle/
├── config/
│   └── settings.py          # Configuration
├── src/
│   ├── collector/            # GitHub data collection
│   ├── preprocessor/         # Code cleaning & AST parsing
│   ├── features/             # Feature extraction & embeddings
│   ├── clustering/           # Pattern discovery
│   ├── detection/            # Risk analysis
│   └── visualization/        # Plots & dashboard
├── scripts/
│   ├── collect_data.py       # Data collection CLI
│   ├── extract_features.py   # Feature extraction CLI
│   └── run_analysis.py       # Full pipeline CLI
├── data/                     # Collected code
├── outputs/                  # Analysis results
└── tests/                    # Unit tests
```

## 🔧 Configuration

Set your GitHub token for higher rate limits:

```bash
export GITHUB_TOKEN="your_token_here"
```

Get a token at: https://github.com/settings/tokens → Generate new token → Select `public_repo` scope

## 📊 What It Analyzes

### Structural Features (25 dimensions)
- Loop counts (for, while)
- Nesting depth
- Cyclomatic complexity
- Function/class counts
- Recursion detection
- Error handling patterns
- And more...

### Code Embeddings (768 dimensions)
- Semantic code understanding via CodeBERT
- Algorithmic similarity detection
- Logic pattern recognition

### Risk Detection
- Bare except clauses
- Deep nesting (>5 levels)
- Mutable default arguments
- Recursion without base case
- Magic numbers
- And 10+ more rules...

## 📈 Example Output

```
🐢 CodeTurtle Analysis Pipeline
==================================================

📊 Loading features from outputs/features.csv...
   Loaded 347 samples with 27 columns

🎯 Clustering with kmeans...
   Found 5 clusters
   Silhouette score: 0.4521

⚠️ Detecting code risks...
   Average risk score: 32.4
   High-risk files (>=60): 23

📋 ANALYSIS SUMMARY
==================================================
Total files analyzed: 347
Clusters identified: 5
Anomalies detected: 31
High-risk files: 23
```

## 🎯 Target Repositories

Default repos for analysis (mix of good and bad code):

| Repository | Type |
|------------|------|
| TheAlgorithms/Python | Clean algorithms |
| ZikaZaki/code-smells-python | Code smell examples |
| ArjanCodes/2021-code-smells | Anti-patterns |
| sobolevn/python-code-disasters | Bad code examples |

## 📝 License

MIT License

---

*Built with 🐢 by CodeTurtle Team*
