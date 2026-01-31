# 📘 Usage & Workflow Guide

How to use **CodeTurtle** effectively to discover patterns in your codebase.

## 🏁 Workflow Overview

1.  **Collect** raw code data.
2.  **Preprocess & Extract** features.
3.  **Analyze** (Cluster & Detect).
4.  **Visualize** & Interpret.

---

## Step 1: Data Collection

Use the `collect_data.py` script to grab repositories.

**Configuration (Optional but Recommended):**
To avoid GitHub API rate limits or to access private repositories, set your token:
```bash
export GITHUB_TOKEN="your_github_token"
```

**Basic Usage:**
```bash
uv run python scripts/collect_data.py --repos "TheAlgorithms/Python" --limit 5
```

**Custom Repositories:**
You can target any public GitHub repository. For a good analysis mix, try combining "good" code (libraries, algorithms) with "messy" code (student projects, code smell examples).
```bash
uv run python scripts/collect_data.py --repos "django/django" "pallets/flask" "my-username/my-messy-project"
```

**Check Stats:**
See what you've downloaded without re-downloading.
```bash
uv run python scripts/collect_data.py --stats
```

---

## Step 2: Feature Extraction

Convert raw `.py` files into mathematical vectors.

**Standard Run (Structural Only):**
Fast run, mostly for testing pipeline.
```bash
uv run python scripts/extract_features.py --clean
```

**Full Run (Structural + Embeddings - Recommended):**
This enables the AI (CodeBERT) features. Requires GPU for speed.
```bash
uv run python scripts/extract_features.py --clean --embeddings --batch-size 32
```
*Tip: Reduce `--batch-size` if you run out of GPU memory.*

**With Caching & ONNX (Fastest):**
Skip unchanged files and run inference 3x faster.
```bash
uv run python scripts/extract_features.py --clean --embeddings --cache --onnx
```
*Use `--clear-cache` to reset the embedding cache.*

**Parallel Processing (Multi-Core Speedup):**
Use multiple CPU cores for structural feature extraction.
```bash
# Auto-detect cores (-1 = all but one)
uv run python scripts/extract_features.py --clean --embeddings --onnx --parallel -1

# Specify exact worker count
uv run python scripts/extract_features.py --clean --parallel 4
```


**With Graph Neural Networks (Deep Structural Analysis):**
Capture complex structural patterns (like recursive logic flow) using a GNN.
```bash
# 1. Train the GNN on your collected data (Self-Supervised)
uv run python scripts/train_gnn.py --epochs 20

# 2. Extract features using the trained model (with parallel + ONNX)
uv run python scripts/extract_features.py --clean --embeddings --onnx --gnn --parallel -1

# 3. Analyze with both embedding types
uv run python scripts/run_analysis.py --embeddings outputs/embeddings.npy --gnn-embeddings outputs/gnn_embeddings.npy --visualize --report
```

---

## Step 3: Analysis & Pattern Discovery

Run the ML pipeline to cluster code and find risks.

```bash
uv run python scripts/run_analysis.py --output outputs/my_experiment --visualize --report --html
```

**Options:**
-   `--algorithm`: Choose `kmeans` (default), `dbscan`, or `hierarchical`.
-   `--n-clusters`: Force a specific number of clusters (e.g., `5`). If omitted, it auto-detects.
-   `--visualize`: Generates static plots (PNGs) in the output folder.
-   `--html`: Generate a beautiful HTML report with charts and tables.
-   `--anomaly-algorithm`: Choose `ensemble` (default), `isolation_forest`, or `one_class_svm`.

---

## Step 4: Configuration File (Optional)

Create a `codeturtle.yaml` in your project root for persistent configuration:

```yaml
# codeturtle.yaml
extraction:
  parallel_workers: -1
  use_onnx: true
  use_gnn: true

analysis:
  algorithm: kmeans
  anomaly_algorithm: ensemble
  contamination: 0.1

output:
  html_report: true
  plots: true
```

Then run analysis with `--config codeturtle.yaml` or it will auto-detect.

---

## Step 5: Visual Dashboard (Interpretation)

Launch the interactive web UI to explore the results.

```bash
uv run streamlit run src/visualization/dashboard.py
```

**How to Use the Dashboard:**
1.  **Select Mode**: Choose "**Load Results**" in the sidebar to visualize the actual analysis output (from `outputs/analysis_results.csv`).
2.  **Upload Files**: Alternatively, upload new Python files for instant analysis.
3.  **Demo Data**: Use the demo mode to explore dashboard features without data.
2.  **Clusters Tab**: Look at the scatter plot.
    -   *Hover* over dots to see filenames.
    -   *Check* if similar files (e.g., sorting algorithms vs web views) are grouped together.
3.  **Risk Tab**: Sort by "Risk Score". Open the highest-risk files to see *why* they were flagged.
4.  **Code Viewer**: Select a file to see its source code alongside its metrics.

---

## 🛠️ Improving Your Data

Garbage In, Garbage Out. To get better clusters:

1.  **More Data**: 10 files isn't enough for ML. Aim for **200-500 files**.
2.  **Diverse Sources**: Don't just analyze one repo. Mix different *types* of code (scripts, web apps, algorithm libs).
3.  **Clean Up**: If you see `setup.py` or config files clustering together, add their patterns to `config/settings.py` under `EXCLUDED_PATTERNS`.

## 🔄 Iteration Loop

1.  Run Analysis.
2.  Check Dashboard -> "Are these clusters meaningful?"
3.  If **No**:
    -   Adjust *Features*: Edit `src/features/structural.py` to add new counts (e.g., "list_comprehensions").
    -   Adjust *Weights*: Change how much Embeddings vs Structural features matter in `scripts/run_analysis.py`.
4.  Re-run `run_analysis.py`.

---

## 🛡️ Troubleshooting & Stability

**Deep Learning Failures (CUDA/GPU):**
The system is designed to be **fault-tolerant**:
-   If you see `CUDA Error` or `illegal memory access` in the logs, **don't panic**. The system will automatically catch this and switch to **CPU mode** for the affected batch.
-   If `ONNX` and `GNN` are both enabled, GNN inference is automatically moved to CPU to prevent GPU context conflicts.

**Memory Issues:**
-   **Graph Too Large**: If a single file produces a graph > 5,000 nodes, it is automatically skipped to prevent Out-Of-Memory (OOM) crashes.
-   **System OOM**: Reduce `--batch-size` (default 32) in `scripts/extract_features.py`.
