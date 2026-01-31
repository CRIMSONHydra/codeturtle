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

**Basic Usage:**
```bash
uv run python scripts/collect_data.py --repos "TheAlgorithms/Python" --limit 5
```

**Custom Repositories:**
You can target any public GitHub repository. For a good analysis mix, try combining "good" code (libraries, algorithms) with "messy" code (student projects, code smell examples).
```bash
uv run python scripts/collect_data.py --repos "django/django" "flask/flask" "my-username/my-messy-project"
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

---

## Step 3: Analysis & Pattern Discovery

Run the ML pipeline to cluster code and find risks.

```bash
uv run python scripts/run_analysis.py --output outputs/my_experiment --visualize --report
```

**Options:**
-   `--algorithm`: Choose `kmeans` (default), `dbscan`, or `hierarchical`.
-   `--n-clusters`: Force a specific number of clusters (e.g., `5`). If omitted, it auto-detects.
-   `--visualize`: Generates static plots (PNGs) in the output folder.

---

## Step 4: Visual Dashboard (Interpretation)

Launch the interactive web UI to explore the results.

```bash
uv run streamlit run src/visualization/dashboard.py
```

**How to Use the Dashboard:**
1.  **Overview Tab**: Check general stats. Are there many high-risk files?
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
