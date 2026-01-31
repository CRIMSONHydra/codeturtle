# 🏗️ Technical Architecture & Implementation

This document details the internal working of the **CodeTurtle** system, explaining how code is ingested, processed, and analyzed using machine learning.

## 1. Data Ingestion & Filtering
**Module**: `src.collector`

The ingestion pipeline is designed to be lightweight but robust, avoiding "garbage" data that would confuse the ML models.

-   **Cloning**: We use `src.collector.github_client.py` to perform **shallow clones** (`depth=1`) of repositories. This avoids downloading entire git histories, saving significant bandwidth and storage.
-   **Streaming Processing**: `scripts/extract_features.py` uses a Python generator (`src/utils.batch_generator`) to read and process files in chunks (e.g., 32 at a time). This ensures that **RAM usage remains constant** even when analyzing datasets with 10,000+ files.
-   **Filtering**: `src.collector.file_filter.py` applies strict filters:
    -   **Extension**: Only `.py` files.
    -   **Exclusions**: Ignores `tests/`, `migrations/`, `setup.py`, and `__pycache__` to focus on core logic.
    -   **Size Constraints**: Skips files < 50 bytes (empty/trivial) or > 100KB (likely auto-generated or data dumps).

## 2. Code Preprocessing (Cleaning)
**Module**: `src.preprocessor`

Raw code contains noise (comments, inconsistent formatting) that is irrelevant to algorithmic logic. We clean it to ensure the embeddings capture *semantics*, not *style*.

### The Cleaning Pipeline (`cleaner.py`)
1.  **Docstring Removal**: Uses Python's `ast` module to safely identify and remove docstrings (module, class, and function levels) *before* treating the code as text. This prevents the "comment remover" from accidentally stripping valid string literals.
2.  **Comment Removal**: Uses python's `tokenize` module to tokenize the source, filter out `COMMENT` tokens, and `untokenize` back to string. This is safer than regex.
3.  **Whitespace Normalization**: Collapses multiple blank lines and trims trailing whitespace to standardize the "shape" of the code.

## 3. Feature Extraction
**Module**: `src.features`

We extract two types of features to represent code: **Structural** (Explicit) and **Semantic** (Implicit).

### A. Structural Features (Explicit)
**File**: `structural.py`
We parse the cleaned code into an **Abstract Syntax Tree (AST)** and traverse it to count nodes. We extract a **25-dimensional vector** per file, including:
-   **Complexity**: `cyclomatic_complexity`, `max_nesting_depth`
-   **Control Flow**: `loop_count` (for/while), `if_count`, `try_except_count`
-   **Patterns**: `has_recursion` (checks if a function calls itself), `recursion_count`
-   **Style**: `avg_function_length`, `global_var_count`

### B. Semantic Embeddings (Implicit)
**File**: `embeddings.py`
We use Transformer-based models to convert code into a **768-dimensional dense vector**.
-   **Model**: Defaults to `microsoft/unixcoder-base` (or `CodeBERT`). These models are pre-trained on millions of code snippets to understand algorithmic intent.
-   **ONNX Acceleration**:
    -   We provide an ONNX Export utility (`scripts/export_onnx.py`) to convert models for high-performance inference.
    -   The `ONNXEmbedder` backend uses `onnxruntime` (CPU/GPU) to bypass PyTorch overhead, delivering significant speedups.
-   **Vector Store (Caching)**:
    -   We use **ChromaDB** to persist embeddings.
    -   Before processing, the system computes a SHA-256 hash of the code. If the hash exists in the store, the file is skipped, enabling incremental updates.
-   **Processing**:
    -   Code is tokenized and truncated to `512` tokens.
    -   Model runs in **inference mode** on GPU (if available).
    -   We extract the `[CLS]` token embedding, which represents the aggregate semantic meaning of the sequence.
-   **Fallback**: If deep learning dependencies fail or GPU is missing, we fall back to **TF-IDF** (Term Frequency-Inverse Document Frequency) to generate "bag-of-words" embeddings.

## 4. Machine Learning Models
**Modules**: `src.clustering`, `src.detection`

### A. Pattern Discovery (Clustering)
We group the code based on the combined features (Structural + Embeddings).
-   **Models**:
    -   **K-Means** (Default): Fast, partitioned clustering. We use the **Elbow Method** (via `find_optimal_k`) to automatically select the best number of clusters (K).
    -   **DBSCAN**: Density-based clustering. Good for finding outliers and non-spherical clusters.
    -   **Hierarchical**: Used for dendrogram visualizations and understanding cluster relationships.

### B. Risk & Anomaly Detection
We use a hybrid approach to flag "risky" code:

1.  **Rule-Based Static Analysis** (`rules.py`):
    -   Checks for known anti-patterns (e.g., mutable default args, bare excepts, recursion without base case).
    -    assigns a weighted **Risk Score** (0-100) based on severity.

2.  **Unsupervised Anomaly Detection** (`anomaly.py`):
    -   **Algorithm**: **Isolation Forest**.
    -   **Logic**: It constructs random decision trees. Anomalies (rare patterns) are isolated closer to the root of the tree (shorter path length).
    -   **Outcome**: Files that look "mathematically different" from the rest of the codebase are flagged as anomalies.

### C. Final Risk Scoring
The final risk score for a file is a weighted combination:
$$ \text{Score} = 0.6 \times \text{RuleScore} + 0.4 \times \text{AnomalyScore} $$

This ensures we catch both *known* bad practices (rules) and *unknown* weird patterns (anomalies).
