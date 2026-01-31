# 🚀 Future Roadmap & Optimizations

Ideas for extending CodeTurtle and optimizing its performance.

## ⚡ Performance Optimizations

1.  **Batch Processing for Embeddings** (✅ Completed):
    -   Implemented streaming generator in `src/utils.py`.
    -   Processes batches (default 32) to keep RAM usage constant.
2.  **Cached Embeddings** (✅ Completed):
    -   Implemented ChromaDB-backed vector store in `src/features/vector_store.py`.
    -   Uses SHA-256 hashing for change detection.
    -   Use `--cache` flag to enable, `--clear-cache` to reset.
3.  **ONNX Export** (✅ Completed):
    -   Convert the PyTorch CodeBERT model to ONNX runtime for 2-3x faster inference on CPU/GPU.
    -   Use `--onnx` flag to enable. Auto-falls back to CPU if CUDA unavailable.
4.  **Parallel Feature Extraction** (✅ Completed):
    -   Multi-core structural feature extraction with `--parallel` flag.
    -   Use `--parallel -1` for auto-detection or specify worker count.
5.  **Progress Bars** (✅ Completed):
    -   `tqdm` progress bars during batch processing.

## 🧠 Model Improvements

1.  **Graph Neural Networks (GNNs)** (✅ Completed):
    -   Implemented a Graph Neural Network (GNN) to capture the *structure* of data flow much better than simple counts.
    -   Files: `src/features/gnn.py`, `graph_converter.py`.
2.  **Cluster Quality** (✅ Completed):
    -   Meaningful cluster descriptions derived from structural features (not PCA components).
    -   Logarithmic risk scoring to avoid ceiling effects.
3.  **Ensemble Anomaly Detection** (✅ Completed):
    -   Combines Isolation Forest + Local Outlier Factor for more robust outlier detection.
    -   Use `--anomaly-algorithm ensemble` (default).
4.  **Contrastive Fine-Tuning**:
    -   Fine-tune CodeBERT on the specific target dataset using contrastive loss (SimCLR) to force the model to separate "clean" and "messy" code further apart.
5.  **LLM Integration**:
    -   Use a small LLM (e.g., Llama-3-8B) to generate *explanations* for why a cluster exists. "This cluster contains recursive dynamic programming solutions."

## 📊 Reporting & Output

1.  **HTML Reports** (✅ Completed):
    -   Beautiful, dark-themed HTML reports with cluster summaries and risk tables.
    -   Use `--html` flag to generate.
2.  **Configuration File Support** (✅ Completed):
    -   Project-specific settings via `codeturtle.yaml`.
    -   See `codeturtle.example.yaml` for template.

## 🛠️ Feature Extensions

1.  **Language Agnostic Support**:
    -   Currently Python-only (`ast` module).
    -   Use `tree-sitter` for parsing to support JavaScript, Go, Rust, and Java.
2.  **CI/CD Action**:
    -   Turn `collect_data.py` + `run_analysis.py` into a GitHub Action that runs on PRs and flags "High Risk" code automatically.
3.  **Refactoring Recommender**:
    -   If code falls into a "Bad Pattern" cluster, perform potential refactor suggestions (e.g., "Convert recursion to iteration").
