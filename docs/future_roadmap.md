# 🚀 Future Roadmap & Optimizations

Ideas for extending CodeTurtle and optimizing its performance.

## ⚡ Performance Optimizations

1.  **Batch Processing for Embeddings** (✅ Completed):
    -   Implemented streaming generator in `src/utils.py`.
    -   Processes batches (default 32) to keep RAM usage constant.
2.  **Cached Embeddings**:
    -   Use a dedicated vector database (like FAISS or ChromaDB) instead of `.npy` files to store embeddings, allowing incremental updates instead of re-processing everything.
3.  **ONNX Export**:
    -   Convert the PyTorch CodeBERT model to ONNX runtime for 2-3x faster inference on CPU/GPU.

## 🧠 Model Improvements

1.  **Graph Neural Networks (GNNs)**:
    -   Instead of just counting AST nodes (`structural.py`), convert the AST into a graph and run a GNN (like GCN or GAT). This captures the *structure* of data flow much better than simple counts.
2.  **Contrastive Fine-Tuning**:
    -   Fine-tune CodeBERT on the specific target dataset using contrastive loss (SimCLR) to force the model to separate "clean" and "messy" code further apart.
3.  **LLM Integration**:
    -   Use a small LLM (e.g., Llama-3-8B) to generate *explanations* for why a cluster exists. "This cluster contains recursive dynamic programming solutions."

## 🛠️ Feature Extensions

1.  **Language Agnostic Support**:
    -   Currently Python-only (`ast` module).
    -   Use `tree-sitter` for parsing to support JavaScript, Go, Rust, and Java.
2.  **CI/CD Action**:
    -   Turn `collect_data.py` + `run_analysis.py` into a GitHub Action that runs on PRs and flags "High Risk" code automatically.
3.  **Refactoring Recommender**:
    -   If code falls into a "Bad Pattern" cluster, perform potential refactor suggestions (e.g., "Convert recursion to iteration").
