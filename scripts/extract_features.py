#!/usr/bin/env python3
"""
Feature Extraction Script

Extracts structural features and embeddings from collected Python files.
"""

import argparse
import sys
import json
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collector.file_filter import collect_all_python_files, get_file_info
from src.preprocessor import clean_code
from src.features import extract_structural_features, StructuralFeatures
from config.settings import OUTPUTS_DIR, RAW_DATA_DIR


from src.utils import batch_generator
from src.features.vector_store import CodeVectorStore, compute_code_hash


def _extract_single_file(args_tuple):
    """Helper function for parallel extraction. Must be at module level for pickling."""
    path, code, do_clean = args_tuple
    try:
        if do_clean:
            try:
                code = clean_code(code)
            except Exception:
                pass  # Use original if clean fails
        
        features = extract_structural_features(code)
        if features:
            row = features.to_dict()
            row['filepath'] = str(path)
            row['filename'] = path.name
            return ('ok', row)
        return ('fail', str(path))
    except Exception as e:
        return ('fail', str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from Python code files"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=RAW_DATA_DIR,
        help='Input directory with Python files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=OUTPUTS_DIR / 'features.csv',
        help='Output CSV file for features'
    )
    parser.add_argument(
        '--embeddings',
        action='store_true',
        help='Also generate CodeBERT embeddings (requires GPU)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean code (remove comments/docstrings) before analysis'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Number of files to process per batch (default: 32)'
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Use ChromaDB to cache embeddings (skip unchanged files)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the embedding cache before processing'
    )
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Use ONNX Runtime for accelerated inference'
    )
    parser.add_argument(
        '--gnn',
        action='store_true',
        help='Generate GNN structural embeddings (requires trained GNN model)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=0,
        help='Number of parallel workers for structural features (0=sequential, -1=auto)'
    )
    
    args = parser.parse_args()

    if args.batch_size < 1:
        print(f"❌ Error: --batch-size must be at least 1 (got {args.batch_size})")
        return
    
    print("\n🐢 CodeTurtle Feature Extraction")
    print("=" * 50)
    
    # Collect files
    print(f"\n📁 Scanning {args.input}...")
    files = collect_all_python_files(args.input, limit=args.limit)
    print(f"Found {len(files)} Python files")
    
    if not files:
        print("❌ No files found!")
        return
    
    # Initialize embedder if needed
    embedder = None
    if args.embeddings:
        print("\n🧠 Initializing CodeBERT model...")
        try:
            from src.features import CodeBERTEmbedder
            embedder = CodeBERTEmbedder(use_onnx=args.onnx)
        except ImportError as e:
            print(f"⚠️ Could not load embeddings module: {e}")
            print("   Run: uv pip install transformers torch")
            return

    # Initialize GNN if needed
    gnn_embedder = None
    if args.gnn:
        print("\n🕸️  Initializing GNN model...")
        try:
            from src.features.gnn import GNNEmbedder
            # Use CPU for GNN when ONNX is enabled to avoid CUDA conflicts
            # GNN is fast enough on CPU for graph processing
            gnn_use_gpu = not args.onnx and (args.embeddings or torch.cuda.is_available())
            gnn_embedder = GNNEmbedder(use_gpu=gnn_use_gpu)
            if gnn_use_gpu:
                print("   Using GPU for GNN")
            else:
                print("   Using CPU for GNN (avoids CUDA conflicts with ONNX)")
            if not gnn_embedder._has_model:
                print("   ⚠️  GNN model not found or failed to load. Skipping GNN.")
                gnn_embedder = None
        except ImportError as e:
            print(f"⚠️ Could not load GNN module: {e}")
            gnn_embedder = None

    # Initialize vector store if caching is enabled
    vector_store = None
    if args.cache and args.embeddings:
        from config.settings import DATA_DIR
        store_path = DATA_DIR / "vector_store"
        vector_store = CodeVectorStore(persist_directory=store_path)
        
        if args.clear_cache:
            print("\n🗑️ Clearing embedding cache...")
            vector_store.clear()
        else:
            cached_count = vector_store.count()
            print(f"\n💾 Embedding cache: {cached_count} files stored")

    # Process in batches
    print(f"\n⚙️ Processing in batches of {args.batch_size}...")
    
    all_features = []
    all_embeddings = []
    embedding_files = []
    all_gnn_embeddings = []
    gnn_files = []
    failed_count = 0
    total_processed = 0
    
    # Iterate through batches
    num_batches = (len(files) + args.batch_size - 1) // args.batch_size
    
    # Initialize parallel executor once if needed
    n_workers = args.parallel
    if n_workers == -1:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    executor = None
    if n_workers > 0:
        executor = ProcessPoolExecutor(max_workers=n_workers)
        print(f"🚀 Using {n_workers} workers for structural extraction")

    try:
        for i, (paths, codes) in enumerate(tqdm(
            batch_generator(files, args.batch_size),
            total=num_batches,
            desc="Processing batches",
            unit="batch"
        )):
            
            # 1. Structural Features (with optional parallel processing)
            if executor:
                # Parallel extraction
                tasks = [(p, c, args.clean) for p, c in zip(paths, codes)]
                # Using map directly on the existing executor
                for result in executor.map(_extract_single_file, tasks):
                    if result[0] == 'ok':
                        all_features.append(result[1])
                    else:
                        failed_count += 1
            else:
                # Sequential extraction (original behavior)
                if args.clean:
                    cleaned_codes = []
                    for code in codes:
                        try:
                            cleaned_codes.append(clean_code(code))
                        except Exception as e:
                            cleaned_codes.append(code)
                    codes = cleaned_codes
                
            for path, code in zip(paths, codes, strict=True):
                try:
                    features = extract_structural_features(code)
                    if features:
                        row = features.to_dict()
                        row['filepath'] = str(path)
                        row['filename'] = path.name
                        all_features.append(row)
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1

        # 2. Embeddings
        if embedder:
            if vector_store:
                # 1. Identify what needs to be computed
                emb_paths, emb_codes, _ = vector_store.filter_new_or_changed(paths, codes)
                skipped = len(paths) - len(emb_paths)
                
                if skipped > 0:
                    print(f"      ⏭️ Retrieved {skipped} cached files")
                
                # 2. Compute & Upsert new embeddings
                if len(emb_codes) > 0:
                    try:
                        new_embeddings = embedder.get_embeddings_batch(
                            emb_codes, 
                            batch_size=args.batch_size, 
                            show_progress=False
                        )
                        vector_store.add_embeddings(emb_paths, emb_codes, new_embeddings)
                    except Exception as e:
                        print(f"   ⚠️ Embedding computation/storage failed: {e}")
                        # Continue to try retiring existing ones? Or skip batch?
                        # If computation failed, we might miss these in retrieval.
                
                # 3. Retrieve ALL embeddings for this batch (cached + new)
                # This ensures output matches the input file list perfectly
                try:
                    batch_full_paths = [str(p) for p in paths]
                    retrieved = vector_store.get_embeddings(batch_full_paths)
                    
                    # Convert to numpy, handling potential missing ones (if computation failed)
                    valid_batch_embeddings = []
                    valid_batch_files = []
                    
                    for idx, emb in enumerate(retrieved):
                        if emb is not None:
                            valid_batch_embeddings.append(emb)
                            valid_batch_files.append(batch_full_paths[idx])
                        else:
                            # This implies computation failed for this file
                            print(f"      ⚠️ Failed to retrieve embedding for {batch_full_paths[idx]}")
                            
                    if valid_batch_embeddings:
                        all_embeddings.append(np.array(valid_batch_embeddings))
                        embedding_files.extend(valid_batch_files)
                        
                except Exception as e:
                    print(f"   ⚠️ Cache retrieval failed: {e}")

            else:
                # No Caching - Standard Stream
                try:
                    batch_embeddings = embedder.get_embeddings_batch(
                        codes, 
                        batch_size=args.batch_size, 
                        show_progress=False
                    )
                    all_embeddings.append(batch_embeddings)
                    embedding_files.extend([str(p) for p in paths])
                    
                except Exception as e:
                    print(f"   ⚠️ Embedding batch failed: {e}")
                
        # 3. GNN Embeddings
        if gnn_embedder:
            batch_gnn = []
            batch_gnn_files = []
            
            for path, code in zip(paths, codes, strict=True):
                try:
                    gnn_emb = gnn_embedder.get_embedding(code)
                    batch_gnn.append(gnn_emb)
                    batch_gnn_files.append(str(path))
                except Exception as e:
                    # Log failure and append zero vector to keep alignment
                    logger.warning(f"GNN failed for {path}: {e}")
                    
                    # Determine dimension for zero vector
                    dim = 32 # Default CodeGNN output
                    if batch_gnn:
                        dim = batch_gnn[0].shape[0]
                    elif all_gnn_embeddings:
                         dim = all_gnn_embeddings[0].shape[1]
                         
                    batch_gnn.append(np.zeros(dim))
                    batch_gnn_files.append(str(path))
            
            if batch_gnn:
                all_gnn_embeddings.append(np.array(batch_gnn))
                gnn_files.extend(batch_gnn_files)
                
        total_processed += len(codes)
    finally:
        if executor:
            executor.shutdown()
            print("\n🛑 Parallel executor shut down")

    print(f"✅ Successfully extracted features from {len(all_features)} files")
    if failed_count:
        print(f"⚠️ Failed to process {failed_count} files")
    
    # Save Features CSV
    if all_features:
        df = pd.DataFrame(all_features)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\n💾 Saved features to {args.output}")
        
        # Show stats
        print("\n📊 Feature Summary:")
        print(f"   Feature columns: {len(df.columns) - 2}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        print("   Sample statistics:")
        for col in numeric_cols:
            print(f"   - {col}: mean={df[col].mean():.2f}")

    # Save Embeddings
    if all_embeddings:
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)
        
        emb_output = args.output.parent / 'embeddings.npy'
        np.save(emb_output, final_embeddings)
        
        mapping_output = args.output.parent / 'embedding_files.json'
        with open(mapping_output, 'w') as f:
            json.dump(embedding_files, f, indent=2)
            
        print(f"\n💾 Saved embeddings to {emb_output}")
        print(f"   Shape: {final_embeddings.shape}")

    # Save GNN Embeddings
    if all_gnn_embeddings:
        final_gnn = np.vstack(all_gnn_embeddings)
        gnn_output = args.output.parent / 'gnn_embeddings.npy'
        np.save(gnn_output, final_gnn)
        
        gnn_map_output = args.output.parent / 'gnn_files.json'
        with open(gnn_map_output, 'w') as f:
            json.dump(gnn_files, f, indent=2)
            
        print(f"\n💾 Saved GNN embeddings to {gnn_output}")
        print(f"   Shape: {final_gnn.shape}")

    print("\n🎉 Feature extraction complete!")


if __name__ == "__main__":
    main()
