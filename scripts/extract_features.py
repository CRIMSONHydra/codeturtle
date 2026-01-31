#!/usr/bin/env python3
"""
Feature Extraction Script

Extracts structural features and embeddings from collected Python files.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collector.file_filter import collect_all_python_files, get_file_info
from src.preprocessor import clean_code
from src.features import extract_structural_features, StructuralFeatures
from config.settings import OUTPUTS_DIR, RAW_DATA_DIR


from src.utils import batch_generator


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
    
    args = parser.parse_args()
    
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
            embedder = CodeBERTEmbedder()
        except ImportError as e:
            print(f"⚠️ Could not load embeddings module: {e}")
            print("   Run: uv pip install transformers torch")
            return

    # Process in batches
    print(f"\n⚙️ Processing in batches of {args.batch_size}...")
    
    all_features = []
    all_embeddings = []
    embedding_files = []
    failed_count = 0
    total_processed = 0
    
    # Iterate through batches
    num_batches = (len(files) + args.batch_size - 1) // args.batch_size
    
    for i, (paths, codes) in enumerate(batch_generator(files, args.batch_size)):
        print(f"  Batch {i+1}/{num_batches} ({len(codes)} files)...")
        
        # Clean code if requested
        if args.clean:
            cleaned_codes = []
            for code in codes:
                try:
                    cleaned_codes.append(clean_code(code))
                except:
                    cleaned_codes.append(code) # Fallback to raw if clean fails
            codes = cleaned_codes
            
        # 1. Structural Features
        for path, code in zip(paths, codes):
            try:
                features = extract_structural_features(code)
                if features:
                    row = features.to_dict()
                    row['filepath'] = str(path)
                    row['filename'] = path.name
                    all_features.append(row)
                else:
                    failed_count += 1
            except:
                failed_count += 1

        # 2. Embeddings
        if embedder:
            try:
                # Embedder handles batching internally, but we feed it our batch
                batch_embeddings = embedder.get_embeddings_batch(
                    codes, 
                    batch_size=args.batch_size, 
                    show_progress=False
                )
                
                # Careful: batch_embeddings might be smaller if some failed inside embedder?
                # Actually get_embeddings_batch returns zeros for failed ones, so length matches `codes`
                
                # We need to sync embeddings with filenames. 
                # Since we are iterating strictly parallel, logic should hold.
                
                all_embeddings.append(batch_embeddings)
                embedding_files.extend([str(p) for p in paths])
                
            except Exception as e:
                print(f"   ⚠️ Embedding batch failed: {e}")
                
        total_processed += len(codes)

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

    print("\n🎉 Feature extraction complete!")


if __name__ == "__main__":
    main()
