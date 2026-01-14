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
    
    # Extract features
    print("\n⚙️ Extracting structural features...")
    
    results = []
    failed = 0
    
    for i, filepath in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(files)} files...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if args.clean:
                code = clean_code(code)
            
            features = extract_structural_features(code)
            
            if features:
                row = features.to_dict()
                row['filepath'] = str(filepath)
                row['filename'] = filepath.name
                results.append(row)
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            continue
    
    print(f"✅ Successfully extracted features from {len(results)} files")
    if failed:
        print(f"⚠️ Failed to process {failed} files")
    
    # Save to CSV
    df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n💾 Saved features to {args.output}")
    
    # Extract embeddings if requested
    if args.embeddings:
        print("\n🧠 Generating CodeBERT embeddings...")
        try:
            from src.features import CodeBERTEmbedder
            
            embedder = CodeBERTEmbedder()
            
            codes = []
            valid_files = []
            
            for filepath in files[:len(results)]:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()
                    if args.clean:
                        code = clean_code(code)
                    codes.append(code)
                    valid_files.append(str(filepath))
                except:
                    continue
            
            embeddings = embedder.get_embeddings_batch(codes)
            
            # Save embeddings
            emb_output = args.output.parent / 'embeddings.npy'
            np.save(emb_output, embeddings)
            
            # Save file mapping
            mapping_output = args.output.parent / 'embedding_files.json'
            with open(mapping_output, 'w') as f:
                json.dump(valid_files, f, indent=2)
            
            print(f"✅ Saved embeddings to {emb_output}")
            print(f"   Shape: {embeddings.shape}")
            
        except ImportError as e:
            print(f"⚠️ Could not load embeddings module: {e}")
            print("   Run: uv pip install transformers torch")
    
    # Print summary
    print("\n📊 Feature Summary:")
    print(f"   Total files: {len(results)}")
    print(f"   Feature columns: {len(df.columns) - 2}")  # Exclude filepath, filename
    
    # Show some stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
    print("\n   Sample statistics:")
    for col in numeric_cols:
        print(f"   - {col}: mean={df[col].mean():.2f}, max={df[col].max()}")
    
    print("\n🎉 Feature extraction complete!")


if __name__ == "__main__":
    main()
