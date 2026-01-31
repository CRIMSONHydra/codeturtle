#!/usr/bin/env python3
"""
Full Analysis Pipeline

Runs the complete CodeTurtle analysis:
1. Load features
2. Cluster code patterns
3. Detect risks
4. Generate visualizations
5. Create reports
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import cluster_codes, analyze_clusters, print_cluster_report
from src.detection import check_code_risks, detect_anomalies
from src.features import StructuralFeatures
from src.visualization.plots import create_summary_dashboard
from src.visualization.html_report import generate_html_report
from config.settings import OUTPUTS_DIR
from config.project_config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Run full CodeTurtle analysis pipeline"
    )
    parser.add_argument(
        '--features',
        type=Path,
        default=OUTPUTS_DIR / 'features.csv',
        help='Input features CSV file'
    )
    parser.add_argument(
        '--embeddings',
        type=Path,
        default=None,
        help='Optional embeddings file (.npy)'
    )
    parser.add_argument(
        '--gnn-embeddings',
        type=Path,
        default=None,
        help='Optional GNN structural embeddings file (.npy)'
    )
    parser.add_argument(
        '--algorithm',
        choices=['kmeans', 'dbscan', 'hierarchical'],
        default='kmeans',
        help='Clustering algorithm'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters (auto-detect if not specified)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=OUTPUTS_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Print detailed analysis report'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML report'
    )
    parser.add_argument(
        '--anomaly-algorithm',
        choices=['isolation_forest', 'one_class_svm', 'ensemble'],
        default='ensemble',
        help='Anomaly detection algorithm (default: ensemble)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to codeturtle.yaml config file'
    )
    
    args = parser.parse_args()
    
    print("\n🐢 CodeTurtle Analysis Pipeline")
    print("=" * 50)
    
    # Load features
    print(f"\n📊 Loading features from {args.features}...")
    
    if not args.features.exists():
        print(f"❌ Features file not found: {args.features}")
        print("   Run: python scripts/extract_features.py first")
        return
    
    df = pd.read_csv(args.features)
    print(f"   Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Get feature matrix
    feature_names = StructuralFeatures.feature_names()
    available_features = [f for f in feature_names if f in df.columns]
    
    if not available_features:
        print("❌ No structural features found in CSV!")
        return
    
    feature_matrix = df[available_features].values.astype(float)
    file_names = df['filepath'].tolist() if 'filepath' in df.columns else None
    
    print(f"   Using {len(available_features)} features for clustering")
    
    # Load embeddings if provided
    current_features = [feature_matrix]
    struct_weight = 1.0
    
    # Guard: fail fast if embeddings requested but filepath column missing
    if (args.embeddings or args.gnn_embeddings) and file_names is None:
        raise ValueError(
            "Embeddings requested but 'filepath' column missing from features CSV. "
            "Cannot align embeddings without file paths."
        )
    
    # Helper to align embeddings
    def load_and_align_embeddings(emb_path: Path, target_files: list, scaling_factor: float):
        if not emb_path.exists():
            return None
            
        print(f"\n   Loading embeddings from {emb_path}...")
        embeddings = np.load(emb_path)
        
        # Check for sidecar mapping
        # Try both naming conventions: `embedding_files.json` or `gnn_files.json`
        # Using name convention relative to the input file
        mapping_path = None
        if "gnn" in emb_path.name:
            mapping_path = emb_path.parent / "gnn_files.json"
        elif "embeddings" in emb_path.name:
            mapping_path = emb_path.parent / "embedding_files.json"
            
        if not mapping_path or not mapping_path.exists():
            print(f"      ⚠️ No sidecar mapping found ({mapping_path}), assuming direct alignment.")
            if embeddings.shape[0] != len(target_files):
                 print(f"      ❌ Shape mismatch: Embeddings {embeddings.shape[0]} vs Data {len(target_files)}")
                 return None
            # Standardize even in direct-alignment branch
            from sklearn.preprocessing import StandardScaler
            scaled = StandardScaler().fit_transform(embeddings)
            return scaled * scaling_factor

        print(f"      Using mapping file: {mapping_path}")
        with open(mapping_path, 'r') as f:
            emb_files = json.load(f)
        
        # Validate lengths match before creating map
        if len(emb_files) != embeddings.shape[0]:
            raise ValueError(
                f"Mapping file has {len(emb_files)} entries but embeddings has "
                f"{embeddings.shape[0]} rows. Files are out of sync."
            )
            
        # Create map: filepath -> embedding vector
        # Using string representation to match CSV filepath column
        emb_map = {str(f): emb for f, emb in zip(emb_files, embeddings, strict=True)}
        
        aligned_embeddings = []
        missing_count = 0
        
        # Align with target_files (df['filepath'])
        # target_files contains strings of filepaths
        
        for fpath in target_files:
            fpath_str = str(fpath)
            if fpath_str in emb_map:
                aligned_embeddings.append(emb_map[fpath_str])
            else:
                # If missing, fill with zeros
                aligned_embeddings.append(np.zeros(embeddings.shape[1]))
                missing_count += 1
                
        if missing_count > 0:
            print(f"      ⚠️ {missing_count} files missing from embeddings (filled with zeros)")
        else:
            print(f"      ✅ All {len(target_files)} files aligned successfully")

        # Create aligned array
        aligned = np.array(aligned_embeddings)
        
        # Final validation check
        if aligned.shape[0] != len(target_files):
             raise ValueError(f"Alignment failed: Result has {aligned.shape[0]} rows, expected {len(target_files)}")
             
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(aligned)
        return scaled * scaling_factor

    # 1. CodeBERT Embeddings
    if args.embeddings:
        # Use simple file_names list which comes from df['filepath']
        emb_data = load_and_align_embeddings(args.embeddings, file_names, 0.6)
        if emb_data is not None:
            current_features.append(emb_data)
            struct_weight = 0.4 
            
    # 2. GNN Embeddings
    if args.gnn_embeddings:
        gnn_data = load_and_align_embeddings(args.gnn_embeddings, file_names, 0.5)
        if gnn_data is not None:
            current_features.append(gnn_data)

    # Apply scaling to structural features
    from sklearn.preprocessing import StandardScaler
    current_features[0] = StandardScaler().fit_transform(feature_matrix) * struct_weight
    
    # Validation
    base_rows = current_features[0].shape[0]
    for i, feats in enumerate(current_features[1:], 1):
        if feats.shape[0] != base_rows:
            raise ValueError(f"Feature mismatch! Structural has {base_rows} rows, but input #{i} has {feats.shape[0]}")

    # Combine all
    clustering_features = np.hstack(current_features)
    print(f"   Combined feature matrix: {clustering_features.shape}")
    
    # Clustering
    print(f"\n🎯 Clustering with {args.algorithm}...")
    
    from src.clustering.clusterer import CodeClusterer
    clusterer = CodeClusterer(normalize=False, reduce_dims=True) # Normalize already done
    
    # Manually preprocess to get features matching cluster centers
    # Re-instantiate standard scaler to ensure consistency if clusterer uses it, 
    # but since we set normalize=False, we trust our manual scaling above.
    # However, PCA happens inside. We need the PCA-transformed features.
    
    # Let's use the clusterer's public API but we need the features it used.
    # CodeClusterer doesn't expose _preprocess publicly in a way that returns the fitted model easily
    # unless we check internal state.
    
    # Alternative: Do PCA here if needed.
    pca_features = clustering_features
    if clustering_features.shape[1] > 50:
         from sklearn.decomposition import PCA
         print(f"   Reducing dimensions from {clustering_features.shape[1]} to 50...")
         pca = PCA(n_components=50)
         pca_features = pca.fit_transform(clustering_features)
    
    # Now use clusterer with reduce_dims=False since we did it.
    clusterer = CodeClusterer(normalize=False, reduce_dims=False)
    
    if args.algorithm == 'kmeans':
         if args.n_clusters is None:
             args.n_clusters, _ = clusterer.find_optimal_k(pca_features)
             print(f"   Optimal k = {args.n_clusters}")
         cluster_result = clusterer.kmeans(pca_features, n_clusters=args.n_clusters)
         
    elif args.algorithm == 'dbscan':
         cluster_result = clusterer.dbscan(pca_features)
         
    elif args.algorithm == 'hierarchical':
         n = args.n_clusters or 7 # Default
         cluster_result = clusterer.hierarchical(pca_features, n_clusters=n)
         
    # Update clustering_features to match the space used for clustering (PCA space)
    # This allows analyze_clusters to work correctly with cluster_centers
    clustering_features = pca_features
    
    print(f"   Found {cluster_result.n_clusters} clusters")
    print(f"   Cluster sizes: {cluster_result.get_cluster_sizes()}")
    if 'silhouette' in cluster_result.metrics:
        print(f"   Silhouette score: {cluster_result.metrics['silhouette']:.4f}")
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_result.labels
    
    # Risk detection
    print("\n⚠️ Detecting code risks...")
    
    risk_scores = []
    risk_findings = []
    
    for filepath in (file_names or []):
        try:
            path = Path(filepath)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()
                report = check_code_risks(code, str(path))
                risk_scores.append(report.risk_score)
                risk_findings.append(len(report.findings))
            else:
                risk_scores.append(0)
                risk_findings.append(0)
        except:
            risk_scores.append(0)
            risk_findings.append(0)
    
    if risk_scores:
        df['risk_score'] = risk_scores
        df['risk_findings'] = risk_findings
        print(f"   Average risk score: {np.mean(risk_scores):.1f}")
        print(f"   High-risk files (>=60): {sum(1 for s in risk_scores if s >= 60)}")
    
    # Anomaly detection
    print(f"\\n🔍 Detecting anomalies (algorithm: {args.anomaly_algorithm})...")
    
    anomaly_report = detect_anomalies(
        feature_matrix, 
        contamination=0.1,
        algorithm=args.anomaly_algorithm
    )
    df['is_anomaly'] = anomaly_report.results
    df['anomaly_score'] = [r.anomaly_score for r in anomaly_report.results]
    
    print(f"   Anomalies detected: {len(anomaly_report.anomaly_indices)}")
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    
    results_file = args.output / 'analysis_results.csv'
    df.to_csv(results_file, index=False)
    print(f"\n💾 Saved results to {results_file}")
    
    # Save cluster info
    # For meaningful cluster descriptions, use the ORIGINAL structural features
    # not the PCA-reduced clustering_features. This gives human-readable labels.
    # We still use cluster_result.labels which map correctly to the samples.
    summaries = analyze_clusters(cluster_result, feature_matrix, available_features)
    cluster_info = []
    for s in summaries:
        cluster_info.append({
            'cluster_id': int(s.cluster_id),
            'size': int(s.size),
            'percentage': float(s.percentage),
            'description': s.description,
        })
    
    with open(args.output / 'cluster_info.json', 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    # Generate visualizations
    if args.visualize:
        print("\n📈 Generating visualizations...")
        
        risk_array = np.array(risk_scores) if risk_scores else np.random.uniform(0, 50, len(df))
        
        figures = create_summary_dashboard(
            cluster_labels=cluster_result.labels,
            risk_scores=risk_array,
            features=feature_matrix,
            feature_names=available_features,
            file_names=file_names,
            save_dir=args.output,
        )
        print(f"   Created {len(figures)} visualization plots")
    
    # Print report
    if args.report:
        print_cluster_report(summaries)
        
        print("\n📋 ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total files analyzed: {len(df)}")
        print(f"Clusters identified: {cluster_result.n_clusters}")
        print(f"Anomalies detected: {len(anomaly_report.anomaly_indices)}")
        if risk_scores:
            print(f"High-risk files: {sum(1 for s in risk_scores if s >= 60)}")
        print(f"\nResults saved to: {args.output}")
    
    # Generate HTML report
    if args.html:
        print("\n📄 Generating HTML report...")
        html_path = generate_html_report(
            df=df,
            cluster_info=cluster_info,
            output_path=args.output,
            include_plots=args.visualize,
        )
        print(f"   HTML report: {html_path}")
    
    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()
