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
from config.settings import OUTPUTS_DIR


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
    
    feature_matrix = df[available_features].values
    file_names = df['filepath'].tolist() if 'filepath' in df.columns else None
    
    print(f"   Using {len(available_features)} features for clustering")
    
    # Load embeddings if provided
    current_features = [feature_matrix]
    
    if args.embeddings and args.embeddings.exists():
        print(f"\n🧠 Loading CodeBERT embeddings from {args.embeddings}...")
        embeddings = np.load(args.embeddings)
        print(f"   Loaded embeddings: {embeddings.shape}")
        
    if args.embeddings and args.embeddings.exists():
        print(f"\n🧠 Loading CodeBERT embeddings from {args.embeddings}...")
        embeddings = np.load(args.embeddings)
        print(f"   Loaded embeddings: {embeddings.shape}")
        
        from sklearn.preprocessing import StandardScaler
        scaler_emb = StandardScaler()
        emb_scaled = scaler_emb.fit_transform(embeddings)
        current_features.append(emb_scaled * 0.6) # Weight for CodeBERT
        
        # Adjust structural weight if we have embeddings
        current_features[0] = StandardScaler().fit_transform(feature_matrix) * 0.4
        
    if args.gnn_embeddings and args.gnn_embeddings.exists():
        print(f"\n🕸️  Loading GNN embeddings from {args.gnn_embeddings}...")
        gnn_embs = np.load(args.gnn_embeddings)
        print(f"   Loaded GNN embeddings: {gnn_embs.shape}")
        
        from sklearn.preprocessing import StandardScaler
        scaler_gnn = StandardScaler()
        gnn_scaled = scaler_gnn.fit_transform(gnn_embs)
        current_features.append(gnn_scaled * 0.5) # Weight for GNN
        
    # Combine all
    clustering_features = np.hstack(current_features) if len(current_features) > 1 else feature_matrix
    print(f"   Combined feature matrix: {clustering_features.shape}")
    
    # Clustering
    print(f"\n🎯 Clustering with {args.algorithm}...")
    
    cluster_result = cluster_codes(
        clustering_features,
        algorithm=args.algorithm,
        n_clusters=args.n_clusters,
        auto_k=(args.n_clusters is None),
    )
    
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
    print("\n🔍 Detecting anomalies...")
    
    anomaly_report = detect_anomalies(feature_matrix, contamination=0.1)
    df['is_anomaly'] = anomaly_report.results
    df['anomaly_score'] = [r.anomaly_score for r in anomaly_report.results]
    
    print(f"   Anomalies detected: {len(anomaly_report.anomaly_indices)}")
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    
    results_file = args.output / 'analysis_results.csv'
    df.to_csv(results_file, index=False)
    print(f"\n💾 Saved results to {results_file}")
    
    # Save cluster info
    summaries = analyze_clusters(cluster_result, feature_matrix, available_features)
    cluster_info = []
    for s in summaries:
        cluster_info.append({
            'cluster_id': s.cluster_id,
            'size': s.size,
            'percentage': s.percentage,
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
    
    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()
