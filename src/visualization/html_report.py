"""
HTML Report Generator for CodeTurtle Analysis

Generates beautiful, interactive HTML reports from analysis results.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

import pandas as pd
import numpy as np


@dataclass
class ReportSection:
    """A section of the HTML report."""
    title: str
    content: str
    icon: str = "📊"


def generate_html_report(
    df: pd.DataFrame,
    cluster_info: List[Dict],
    output_path: Path,
    title: str = "CodeTurtle Analysis Report",
    include_plots: bool = True,
) -> Path:
    """
    Generate a comprehensive HTML report from analysis results.
    
    Args:
        df: DataFrame with analysis results
        cluster_info: List of cluster summary dicts
        output_path: Directory to save report and assets
        title: Report title
        include_plots: Whether to embed plot images
        
    Returns:
        Path to generated HTML file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    total_files = len(df)
    n_clusters = len(cluster_info)
    
    # Handle different types for is_anomaly column
    n_anomalies = 0
    if 'is_anomaly' in df.columns:
        col = df['is_anomaly']
        if col.dtype == bool:
            n_anomalies = col.sum()
        elif hasattr(col.iloc[0] if len(col) > 0 else None, 'is_anomaly'):
            # It's an AnomalyResult object
            n_anomalies = sum(1 for r in col if r.is_anomaly)
        else:
            n_anomalies = col.astype(bool).sum()
    
    avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
    high_risk = (df['risk_score'] >= 60).sum() if 'risk_score' in df.columns else 0
    
    # Top risky files
    top_risky = []
    if 'risk_score' in df.columns and 'filename' in df.columns:
        top_risky = df.nlargest(10, 'risk_score')[['filename', 'risk_score', 'cluster']].to_dict('records')
    
    # Cluster summary
    cluster_html = ""
    for c in cluster_info:
        cluster_html += f"""
        <div class="cluster-card">
            <h4>Cluster {c.get('cluster_id', 'N/A')}</h4>
            <p class="cluster-desc">{c.get('description', 'No description')}</p>
            <div class="cluster-stats">
                <span class="stat">{c.get('size', 0)} files</span>
                <span class="stat">{c.get('percentage', 0):.1f}%</span>
            </div>
        </div>
        """
    
    # Risk table
    risk_rows = ""
    for r in top_risky:
        risk_class = "high-risk" if r.get('risk_score', 0) >= 60 else "medium-risk" if r.get('risk_score', 0) >= 30 else "low-risk"
        risk_rows += f"""
        <tr class="{risk_class}">
            <td>{r.get('filename', 'Unknown')}</td>
            <td>{r.get('risk_score', 0):.1f}</td>
            <td>{r.get('cluster', 'N/A')}</td>
        </tr>
        """
    
    # Plot images
    plots_html = ""
    if include_plots:
        plot_files = ['clusters.png', 'risk_distribution.png', 'feature_importance.png']
        for plot in plot_files:
            plot_path = output_path / plot
            if plot_path.exists():
                # Use relative path
                plots_html += f"""
                <div class="plot-container">
                    <img src="{plot}" alt="{plot.replace('.png', '').replace('_', ' ').title()}" onerror="this.style.display='none'">
                </div>
                """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #0f172a;
            --card: #1e293b;
            --text: #f1f5f9;
            --muted: #94a3b8;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--primary), #7c3aed);
            border-radius: 1rem;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .header .subtitle {{
            color: rgba(255,255,255,0.8);
            font-size: 1.1rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--card);
            padding: 1.5rem;
            border-radius: 0.75rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, var(--primary), #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .stat-card .label {{
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        
        .section {{
            background: var(--card);
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .clusters-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }}
        
        .cluster-card {{
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid var(--primary);
        }}
        
        .cluster-card h4 {{
            margin-bottom: 0.5rem;
        }}
        
        .cluster-desc {{
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }}
        
        .cluster-stats {{
            display: flex;
            gap: 1rem;
        }}
        
        .cluster-stats .stat {{
            font-size: 0.85rem;
            color: var(--primary);
            font-weight: 600;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        th {{
            color: var(--muted);
            font-weight: 500;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        
        .high-risk {{ background: rgba(239, 68, 68, 0.1); }}
        .medium-risk {{ background: rgba(245, 158, 11, 0.1); }}
        .low-risk {{ background: rgba(34, 197, 94, 0.1); }}
        
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .plot-container {{
            background: rgba(255,255,255,0.02);
            border-radius: 0.5rem;
            overflow: hidden;
        }}
        
        .plot-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .footer {{
            text-align: center;
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            .header h1 {{ font-size: 1.75rem; }}
            .stat-card .value {{ font-size: 2rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐢 {title}</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{total_files}</div>
                <div class="label">Files Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="value">{n_clusters}</div>
                <div class="label">Clusters Found</div>
            </div>
            <div class="stat-card">
                <div class="value">{n_anomalies}</div>
                <div class="label">Anomalies Detected</div>
            </div>
            <div class="stat-card">
                <div class="value">{high_risk}</div>
                <div class="label">High Risk Files</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_risk:.1f}</div>
                <div class="label">Avg Risk Score</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Cluster Analysis</h2>
            <div class="clusters-grid">
                {cluster_html}
            </div>
        </div>
        
        <div class="section">
            <h2>⚠️ Top Risk Files</h2>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Risk Score</th>
                        <th>Cluster</th>
                    </tr>
                </thead>
                <tbody>
                    {risk_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Visualizations</h2>
            <div class="plots-grid">
                {plots_html if plots_html else '<p style="color: var(--muted);">No plots available. Run with --visualize flag.</p>'}
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by CodeTurtle • <a href="https://github.com/CRIMSONHydra/codeturtle" style="color: var(--primary);">GitHub</a></p>
        </div>
    </div>
</body>
</html>
"""
    
    report_file = output_path / "report.html"
    report_file.write_text(html_content)
    
    return report_file


if __name__ == "__main__":
    # Test with mock data
    import pandas as pd
    
    df = pd.DataFrame({
        'filename': ['test.py', 'main.py', 'utils.py'],
        'risk_score': [85, 42, 15],
        'cluster': [0, 1, 2],
        'is_anomaly': [True, False, False],
    })
    
    cluster_info = [
        {'cluster_id': 0, 'description': 'High complexity', 'size': 10, 'percentage': 33.3},
        {'cluster_id': 1, 'description': 'Medium complexity', 'size': 12, 'percentage': 40.0},
        {'cluster_id': 2, 'description': 'Simple code', 'size': 8, 'percentage': 26.7},
    ]
    
    report_path = generate_html_report(df, cluster_info, Path("outputs"))
    print(f"Generated: {report_path}")
