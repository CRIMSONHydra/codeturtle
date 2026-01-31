"""
Streamlit Dashboard for CodeTurtle

Interactive web interface for exploring code patterns,
clusters, and risk analysis results.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_dashboard():
    """Main dashboard entry point."""
    
    st.set_page_config(
        page_title="CodeTurtle",
        page_icon="🐢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e88e5, #43a047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #2ecc71; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🐢 CodeTurtle</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Discover Hidden Programming Patterns</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Upload Files", "Load Results", "Demo Data"]
        )
        
        if analysis_mode == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload Python Files",
                type=['py'],
                accept_multiple_files=True,
            )
        
        st.divider()
        
        st.header("📊 Settings")
        n_clusters = st.slider("Number of Clusters", 2, 15, 5)
        risk_threshold = st.slider("Risk Threshold", 0, 100, 50)
        
        clustering_algo = st.selectbox(
            "Clustering Algorithm",
            ["K-Means", "DBSCAN", "Hierarchical"]
        )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🎯 Clusters",
        "⚠️ Risk Analysis",
        "📝 Code Viewer"
    ])
    
    # Load Results Mode
    df = None
    if analysis_mode == "Load Results":
        result_path = Path("outputs/analysis_results.csv")
        if result_path.exists():
            df = pd.read_csv(result_path)
            # Map columns to dashboard expected names
            df = df.rename(columns={
                'filename': 'file',
                'max_nesting_depth': 'nesting_depth',
                'cyclomatic_complexity': 'complexity'
            })
            # Ensure essential columns exist
            required = ['file', 'cluster', 'risk_score', 'loop_count', 'nesting_depth', 'function_count', 'complexity']
            missing = [c for c in required if c not in df.columns]
            if missing:
                 st.error(f"Missing columns in results: {missing}")
                 st.stop()
        else:
            st.warning(f"No results found at {result_path}. Run analysis first!")
            st.stop()

    # Generate demo data if needed
    elif analysis_mode == "Demo Data":
        np.random.seed(42)
        n_files = 100
        
        demo_data = {
            'file': [f"file_{i}.py" for i in range(n_files)],
            'cluster': np.random.randint(0, 5, n_files),
            'risk_score': np.random.uniform(0, 100, n_files),
            'loop_count': np.random.poisson(3, n_files),
            'nesting_depth': np.random.randint(1, 8, n_files),
            'function_count': np.random.poisson(5, n_files),
            'complexity': np.random.poisson(10, n_files),
        }
        df = pd.DataFrame(demo_data)
        
    # Overview tab
    if df is not None:
        with tab1:
            st.header("📊 Analysis Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Files", len(df))
            with col2:
                st.metric("Clusters Found", df['cluster'].nunique())
            with col3:
                st.metric("High Risk Files", len(df[df['risk_score'] >= 60]))
            with col4:
                st.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}")
            
            st.divider()
            
            # Risk distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Distribution")
                fig = px.histogram(
                    df, x='risk_score',
                    nbins=20,
                    color_discrete_sequence=['#667eea'],
                )
                fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red")
                fig.update_layout(
                    xaxis_title="Risk Score",
                    yaxis_title="Count",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Cluster Distribution")
                cluster_counts = df['cluster'].value_counts().sort_index()
                fig = px.pie(
                    values=cluster_counts.values,
                    names=[f"Cluster {i}" for i in cluster_counts.index],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Clusters tab
        with tab2:
            st.header("🎯 Cluster Analysis")
            
            # Create 2D embedding for visualization
            from sklearn.decomposition import PCA
            features = df[['loop_count', 'nesting_depth', 'function_count', 'complexity']].values
            pca = PCA(n_components=2)
            coords = pca.fit_transform(features)
            df['x'] = coords[:, 0]
            df['y'] = coords[:, 1]
            
            fig = px.scatter(
                df, x='x', y='y',
                color=df['cluster'].astype(str),
                hover_data=['file', 'risk_score'],
                size='risk_score',
                size_max=20,
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(
                xaxis_title="PCA Component 1",
                yaxis_title="PCA Component 2",
                legend_title="Cluster",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster details
            st.subheader("Cluster Statistics")
            cluster_stats = df.groupby('cluster').agg({
                'risk_score': ['mean', 'std', 'max'],
                'loop_count': 'mean',
                'nesting_depth': 'mean',
                'function_count': 'mean',
                'file': 'count',
            }).round(2)
            cluster_stats.columns = ['Avg Risk', 'Risk Std', 'Max Risk', 'Avg Loops', 'Avg Depth', 'Avg Functions', 'File Count']
            st.dataframe(cluster_stats, use_container_width=True)
        
        # Risk tab
        with tab3:
            st.header("⚠️ Risk Analysis")
            
            # Sort by risk
            risky_files = df.sort_values('risk_score', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Riskiest Files")
                
                def risk_color(val):
                    if val >= 60: return 'background-color: #ffcdd2'
                    elif val >= 30: return 'background-color: #fff9c4'
                    return 'background-color: #c8e6c9'
                
                styled_df = risky_files[['file', 'risk_score', 'cluster', 'nesting_depth', 'complexity']].head(20)
                styled_df = styled_df.style.applymap(risk_color, subset=['risk_score'])
                st.dataframe(styled_df, use_container_width=True, height=400)
            
            with col2:
                st.subheader("Risk Breakdown")
                
                high_risk = len(df[df['risk_score'] >= 60])
                medium_risk = len(df[(df['risk_score'] >= 30) & (df['risk_score'] < 60)])
                low_risk = len(df[df['risk_score'] < 30])
                
                fig = go.Figure(go.Bar(
                    x=[high_risk, medium_risk, low_risk],
                    y=['High Risk', 'Medium Risk', 'Low Risk'],
                    orientation='h',
                    marker_color=['#e74c3c', '#f39c12', '#2ecc71'],
                ))
                fig.update_layout(
                    xaxis_title="Number of Files",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Common Issues")
                st.markdown("""
                - 🔴 Deep nesting (>5 levels)
                - 🟠 Long functions (>50 lines)
                - 🟡 Bare except clauses
                - 🟢 Missing type hints
                """)
        
        # Code viewer tab
        with tab4:
            st.header("📝 Code Viewer")
            
            selected_file = st.selectbox(
                "Select a file to view",
                df['file'].tolist()
            )
            
            file_info = df[df['file'] == selected_file].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{file_info['risk_score']:.1f}")
            with col2:
                st.metric("Cluster", f"#{file_info['cluster']}")
            with col3:
                st.metric("Complexity", file_info['complexity'])
            
            st.divider()
            
            # Determine path to read
            file_path_to_read = selected_file
            if 'filepath' in file_info:
                path_val = file_info['filepath']
                # Ensure path is a valid non-empty string (handles NaN/None)
                if isinstance(path_val, str) and path_val.strip():
                    file_path_to_read = path_val

            # Show actual code if file exists locally
            if Path(file_path_to_read).exists():
                 try:
                     code_content = Path(file_path_to_read).read_text(errors='ignore')
                     st.code(code_content, language='python')
                     st.caption(f"Source: {file_path_to_read}")
                 except OSError as e:
                     st.error(f"Error reading file: {e}")
            else:
                 # Demo code fallback
                demo_code = '''
def example_function(data):
    """Example function for demonstration."""
    results = []
    return results
'''
                st.code(demo_code, language='python')
                st.warning("Original source file not found locally.")

    elif analysis_mode == "Demo Data":
         pass # Already handled by creating df but we need to prevent double message
    
    else:
        with tab1:
            st.info("👆 Select 'Demo Data' in the sidebar to see the dashboard in action, or upload Python files for analysis.")


if __name__ == "__main__":
    run_dashboard()
