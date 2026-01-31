================================================================================
                               CODE TURTLE 🐢                                 
================================================================================
Team Leader: Naverdo
Email: naverdo.24bcs10076@sst.scaler.com 
Project: CodeTurtle - Intelligent Pattern Discovery in Software Repositories

Hello! 👋 
This archive contains the complete source code, analysis results, and report for
the CodeTurtle project. Follow the steps below to run the system.

================================================================================
1. PREREQUISITES
================================================================================
You need Python 3.10 or higher installed on your system.
We recommend using a virtual environment.

[Option A] Using Standard Pip (Recommended):
   1. Create a virtual environment:
      python3 -m venv venv
      source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

   2. Install dependencies:
      pip install -r requirements.txt

[Option B] Using uv (Faster):
   If you have `uv` installed:
   uv sync

================================================================================
2. RUNNING THE ANALYSIS
================================================================================
To run the full pipeline (Parsing -> Embedding -> Clustering -> Risk Analysis),
execute the following command from the project root:

   python3 scripts/run_analysis.py --clean --visualize --report

This will:
1. Load the dataset (included in `data/raw`).
2. Train/Load the GNN and CodeBERT models.
3. Perform DBSCAN clustering.
4. Generate the analysis results in `outputs/`.

*Note: The project comes pre-loaded with trained models in `data/models` and 
results in `outputs/` so you can skip this step if you just want to view the UI.*

*Note: Refer to `docs/usage_guide.md` for a step-by-step guide to running the analysis.*

================================================================================
3. LAUNCHING THE DASHBOARD
================================================================================
To explore the findings interactively (Clusters, Risk Scores, Code Viewer):

   streamlit run src/visualization/dashboard.py

Then open your browser to the URL shown (usually http://localhost:8501).

Dashboard Features:
- "Overview": High-level stats.
- "Clusters": Interactive 2D scatter plot of the 19 code clusters.
- "Risk Analysis": Table of high-risk files.
- "Code Viewer": Select a file to see its source code and metrics.
  *Tip: Use "Load Results" mode in the sidebar (selected by default).*

================================================================================
4. PROJECT REPORT & ARTIFACTS
================================================================================
- Full Report: `PROJECT_REPORT.md` (Root directory)
- Visuals:     `docs/images/` (Contains exported plots)
- Results:     `outputs/analysis_results.csv` (Raw data)
- Usage Guide: `docs/usage_guide.md` (Detailed workflow instructions)

Thank you for reviewing CodeTurtle! 🐢
