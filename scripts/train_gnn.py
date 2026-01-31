#!/usr/bin/env python3
"""
Train GNN to predict Code Risk.

This script:
1. Loads collected python files.
2. Converts them to AST Graphs.
3. Calculates their Rule-Based Risk Score (Ground Truth).
4. Trains the GNN to learn this structure-to-risk mapping.
"""

import sys
import argparse
from pathlib import Path
import logging
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, PROCESSED_DATA_DIR as PROCESSED_DIR
from src.features.graph_converter import ASTGraphConverter
from src.features.gnn import CodeGNN
from src.detection.rules import RiskDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.01):
    # 1. Prepare Data
    logger.info("🛠️  Preparing dataset...")
    raw_files = sorted(list(PROCESSED_DIR.rglob("*.py"))) if PROCESSED_DIR.exists() else []
    
    if not raw_files:
        # Fallback to data dir if processed dir empty
        raw_files = sorted(list(DATA_DIR.rglob("*.py")))
        
    if not raw_files:
        logger.error("❌ No .py files found in data/ or data/processed/")
        return
        
    logger.info(f"   Found {len(raw_files)} files.")
    
    converter = ASTGraphConverter()
    detector = RiskDetector()
    
    dataset = []
    
    for fpath in raw_files:
        try:
            code = fpath.read_text(errors='ignore')
            if not code.strip():
                continue
                
            # Convert to Graph
            data = converter.code_to_graph(code)
            if data is None:
                continue
                
            # Calculate Risk (Ground Truth)
            report = detector.analyze(code)
            risk_score = report.risk_score / 100.0 # Normalize 0-1
            
            # Attach label to data object
            data.y = torch.tensor([risk_score], dtype=torch.float)
            
            dataset.append(data)
            
        except Exception as e:
            logger.error(f"Failed to convert {fpath}: {e}")
            continue
            
    logger.info(f"   Successfully converted {len(dataset)} graphs.")
    
    if len(dataset) < 1:
        logger.error("❌ Dataset too small to train. Run collect_data.py first.")
        return

    # 2. Setup Training
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CodeGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    model.train()
    
    logger.info(f"🚀 Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.view(-1), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            logger.info(f"   Epoch {epoch+1:03d}: Loss = {avg_loss:.4f}")
            
    # 3. Save Model
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "gnn_model.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train GNN for Structure Learning")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.parse_args()
    
    args = parser.parse_args()
    train_model(args.epochs, args.batch_size)

if __name__ == "__main__":
    main()
