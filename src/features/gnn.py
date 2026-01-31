"""
Graph Neural Network for Code Analysis.

Learns structural embeddings from AST graphs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

from src.features.graph_converter import NUM_NODE_TYPES, ASTGraphConverter


class CodeGNN(torch.nn.Module):
    """
    GCN-based model for learning code structure.
    
    Architecture:
    Input (One-Hot Node Types) -> GCN -> GCN -> GCN -> Global Pooling -> Linear -> Output
    """
    def __init__(self, hidden_channels: int = 64, out_channels: int = 32, num_classes: int = 1):
        super(CodeGNN, self).__init__()
        torch.manual_seed(42)
        
        # Input dimension is the number of unique AST node types (one-hot encoded)
        input_dim = NUM_NODE_TYPES
        
        # Graph Convolution Layers
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels) # Output of this is the "Structural Embedding"
        
        # Head for self-supervised task (Risk Score Prediction)
        self.lin = Linear(out_channels, num_classes)

    def forward(self, x, edge_index, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, num_node_types]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] automatically created by DataLoader
            
        Returns:
            out: Prediction (Risk Score)
            embedding: The learned structural embedding (after pooling)
        """
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # 2. Readout layer (Pooling)
        # Aggregates all node embeddings into a single graph embedding
        embedding = global_mean_pool(x, batch)  # [batch_size, out_channels]
        
        # 3. Final Classifier (for training)
        out = F.dropout(embedding, p=0.5, training=self.training)
        out = self.lin(out)
        
        return out, embedding


class GNNEmbedder:
    """Inference wrapper for trained CodeGNN."""
    
    def __init__(self, model_path: str = "data/models/gnn_model.pt", use_gpu: bool = False):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Load Model
        try:
            self.model = CodeGNN().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            # Full match expected (strict=True is default)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._has_model = True
        except (RuntimeError, FileNotFoundError, AttributeError) as e:
            logger.warning(f"Failed to load GNN model from {model_path}: {e}") 
            self._has_model = False
            
        self.converter = ASTGraphConverter(use_gpu=use_gpu)
        
    def get_embedding(self, code: str) -> np.ndarray:
        """Get structural embedding for a single code snippet."""
        if not self._has_model:
            return np.zeros(32) # Default codeGNN out_channels
            
        data = self.converter.code_to_graph(code)
        if data is None:
            return np.zeros(32)
        
        try:
            data = data.to(self.device)
            
            # Batch vector for single graph (all zeros)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                _, embedding = self.model(data.x, data.edge_index, batch)
                
            return embedding.cpu().numpy().squeeze()
            
        except RuntimeError as e:
            # CUDA errors - try CPU fallback
            if "CUDA" in str(e) or "cuda" in str(e):
                try:
                    # Clear CUDA cache and try on CPU
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Move model and data to CPU for this inference
                    data_cpu = data.to('cpu')
                    batch_cpu = torch.zeros(data_cpu.x.size(0), dtype=torch.long, device='cpu')
                    
                    # Temporarily move model to CPU
                    model_cpu = self.model.to('cpu')
                    with torch.no_grad():
                        _, embedding = model_cpu(data_cpu.x, data_cpu.edge_index, batch_cpu)
                    
                    # Move model back to GPU
                    self.model.to(self.device)
                    
                    return embedding.numpy().squeeze()
                except Exception as e_cpu:
                    print(f"⚠️ GNN CPU fallback failed: {e_cpu}")
                    # Ensure model is back on device even if CPU inference failed
                    if hasattr(self, 'model'):
                        self.model.to(self.device)
            
            return np.zeros(32)

