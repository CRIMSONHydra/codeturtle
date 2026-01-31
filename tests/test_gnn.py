"""Tests for GNN components."""

import pytest
import torch
import numpy as np
from src.features.graph_converter import ASTGraphConverter, NUM_NODE_TYPES
from src.features.gnn import CodeGNN, GNNEmbedder

class TestGraphConverter:
    def test_convert_simple_code(self):
        code = "x = 1"
        converter = ASTGraphConverter()
        data = converter.code_to_graph(code)
        
        assert data is not None
        assert data.x is not None
        assert data.edge_index is not None
        # Should have at least one node (Assign or Name or Constant)
        assert data.x.size(0) > 0
        
    def test_convert_function(self):
        code = """
def foo(x):
    if x > 0:
        return x
    return -x
"""
        converter = ASTGraphConverter()
        data = converter.code_to_graph(code)
        
        assert data is not None
        # Edges should exist (control flow)
        assert data.edge_index.size(1) > 0
        
    def test_invalid_code(self):
        code = "def broken_syntax("
        converter = ASTGraphConverter()
        data = converter.code_to_graph(code)
        assert data is None

class TestGNNModel:
    def test_model_structure(self):
        # CodeGNN determines input_dim internally from NUM_NODE_TYPES
        model = CodeGNN(hidden_channels=32)
        assert model is not None
        
        # Test forward pass with dummy data
        # Node features: [num_nodes, num_features]
        x = torch.randn(10, NUM_NODE_TYPES)
        # Edge index: [2, num_edges]
        edge_index = torch.randint(0, 10, (2, 20))
        # Batch: [num_nodes]
        batch = torch.zeros(10, dtype=torch.long)
        
        out, emb = model(x, edge_index, batch)
        
        # Risk score output should be scalar (per graph) -> [1, 1]
        assert out.shape == (1, 1)
        # Embedding should be [1, 32]
        assert emb.shape == (1, 32)
