"""
Tests for fault tolerance and fallback mechanisms.
"""

import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import sys

# We move imports inside tests or setup to avoid module-level side effects
# during pytest collection which was causing torch registry errors.

class TestEmbedderFallback:
    """Test CPU fallback logic for embeddings."""
    
    def test_codebert_cpu_fallback(self):
        """Test that CodeBERTEmbedder falls back to CPU on CUDA error."""
        # Mocking modules for this test
        # We need to import inside the patch context
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'optimum.onnxruntime': MagicMock(),
        }):
            from src.features.embeddings import CodeBERTEmbedder
            
            # Mock _load_torch to return our mock torch
            with patch('src.features.embeddings._load_torch') as mock_load_torch:
                mock_torch = MagicMock()
                mock_torch.cuda.is_available.return_value = True
                mock_torch.no_grad = torch.no_grad 
                # Fix for total_memory f-string formatting
                mock_torch.cuda.get_device_properties.return_value.total_memory = 8.0 * 1e9
                mock_load_torch.return_value = mock_torch
                
                # Setup embedder
                embedder = CodeBERTEmbedder(use_gpu=True, use_onnx=True)
                embedder.device_obj = MagicMock()
                
                # Mock initial model (ONNX) - First call fails
                mock_model = MagicMock()
                mock_model.side_effect = RuntimeError("CUDA failure 700: an illegal memory access")
                embedder.model = mock_model
                embedder.provider = "CUDAExecutionProvider"
                # Prevent _initialize() from overwriting our mock
                embedder._initialized = True
                
                # Mock tokenizer
                mock_tokenizer = MagicMock()
                mock_tokenizer.return_value = {'input_ids': torch.tensor([[1]])}
                embedder.tokenizer = mock_tokenizer
                
                # Mock optimum for fallback
                with patch('optimum.onnxruntime.ORTModelForFeatureExtraction') as mock_ort:
                    # Setup fallback model to succeed
                    mock_fallback_model = MagicMock()
                    mock_output = MagicMock()
                    mock_output.last_hidden_state = torch.randn(1, 10, 768)
                    mock_fallback_model.return_value = mock_output
                    mock_ort.from_pretrained.return_value = mock_fallback_model
                    
                    # Execute
                    embeddings = embedder.get_embeddings_batch(["def foo(): pass"])
                    
                    # Verify fallback happened
                    assert embedder.provider == "CPUExecutionProvider"
                    mock_ort.from_pretrained.assert_called_with(
                        embedder.onnx_path,
                        provider="CPUExecutionProvider"
                    )
                    # Should have valid embedding from fallback
                    assert embeddings.shape == (1, 768)
                    assert np.any(embeddings != 0)

    def test_gnn_cpu_fallback(self):
        """Test that GNNEmbedder falls back to CPU on CUDA error."""
        with patch.dict('sys.modules', {
             # We might need to mock gnn dependencies if they are heavy
        }):
             from src.features.gnn import GNNEmbedder
             
             with patch('src.features.gnn.CodeGNN') as mock_codegnn:
                # Setup GNN embedder
                embedder = GNNEmbedder(use_gpu=True)
                
                # Mock converter
                mock_data = MagicMock()
                mock_data.x = torch.randn(5, 10)
                mock_data.edge_index = torch.zeros(2, 5).long()
                # Important: Allow .to() calls
                mock_data.to.return_value = mock_data
                
                embedder.converter = MagicMock()
                embedder.converter.code_to_graph.return_value = mock_data
                
                # Mock model to fail first, then succeed
                mock_model = MagicMock()
                
                def model_side_effect(*args):
                    # Check the device of the embedder or inputs
                    # Implementation detail: embedder tries .to(device)
                    # We simulate failure when it's on CUDA
                    if embedder.device.type == 'cuda':
                        raise RuntimeError("CUDA error: illegal memory access")
                    # Second call (CPU) -> Succeed
                    return (torch.tensor([0]), torch.randn(1, 32))
                    
                mock_model.side_effect = model_side_effect
                embedder.model = mock_model
                # Wire the CodeGNN constructor to return our mock model for the fallback path
                mock_codegnn.return_value = mock_model
                # Important: Set check flag to True since we injected a model
                embedder._has_model = True
                embedder.device = torch.device('cuda')
                
                # Execute
                embedding = embedder.get_embedding("x = 1")
                
                # Verify
                assert embedding.shape == (32,)
                # Should have tried CPU
                assert mock_model.call_count >= 1
