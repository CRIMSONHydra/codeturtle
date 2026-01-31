"""
Tests for fault tolerance and fallback mechanisms.
"""

import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Mocking imports that might require GPU/heavy libs
with patch.dict('sys.modules', {
    'transformers': MagicMock(),
    'optimum.onnxruntime': MagicMock(),
}):
    from src.features.embeddings import CodeBERTEmbedder
    from src.features.gnn import GNNEmbedder

class TestEmbedderFallback:
    """Test CPU fallback logic for embeddings."""
    
    @patch('src.features.embeddings._load_torch')
    def test_codebert_cpu_fallback(self, mock_load_torch):
        """Test that CodeBERTEmbedder falls back to CPU on CUDA error."""
        # Mock torch
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.no_grad = torch.no_grad 
        mock_load_torch.return_value = mock_torch
        
        # Setup embedder
        embedder = CodeBERTEmbedder(use_gpu=True, use_onnx=True)
        embedder.device_obj = MagicMock()
        
        # Mock initial model (ONNX) - First call fails
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("CUDA failure 700: an illegal memory access")
        embedder.model = mock_model
        embedder.provider = "CUDAExecutionProvider"
        
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

    @patch('src.features.gnn.CodeGNN')
    def test_gnn_cpu_fallback(self, mock_codegnn):
        """Test that GNNEmbedder falls back to CPU on CUDA error."""
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
            # First call (GPU) -> Fail
            if embedder.device.type == 'cuda':
                raise RuntimeError("CUDA error: illegal memory access")
            # Second call (CPU) -> Succeed
            return (torch.tensor([0]), torch.randn(1, 32))
            
        mock_model.side_effect = model_side_effect
        embedder.model = mock_model
        embedder.device = torch.device('cuda')
        
        # Execute
        embedding = embedder.get_embedding("x = 1")
        
        # Verify
        assert embedding.shape == (32,)
        # Should have tried CPU
        assert mock_model.call_count >= 1
