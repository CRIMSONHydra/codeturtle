"""
Code Embeddings using CodeBERT/UniXcoder

Generates semantic embeddings from code using transformer models,
capturing algorithmic similarity and logic patterns.

Optimized for GPU acceleration
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import logging
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    MAX_CODE_LENGTH,
    EMBEDDING_BATCH_SIZE,
    USE_GPU,
    GPU_DEVICE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy imports for torch/transformers (heavy dependencies)
_torch = None
_transformers = None


def _load_torch():
    """Lazy load torch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _load_transformers():
    """Lazy load transformers."""
    global _transformers
    if _transformers is None:
        import transformers
        _transformers = transformers
    return _transformers


class CodeBERTEmbedder:
    """
    Code embedding generator using CodeBERT/UniXcoder.
    
    Converts Python code into dense vector representations
    that capture semantic meaning and patterns.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        use_gpu: bool = USE_GPU,
        device: str = GPU_DEVICE,
        use_onnx: bool = False,
        onnx_path: str = "data/models/codebert_onnx"
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name
            use_gpu: Whether to use GPU acceleration
            device: CUDA device string
            use_onnx: Whether to use ONNX Runtime for inference
            onnx_path: Path to ONNX model directory
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = device
        self.use_onnx = use_onnx
        self.onnx_path = onnx_path
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of model and tokenizer."""
        if self._initialized:
            return
        
        torch = _load_torch()
        transformers = _load_transformers()
        
        # Determine device
        if self.use_gpu and torch.cuda.is_available():
            self.device_obj = torch.device(self.device)
            self.provider = "CUDAExecutionProvider"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device_obj = torch.device("cpu")
            self.provider = "CPUExecutionProvider"
            logger.info("Using CPU (GPU not available or disabled)")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        
        if self.use_onnx:
            logger.info(f"Loading ONNX model from: {self.onnx_path}")
            try:
                import onnxruntime as ort
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                
                # Check which providers are actually available
                available_providers = ort.get_available_providers()
                logger.debug(f"Available ONNX providers: {available_providers}")
                
                # Determine the best available provider
                if "CUDAExecutionProvider" in available_providers and self.use_gpu:
                    selected_provider = "CUDAExecutionProvider"
                    logger.info("Using ONNX with CUDA acceleration")
                elif "CPUExecutionProvider" in available_providers:
                    selected_provider = "CPUExecutionProvider"
                    if self.use_gpu:
                        logger.warning("CUDAExecutionProvider not available, falling back to CPU")
                    else:
                        logger.info("Using ONNX with CPU")
                else:
                    # Fallback to first available (e.g., AzureExecutionProvider)
                    selected_provider = available_providers[0] if available_providers else None
                    logger.warning(f"Using fallback provider: {selected_provider}")
                    
                if selected_provider is None:
                    raise RuntimeError("No ONNX execution providers available")
                    
                self.model = ORTModelForFeatureExtraction.from_pretrained(
                    self.onnx_path,
                    provider=selected_provider
                )
                logger.info(f"ONNX Model loaded with provider: {selected_provider}")
            except ImportError:
                raise ImportError("optimum[onnxruntime] required. Run: uv sync --all-extras")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                raise
        else:
            logger.info(f"Loading PyTorch model: {self.model_name}")
            self.model = transformers.AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device_obj)
            self.model.eval()
        
        self._initialized = True
        logger.info("Model loaded successfully")
    
    def get_embedding(self, code: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single code snippet.
        
        Args:
            code: Python source code
            
        Returns:
            768-dimensional numpy array or None if failed
        """
        self._initialize()
        torch = _load_torch()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=MAX_CODE_LENGTH,
                truncation=True,
                padding=True,
            )
            
            # Move to device (only for PyTorch)
            if not self.use_onnx:
                inputs = {k: v.to(self.device_obj) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use CLS token embedding (first token)
                # Shape: [batch_size, seq_len, hidden_size]
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Convert to numpy
                embedding = embeddings.cpu().numpy().squeeze()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None
    
    def get_embeddings_batch(
        self,
        codes: List[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple code snippets.
        
        Args:
            codes: List of Python source code strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress
            
        Returns:
            Array of shape (n_samples, 768)
        """
        self._initialize()
        torch = _load_torch()
        
        all_embeddings = []
        n_batches = (len(codes) + batch_size - 1) // batch_size
        
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            batch_idx = i // batch_size + 1
            
            if show_progress:
                logger.info(f"Processing batch {batch_idx}/{n_batches}")
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=MAX_CODE_LENGTH,
                    truncation=True,
                    padding=True,
                )
                
                if not self.use_onnx:
                    inputs = {k: v.to(self.device_obj) for k, v in inputs.items()}
                
                with torch.no_grad():
                    try:
                        outputs = self.model(**inputs)
                        embeddings = outputs.last_hidden_state[:, 0, :]
                        all_embeddings.append(embeddings.cpu().numpy())
                    except Exception as e:
                        error_msg = str(e)
                        is_cuda_error = "CUDA" in error_msg or "CUDNN" in error_msg
                        
                        if is_cuda_error and self.use_onnx and self.provider == "CUDAExecutionProvider":
                            logger.warning(f"CUDA Error encountered: {e}")
                            logger.warning("Attempting to fallback to CPU execution for ONNX...")
                            
                            # Re-initialize model on CPU
                            try:
                                from optimum.onnxruntime import ORTModelForFeatureExtraction
                                self.model = ORTModelForFeatureExtraction.from_pretrained(
                                    self.onnx_path,
                                    provider="CPUExecutionProvider"
                                )
                                self.provider = "CPUExecutionProvider"
                                logger.info("Successfully switched to CPU provider. Retrying batch...")
                                
                                # Retry inference
                                outputs = self.model(**inputs)
                                embeddings = outputs.last_hidden_state[:, 0, :]
                                all_embeddings.append(embeddings.cpu().numpy())
                                
                            except Exception as fallback_error:
                                logger.error(f"CPU Fallback failed: {fallback_error}")
                                zeros = np.zeros((len(batch), EMBEDDING_DIMENSION))
                                all_embeddings.append(zeros)
                        else:
                            raise e

            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
                # Add zeros for failed samples
                zeros = np.zeros((len(batch), EMBEDDING_DIMENSION))
                all_embeddings.append(zeros)
        
        return np.vstack(all_embeddings)
    
    def get_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate cosine similarity between two code snippets.
        
        Args:
            code1: First code snippet
            code2: Second code snippet
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.get_embedding(code1)
        emb2 = self.get_embedding(code2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# Simple fallback using TF-IDF (no GPU required)
class TFIDFEmbedder:
    """
    Lightweight fallback embedder using TF-IDF.
    
    Useful when GPU/transformers are not available.
    """
    
    def __init__(self, max_features: int = 768):
        self.max_features = max_features
        self.vectorizer = None
        self._fitted = False
    
    def fit(self, codes: List[str]):
        """Fit the vectorizer on a corpus of code."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
            ngram_range=(1, 2),
        )
        self.vectorizer.fit(codes)
        self._fitted = True
    
    def get_embedding(self, code: str) -> np.ndarray:
        """Get TF-IDF embedding for code."""
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")
        
        embedding = self.vectorizer.transform([code]).toarray().squeeze()
        return embedding
    
    def get_embeddings_batch(self, codes: List[str]) -> np.ndarray:
        """Get embeddings for multiple code snippets."""
        if not self._fitted:
            self.fit(codes)
        
        return self.vectorizer.transform(codes).toarray()


# Convenience function
def get_code_embedding(
    code: str,
    use_transformer: bool = True,
    embedder: Optional[Union[CodeBERTEmbedder, TFIDFEmbedder]] = None,
) -> Optional[np.ndarray]:
    """
    Get embedding for code using the best available method.
    
    Args:
        code: Python source code
        use_transformer: Whether to try transformer model first
        embedder: Optional pre-initialized embedder
        
    Returns:
        Embedding array or None
    """
    if embedder is not None:
        return embedder.get_embedding(code)
    
    if use_transformer:
        try:
            embedder = CodeBERTEmbedder()
            return embedder.get_embedding(code)
        except ImportError:
            logger.warning("Transformers not available, falling back to TF-IDF")
    
    # Fallback to TF-IDF
    tfidf = TFIDFEmbedder()
    tfidf.fit([code])
    return tfidf.get_embedding(code)


if __name__ == "__main__":
    # Test embedding generation
    test_codes = [
        '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
''',
        '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
''',
        '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
''',
    ]
    
    print("Testing CodeBERT Embedder...")
    try:
        embedder = CodeBERTEmbedder()
        
        for i, code in enumerate(test_codes):
            emb = embedder.get_embedding(code)
            if emb is not None:
                print(f"Code {i+1}: Embedding shape = {emb.shape}, norm = {np.linalg.norm(emb):.4f}")
        
        # Test similarity
        sim = embedder.get_similarity(test_codes[0], test_codes[1])
        print(f"\nSimilarity (fibonacci vs factorial): {sim:.4f}")
        
        sim2 = embedder.get_similarity(test_codes[0], test_codes[2])
        print(f"Similarity (fibonacci vs bubble_sort): {sim2:.4f}")
        
    except ImportError as e:
        print(f"Transformers not available: {e}")
        print("\nTesting TF-IDF fallback...")
        
        tfidf = TFIDFEmbedder()
        embeddings = tfidf.get_embeddings_batch(test_codes)
        print(f"TF-IDF embeddings shape: {embeddings.shape}")
