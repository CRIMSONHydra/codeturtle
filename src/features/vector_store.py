"""
Vector Store for Cached Code Embeddings using ChromaDB.

Provides persistent storage and retrieval of code embeddings,
allowing the extraction script to skip files that haven't changed.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

# Import embedding dimension for shape safety
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# Lazy import for chromadb
_chromadb = None


def _load_chromadb():
    """Lazy load chromadb."""
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


def compute_code_hash(code: str) -> str:
    """
    Compute SHA-256 hash of code content.
    
    Args:
        code: Source code string
        
    Returns:
        Hex digest of the hash
    """
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


class CodeVectorStore:
    """
    Persistent vector store for code embeddings using ChromaDB.
    
    Stores embeddings alongside metadata (filepath, code hash) to enable:
    - Fast lookup of existing embeddings
    - Change detection via hash comparison
    - Incremental updates (only process new/changed files)
    """
    
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "code_embeddings",
        embedding_dim: int = EMBEDDING_DIMENSION,
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            embedding_dim: Dimension of embeddings (default: from settings)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._client = None
        self._collection = None
        
    def _initialize(self):
        """Lazy initialization of ChromaDB client and collection."""
        if self._client is not None:
            return
            
        chromadb = _load_chromadb()
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "CodeTurtle code embeddings cache"}
        )
        
        logger.info(f"Initialized ChromaDB at {self.persist_directory}")
        logger.info(f"Collection '{self.collection_name}' has {self._collection.count()} items")
    
    def get_cached_hashes(self) -> Dict[str, str]:
        """
        Get all cached file paths and their code hashes.
        
        Returns:
            Dict mapping filepath -> code_hash
        """
        self._initialize()
        
        # Get all items from collection
        results = self._collection.get(include=["metadatas"])
        
        if not results["ids"]:
            return {}
        
        cached = {}
        for id_, metadata in zip(results["ids"], results["metadatas"]):
            if metadata and "filepath" in metadata and "code_hash" in metadata:
                cached[metadata["filepath"]] = metadata["code_hash"]
        
        return cached
    
    def filter_new_or_changed(
        self,
        files: List[Path],
        codes: List[str],
    ) -> Tuple[List[Path], List[str], List[int]]:
        """
        Filter files to only those that are new or have changed.
        
        Args:
            files: List of file paths
            codes: List of corresponding code contents
            
        Returns:
            Tuple of (filtered_files, filtered_codes, original_indices)
        """
        cached_hashes = self.get_cached_hashes()
        
        new_files = []
        new_codes = []
        new_indices = []
        
        for i, (file, code) in enumerate(zip(files, codes, strict=True)):
            filepath_str = str(file)
            current_hash = compute_code_hash(code)
            
            cached_hash = cached_hashes.get(filepath_str)
            
            if cached_hash is None:
                # New file
                new_files.append(file)
                new_codes.append(code)
                new_indices.append(i)
                logger.debug(f"New file: {file.name}")
            elif cached_hash != current_hash:
                # Changed file
                new_files.append(file)
                new_codes.append(code)
                new_indices.append(i)
                logger.debug(f"Changed file: {file.name}")
            # else: unchanged, skip
            
        return new_files, new_codes, new_indices
    
    def add_embeddings(
        self,
        files: List[Path],
        codes: List[str],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add or update embeddings in the store.
        
        Args:
            files: List of file paths
            codes: List of code contents (for hashing)
            embeddings: Numpy array of shape (n_files, embedding_dim)
            
        Returns:
            Number of embeddings added/updated
        """
        self._initialize()
        
        if len(files) == 0:
            return 0
            
        ids = []
        metadatas = []
        embedding_list = []
        
        for i, (file, code) in enumerate(zip(files, codes, strict=True)):
            filepath_str = str(file)
            code_hash = compute_code_hash(code)
            
            # Use filepath as unique ID (ChromaDB requires string IDs)
            ids.append(filepath_str)
            metadatas.append({
                "filepath": filepath_str,
                "filename": file.name,
                "code_hash": code_hash,
            })
            embedding_list.append(embeddings[i].tolist())
        
        # Upsert (add or update)
        self._collection.upsert(
            ids=ids,
            embeddings=embedding_list,
            metadatas=metadatas,
        )
        
        logger.info(f"Stored {len(ids)} embeddings")
        return len(ids)
    
    def get_embeddings(self, filepaths: List[str]) -> List[Optional[np.ndarray]]:
        """
        Retrieve embeddings for specific files.
        
        Args:
            filepaths: List of file paths to retrieve
            
        Returns:
            List of embeddings (or None if not found), in same order as input
        """
        self._initialize()
        
        if not filepaths:
            return []
            
        # ChromaDB .get(ids=...) returns results in arbitrary order, 
        # so we need to reorder them manually.
        results = self._collection.get(ids=filepaths, include=["embeddings"])
        
        # Map ID -> Embedding
        id_to_emb = {}
        if results["ids"]:
            for id_, emb in zip(results["ids"], results["embeddings"]):
                if emb is not None:
                    id_to_emb[id_] = np.array(emb)
        
        # Reconstruct list in requested order
        ordered_embeddings = []
        for fp in filepaths:
            ordered_embeddings.append(id_to_emb.get(fp))
            
        return ordered_embeddings

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Retrieve all stored embeddings.
        
        Returns:
            Tuple of (file_paths, embeddings_array)
        """
        self._initialize()
        
        results = self._collection.get(include=["embeddings", "metadatas"])
        
        if not results["ids"]:
            return [], np.zeros((0, self.embedding_dim))
        
        filepaths = []
        embeddings = []
        
        for metadata, embedding in zip(results["metadatas"], results["embeddings"]):
            if metadata is not None and embedding is not None:
                filepaths.append(metadata.get("filepath", ""))
                embeddings.append(embedding)
        
        return filepaths, np.array(embeddings)
    
    def count(self) -> int:
        """Get the number of stored embeddings."""
        self._initialize()
        return self._collection.count()
    
    def clear(self):
        """Clear all stored embeddings."""
        self._initialize()
        # Delete collection and recreate
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"description": "CodeTurtle code embeddings cache"}
        )
        logger.info("Cleared vector store")


# Convenience function for default store location
def get_default_store(base_dir: Path = None) -> CodeVectorStore:
    """
    Get a CodeVectorStore at the default location.
    
    Args:
        base_dir: Base directory (defaults to data/vector_store)
        
    Returns:
        CodeVectorStore instance
    """
    if base_dir is None:
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from config.settings import DATA_DIR
        base_dir = DATA_DIR / "vector_store"
    
    return CodeVectorStore(persist_directory=base_dir)
