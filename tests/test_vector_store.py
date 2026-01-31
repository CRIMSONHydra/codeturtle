"""Tests for vector store caching functionality."""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from src.features.vector_store import CodeVectorStore, compute_code_hash


class TestComputeCodeHash:
    """Tests for the compute_code_hash function."""
    
    def test_same_code_same_hash(self):
        """Same code should produce the same hash."""
        code = "def foo(): pass"
        assert compute_code_hash(code) == compute_code_hash(code)
    
    def test_different_code_different_hash(self):
        """Different code should produce different hashes."""
        code1 = "def foo(): pass"
        code2 = "def bar(): pass"
        assert compute_code_hash(code1) != compute_code_hash(code2)
    
    def test_hash_is_hex_string(self):
        """Hash should be a valid hex string."""
        code = "print('hello')"
        hash_val = compute_code_hash(code)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 produces 64 hex chars
        int(hash_val, 16)  # Should not raise


class TestCodeVectorStore:
    """Tests for the CodeVectorStore class."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CodeVectorStore(persist_directory=Path(tmpdir))
            yield store
    
    @pytest.fixture
    def sample_data(self):
        """Create sample files and embeddings."""
        files = [
            Path("/fake/path/file1.py"),
            Path("/fake/path/file2.py"),
            Path("/fake/path/file3.py"),
        ]
        codes = [
            "def foo(): pass",
            "def bar(): return 1",
            "class Baz: pass",
        ]
        embeddings = np.random.randn(3, 768).astype(np.float32)
        return files, codes, embeddings
    
    def test_add_and_count(self, temp_store, sample_data):
        """Test adding embeddings and counting."""
        files, codes, embeddings = sample_data
        
        assert temp_store.count() == 0
        
        temp_store.add_embeddings(files, codes, embeddings)
        
        assert temp_store.count() == 3
    
    def test_get_cached_hashes(self, temp_store, sample_data):
        """Test retrieving cached hashes."""
        files, codes, embeddings = sample_data
        
        temp_store.add_embeddings(files, codes, embeddings)
        
        cached = temp_store.get_cached_hashes()
        
        assert len(cached) == 3
        assert str(files[0]) in cached
        assert cached[str(files[0])] == compute_code_hash(codes[0])
    
    def test_filter_new_files(self, temp_store, sample_data):
        """Test filtering identifies new files."""
        files, codes, embeddings = sample_data
        
        # Add first two files to cache
        temp_store.add_embeddings(files[:2], codes[:2], embeddings[:2])
        
        # Now filter all three - file3 should be new
        new_files, new_codes, indices = temp_store.filter_new_or_changed(files, codes)
        
        assert len(new_files) == 1
        assert new_files[0] == files[2]
        assert new_codes[0] == codes[2]
        assert indices == [2]
    
    def test_filter_changed_files(self, temp_store, sample_data):
        """Test filtering identifies changed files."""
        files, codes, embeddings = sample_data
        
        # Add all files
        temp_store.add_embeddings(files, codes, embeddings)
        
        # Modify one code
        modified_codes = codes.copy()
        modified_codes[1] = "def bar(): return 42"  # Changed!
        
        new_files, new_codes, indices = temp_store.filter_new_or_changed(files, modified_codes)
        
        assert len(new_files) == 1
        assert new_files[0] == files[1]
        assert indices == [1]
    
    def test_filter_unchanged_files(self, temp_store, sample_data):
        """Test filtering returns empty for unchanged files."""
        files, codes, embeddings = sample_data
        
        temp_store.add_embeddings(files, codes, embeddings)
        
        # Same files, same codes
        new_files, new_codes, indices = temp_store.filter_new_or_changed(files, codes)
        
        assert len(new_files) == 0
        assert len(new_codes) == 0
        assert len(indices) == 0
    
    def test_get_all_embeddings(self, temp_store, sample_data):
        """Test retrieving all embeddings."""
        files, codes, embeddings = sample_data
        
        temp_store.add_embeddings(files, codes, embeddings)
        
        retrieved_files, retrieved_embs = temp_store.get_all_embeddings()
        
        assert len(retrieved_files) == 3
        assert retrieved_embs.shape == embeddings.shape
    
    def test_clear(self, temp_store, sample_data):
        """Test clearing the store."""
        files, codes, embeddings = sample_data
        
        temp_store.add_embeddings(files, codes, embeddings)
        assert temp_store.count() == 3
        
        temp_store.clear()
        assert temp_store.count() == 0
    
    def test_upsert_updates_existing(self, temp_store, sample_data):
        """Test that add_embeddings updates existing entries."""
        files, codes, embeddings = sample_data
        
        temp_store.add_embeddings(files, codes, embeddings)
        
        # Modify code and re-add
        modified_codes = codes.copy()
        modified_codes[0] = "def foo(): return 'updated'"
        new_embeddings = np.random.randn(3, 768).astype(np.float32)
        
        temp_store.add_embeddings(files, modified_codes, new_embeddings)
        
        # Should still be 3 (upsert, not duplicate)
        assert temp_store.count() == 3
        
        # Hash should be updated
        cached = temp_store.get_cached_hashes()
        assert cached[str(files[0])] == compute_code_hash(modified_codes[0])
