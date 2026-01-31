"""Tests for batch processing utilities."""

import pytest
from pathlib import Path
import tempfile
import os
from src.utils import batch_generator

class TestBatchGenerator:
    """Tests for the batch_generator function."""
    
    @pytest.fixture
    def sample_files(self):
        """Create sample python files in a temp dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = []
            # Create 10 files
            for i in range(10):
                p = tmp_path / f"file_{i}.py"
                p.write_text(f"print('file {i}')", encoding='utf-8')
                files.append(p)
            yield files

    def test_yields_correct_batches(self, sample_files):
        """Test basic batching logic."""
        batch_size = 3
        batches = list(batch_generator(sample_files, batch_size=batch_size))
        
        # 10 files with batch size 3 -> 4 batches (3, 3, 3, 1)
        assert len(batches) == 4
        
        # Check sizes
        assert len(batches[0][0]) == 3
        assert len(batches[1][0]) == 3
        assert len(batches[2][0]) == 3
        assert len(batches[3][0]) == 1
        
        # Check content
        assert batches[0][1][0] == "print('file 0')"
        assert batches[3][1][0] == "print('file 9')"

    def test_handles_read_errors_gracefully(self, sample_files):
        """Test skipping of unreadable files."""
        # Corrupt one file effectively (permissions or directory)
        # Easier: just make the path invalid in the list passed
        bad_path = Path("/nonexistent/path/foo.py")
        mixed_files = sample_files[:2] + [bad_path] + sample_files[2:]
        
        batches = list(batch_generator(mixed_files, batch_size=5))
        
        # The generator uses simple slicing [i:i+batch_size], so if a file is skipped,
        # that batch will just be smaller. It doesn't "fill up" from the next batch.
        # Input: 11 files. Batch size 5.
        # Batch 1 (0-5): 1 bad -> 4 valid
        # Batch 2 (5-10): 5 valid -> 5 valid
        # Batch 3 (10-11): 1 valid -> 1 valid
        
        assert len(batches) == 3
        assert sum(len(b[0]) for b in batches) == 10
