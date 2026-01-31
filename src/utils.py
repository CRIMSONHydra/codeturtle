"""
Common utility functions for CodeTurtle.
"""

import logging
from typing import List, Generator, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def batch_generator(files: List[Path], batch_size: int = 32) -> Generator[Tuple[List[Path], List[str]], None, None]:
    """
    Yields batches of files to avoid loading all content into RAM.
    
    Args:
        files: List of file paths
        batch_size: Number of files per batch
        
    Yields:
        Tuple of (batch_file_paths, batch_codes)
    """
    for i in range(0, len(files), batch_size):
        batch_paths = files[i:i + batch_size]
        batch_codes = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                # Assuming simple file reading here.
                # In a real heavy-duty setup, this might be async.
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()
                batch_codes.append(code)
                valid_paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
                continue
                
        if batch_codes:
            yield valid_paths, batch_codes
