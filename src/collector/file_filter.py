"""
File Filtering for Python Code Analysis

Filters and selects Python files suitable for analysis,
excluding tests, configs, and other non-relevant files.
"""

import fnmatch
from pathlib import Path
from typing import List, Optional, Dict, Generator
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    EXCLUDED_PATTERNS,
    MIN_FILE_SIZE,
    MAX_FILE_SIZE,
    RAW_DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def matches_any_pattern(path: Path, patterns: List[str]) -> bool:
    """
    Check if a path matches any of the exclusion patterns.
    
    Args:
        path: File path to check
        patterns: List of glob patterns to match against
        
    Returns:
        True if path matches any pattern
    """
    path_str = str(path)
    
    for pattern in patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True
        # Also check just the filename
        if fnmatch.fnmatch(path.name, pattern):
            return True
    
    return False


def is_valid_python_file(filepath: Path) -> bool:
    """
    Check if a file is a valid Python file for analysis.
    
    Criteria:
    - Has .py extension
    - Not in excluded patterns (tests, configs, etc.)
    - Within size limits
    - Is a valid text file (not binary)
    
    Args:
        filepath: Path to the file
        
    Returns:
        True if file should be analyzed
    """
    # Check extension
    if filepath.suffix != ".py":
        return False
    
    # Check exclusion patterns
    if matches_any_pattern(filepath, EXCLUDED_PATTERNS):
        return False
    
    # Check if file exists and get size
    if not filepath.exists() or not filepath.is_file():
        return False
    
    file_size = filepath.stat().st_size
    
    # Check size limits
    if file_size < MIN_FILE_SIZE:
        return False
    if file_size > MAX_FILE_SIZE:
        return False
    
    # Try to read as text (skip binary files)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(100)  # Read first 100 chars to verify it's text
        return True
    except (UnicodeDecodeError, PermissionError):
        return False


def filter_python_files(
    directory: Path,
    recursive: bool = True
) -> Generator[Path, None, None]:
    """
    Filter and yield valid Python files from a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Yields:
        Paths to valid Python files
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return
    
    pattern = "**/*.py" if recursive else "*.py"
    
    for filepath in directory.glob(pattern):
        if is_valid_python_file(filepath):
            yield filepath


def get_file_info(filepath: Path) -> Dict:
    """
    Get detailed information about a Python file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary with file metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return {"error": "File not found", "path": str(filepath)}
    
    stat = filepath.stat()
    
    # Count lines
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    except Exception as e:
        return {"error": str(e), "path": str(filepath)}
    
    return {
        "path": str(filepath),
        "filename": filepath.name,
        "size_bytes": stat.st_size,
        "total_lines": len(lines),
        "code_lines": len(code_lines),
        "is_valid": is_valid_python_file(filepath),
    }


def collect_all_python_files(
    source_dir: Optional[Path] = None,
    limit: Optional[int] = None
) -> List[Path]:
    """
    Collect all valid Python files from the raw data directory.
    
    Args:
        source_dir: Directory to search (default: RAW_DATA_DIR)
        limit: Maximum number of files to return
        
    Returns:
        List of paths to Python files
    """
    if source_dir is None:
        source_dir = RAW_DATA_DIR
    
    source_dir = Path(source_dir)
    
    files = list(filter_python_files(source_dir))
    
    if limit:
        files = files[:limit]
    
    logger.info(f"Collected {len(files)} Python files from {source_dir}")
    return files


def get_collection_stats(files: List[Path]) -> Dict:
    """
    Get statistics about a collection of Python files.
    
    Args:
        files: List of file paths
        
    Returns:
        Dictionary with collection statistics
    """
    total_size = 0
    total_lines = 0
    total_code_lines = 0
    
    for f in files:
        info = get_file_info(f)
        if "error" not in info:
            total_size += info["size_bytes"]
            total_lines += info["total_lines"]
            total_code_lines += info["code_lines"]
    
    return {
        "file_count": len(files),
        "total_size_kb": round(total_size / 1024, 2),
        "total_lines": total_lines,
        "total_code_lines": total_code_lines,
        "avg_lines_per_file": round(total_lines / len(files), 1) if files else 0,
    }


if __name__ == "__main__":
    # Test file filtering
    print("Testing file filter...")
    files = collect_all_python_files(limit=10)
    print(f"Found {len(files)} files")
    for f in files[:5]:
        print(f"  {f}")
    
    if files:
        stats = get_collection_stats(files)
        print(f"\nStats: {stats}")
