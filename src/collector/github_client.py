"""
GitHub Repository Cloning and Data Collection

Handles cloning repositories from GitHub for analysis.
Supports both authenticated (higher rate limits) and
unauthenticated access.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List
import logging

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    GITHUB_TOKEN,
    TARGET_REPOS,
    RAW_DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_repo_url(repo: str, use_token: bool = True) -> str:
    """
    Generate the clone URL for a GitHub repository.
    
    Args:
        repo: Repository in format "owner/name"
        use_token: Whether to embed token in URL for authentication
        
    Returns:
        Clone URL string
    """
    if use_token and GITHUB_TOKEN:
        return f"https://{GITHUB_TOKEN}@github.com/{repo}.git"
    return f"https://github.com/{repo}.git"


def clone_repository(
    repo: str,
    target_dir: Optional[Path] = None,
    depth: int = 1,
    force: bool = False
) -> Path:
    """
    Clone a GitHub repository to the local filesystem.
    
    Args:
        repo: Repository in format "owner/name"
        target_dir: Where to clone (default: RAW_DATA_DIR/repo_name)
        depth: Git clone depth (1 = shallow clone, saves space)
        force: If True, delete existing directory and re-clone
        
    Returns:
        Path to cloned repository
        
    Raises:
        RuntimeError: If clone fails
    """
    # Parse repo name for directory
    owner, name = repo.split("/")
    
    if target_dir is None:
        target_dir = RAW_DATA_DIR / f"{owner}__{name}"
    else:
        target_dir = Path(target_dir)
    
    # Handle existing directory
    if target_dir.exists():
        if force:
            logger.info(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
        else:
            logger.info(f"Repository already exists: {target_dir}")
            return target_dir
    
    # Clone the repository
    url = get_repo_url(repo)
    cmd = ["git", "clone", "--depth", str(depth), url, str(target_dir)]
    
    logger.info(f"Cloning {repo} to {target_dir}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")
            
        logger.info(f"Successfully cloned {repo}")
        return target_dir
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Clone timeout for {repo}")


def clone_all_target_repos(
    repos: Optional[List[str]] = None,
    max_repos: Optional[int] = None,
    force: bool = False
) -> List[Path]:
    """
    Clone all target repositories for analysis.
    
    Args:
        repos: List of repos to clone (default: TARGET_REPOS from settings)
        max_repos: Maximum number of repos to clone
        force: If True, re-clone existing repos
        
    Returns:
        List of paths to cloned repositories
    """
    if repos is None:
        repos = TARGET_REPOS
    
    if max_repos:
        repos = repos[:max_repos]
    
    cloned_paths = []
    
    for repo in repos:
        try:
            path = clone_repository(repo, force=force)
            cloned_paths.append(path)
        except Exception as e:
            logger.error(f"Failed to clone {repo}: {e}")
            continue
    
    logger.info(f"Successfully cloned {len(cloned_paths)}/{len(repos)} repositories")
    return cloned_paths


def get_repo_info(repo_path: Path) -> dict:
    """
    Get basic information about a cloned repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Dictionary with repo information
    """
    info = {
        "path": str(repo_path),
        "name": repo_path.name,
        "exists": repo_path.exists(),
    }
    
    if repo_path.exists():
        # Count Python files
        py_files = list(repo_path.rglob("*.py"))
        info["python_file_count"] = len(py_files)
        
        # Get total size
        total_size = sum(f.stat().st_size for f in py_files if f.is_file())
        info["total_python_size_kb"] = round(total_size / 1024, 2)
    
    return info


if __name__ == "__main__":
    # Test with a small repo
    print("Testing GitHub client...")
    paths = clone_all_target_repos(max_repos=1)
    for p in paths:
        print(get_repo_info(p))
