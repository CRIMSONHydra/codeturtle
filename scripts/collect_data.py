#!/usr/bin/env python3
"""
Data Collection Script

Clones GitHub repositories and collects Python files for analysis.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collector import clone_all_target_repos, clone_repository
from src.collector.file_filter import collect_all_python_files, get_collection_stats
from config.settings import TARGET_REPOS, RAW_DATA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Collect Python code from GitHub repositories"
    )
    parser.add_argument(
        '--repos',
        nargs='+',
        default=None,
        help='Specific repos to clone (e.g., TheAlgorithms/Python)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of repos to clone'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-clone existing repositories'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Just show stats of existing data'
    )
    
    args = parser.parse_args()
    
    if args.stats:
        print("\n📊 Collection Statistics")
        print("=" * 50)
        files = collect_all_python_files()
        stats = get_collection_stats(files)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    print("\n🐢 CodeTurtle Data Collection")
    print("=" * 50)
    
    repos = args.repos or TARGET_REPOS
    if args.limit:
        repos = repos[:args.limit]
    
    print(f"Target repositories: {len(repos)}")
    for r in repos:
        print(f"  - {r}")
    
    print("\n📥 Cloning repositories...")
    paths = clone_all_target_repos(repos=repos, force=args.force)
    
    print(f"\n✅ Cloned {len(paths)} repositories to {RAW_DATA_DIR}")
    
    # Collect stats
    print("\n📊 Collecting file statistics...")
    files = collect_all_python_files()
    stats = get_collection_stats(files)
    
    print(f"\nCollection Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Data collection complete!")


if __name__ == "__main__":
    main()
