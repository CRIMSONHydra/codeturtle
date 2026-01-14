"""
Code Cleaner Module

Removes comments, docstrings, and normalizes whitespace
so ML models focus on logic, not documentation.
"""

import ast
import re
import tokenize
import io
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_comments(code: str) -> str:
    """
    Remove single-line comments (# ...) from Python code.
    
    Preserves strings that contain # characters.
    
    Args:
        code: Python source code
        
    Returns:
        Code with comments removed
    """
    try:
        # Use tokenizer to properly handle strings vs comments
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        
        # Filter out comments, keeping everything else
        filtered_tokens = []
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                continue
            filtered_tokens.append(tok)
        
        # Untokenize preserves whitespace
        result = tokenize.untokenize(filtered_tokens)
        return result
                
    except tokenize.TokenizeError:
        # Fallback: simple regex removal (less accurate)
        lines = code.split('\n')
        result = []
        for line in lines:
            # Remove inline comments (naive approach)
            in_string = False
            string_char = None
            final_line = []
            i = 0
            while i < len(line):
                char = line[i]
                if char in '"\'':
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                elif char == '#' and not in_string:
                    break
                final_line.append(char)
                i += 1
            result.append(''.join(final_line).rstrip())
        return '\n'.join(result)


def remove_docstrings(code: str) -> str:
    """
    Remove docstrings from Python code using AST.
    
    Removes:
    - Module-level docstrings
    - Function docstrings
    - Class docstrings
    - Method docstrings
    
    Args:
        code: Python source code
        
    Returns:
        Code with docstrings removed
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Can't parse, return original
        logger.warning("Could not parse code for docstring removal")
        return code
    
    # Walk the AST and find docstring locations
    docstring_lines = set()
    
    for node in ast.walk(tree):
        # Check if node can have a docstring
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and 
                isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                
                docstring_node = node.body[0]
                # Mark lines for removal
                for line_no in range(docstring_node.lineno, docstring_node.end_lineno + 1):
                    docstring_lines.add(line_no)
    
    # Remove docstring lines
    lines = code.split('\n')
    result = []
    for i, line in enumerate(lines, 1):
        if i not in docstring_lines:
            result.append(line)
        else:
            # Keep empty line for structure (optional)
            pass
    
    return '\n'.join(result)


def normalize_whitespace(code: str) -> str:
    """
    Normalize whitespace in Python code.
    
    - Removes trailing whitespace
    - Collapses multiple blank lines to max 2
    - Ensures single newline at end
    
    Args:
        code: Python source code
        
    Returns:
        Code with normalized whitespace
    """
    lines = code.split('\n')
    
    # Remove trailing whitespace
    lines = [line.rstrip() for line in lines]
    
    # Collapse multiple blank lines
    result = []
    blank_count = 0
    
    for line in lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    
    # Remove leading/trailing blank lines
    while result and result[0].strip() == '':
        result.pop(0)
    while result and result[-1].strip() == '':
        result.pop()
    
    # Ensure single newline at end
    return '\n'.join(result) + '\n' if result else ''


def clean_code(code: str, remove_docs: bool = True, remove_comments_flag: bool = True) -> str:
    """
    Full cleaning pipeline for Python code.
    
    Args:
        code: Python source code
        remove_docs: Whether to remove docstrings
        remove_comments_flag: Whether to remove comments
        
    Returns:
        Cleaned code ready for analysis
    """
    result = code
    
    # Remove docstrings FIRST (requires valid AST)
    # Comments can break AST parsing in some edge cases
    if remove_docs:
        result = remove_docstrings(result)
    
    if remove_comments_flag:
        result = remove_comments(result)
    
    result = normalize_whitespace(result)
    
    return result


def clean_file(filepath: Path, output_dir: Optional[Path] = None) -> Optional[str]:
    """
    Clean a Python file and optionally save the result.
    
    Args:
        filepath: Path to the Python file
        output_dir: If provided, save cleaned file here
        
    Returns:
        Cleaned code string, or None if failed
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return None
    
    cleaned = clean_code(code)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filepath.name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        logger.info(f"Saved cleaned code to {output_path}")
    
    return cleaned


if __name__ == "__main__":
    # Test code cleaning
    test_code = '''
"""This is a module docstring."""

# This is a comment
def hello(name):
    """Say hello to someone."""
    # Print greeting
    print(f"Hello, {name}!")  # inline comment
    
class Greeter:
    """A class for greeting."""
    
    def greet(self):
        """Greet method."""
        pass
'''
    
    print("Original:")
    print(test_code)
    print("\nCleaned:")
    print(clean_code(test_code))
