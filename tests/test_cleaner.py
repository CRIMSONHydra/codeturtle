"""Tests for the code cleaner module."""

import pytest
from src.preprocessor.cleaner import (
    remove_comments,
    remove_docstrings,
    normalize_whitespace,
    clean_code,
)


class TestRemoveComments:
    """Tests for comment removal."""
    
    def test_removes_inline_comment(self):
        code = "x = 1  # this is a comment"
        result = remove_comments(code)
        assert "#" not in result or "this is a comment" not in result
    
    def test_removes_full_line_comment(self):
        code = "# full line comment\nx = 1"
        result = remove_comments(code)
        assert "full line" not in result
    
    def test_preserves_hash_in_string(self):
        code = 'x = "hello # world"'
        result = remove_comments(code)
        assert "hello" in result


class TestRemoveDocstrings:
    """Tests for docstring removal."""
    
    def test_removes_function_docstring(self):
        code = '''
def hello():
    """This is a docstring."""
    pass
'''
        result = remove_docstrings(code)
        assert "This is a docstring" not in result
    
    def test_removes_class_docstring(self):
        code = '''
class Foo:
    """Class docstring."""
    pass
'''
        result = remove_docstrings(code)
        assert "Class docstring" not in result
    
    def test_preserves_code(self):
        code = '''
def add(a, b):
    """Add two numbers."""
    return a + b
'''
        result = remove_docstrings(code)
        assert "return a + b" in result


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""
    
    def test_removes_trailing_whitespace(self):
        code = "x = 1   \ny = 2   "
        result = normalize_whitespace(code)
        assert not any(line.endswith(" ") for line in result.split("\n"))
    
    def test_collapses_blank_lines(self):
        code = "x = 1\n\n\n\n\ny = 2"
        result = normalize_whitespace(code)
        # Should have at most 2 consecutive blank lines
        assert "\n\n\n\n" not in result


class TestCleanCode:
    """Tests for full cleaning pipeline."""
    
    def test_full_cleaning(self):
        code = '''
"""Module docstring."""

# A comment
def foo():
    """Function docstring."""
    x = 1  # inline
    return x
'''
        result = clean_code(code)
        
        # Should remove docstrings and comments
        assert "Module docstring" not in result
        assert "Function docstring" not in result
        assert "A comment" not in result
        
        # Should preserve code
        assert "def foo" in result
        assert "return x" in result
