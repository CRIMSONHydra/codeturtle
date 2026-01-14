"""Tests for the AST parser module."""

import pytest
from src.preprocessor.ast_parser import (
    parse_to_ast,
    extract_functions,
    extract_classes,
    extract_imports,
    get_ast_summary,
)


class TestParseToAst:
    """Tests for AST parsing."""
    
    def test_parses_valid_code(self):
        code = "x = 1"
        tree = parse_to_ast(code)
        assert tree is not None
    
    def test_returns_none_for_invalid_code(self):
        code = "def broken("
        tree = parse_to_ast(code)
        assert tree is None


class TestExtractFunctions:
    """Tests for function extraction."""
    
    def test_extracts_function(self):
        code = "def hello(): pass"
        tree = parse_to_ast(code)
        funcs = extract_functions(tree)
        assert len(funcs) == 1
        assert funcs[0].name == "hello"
    
    def test_extracts_async_function(self):
        code = "async def fetch(): pass"
        tree = parse_to_ast(code)
        funcs = extract_functions(tree)
        assert len(funcs) == 1
        assert funcs[0].is_async is True
    
    def test_counts_arguments(self):
        code = "def add(a, b, c): pass"
        tree = parse_to_ast(code)
        funcs = extract_functions(tree)
        assert funcs[0].num_args == 3


class TestExtractClasses:
    """Tests for class extraction."""
    
    def test_extracts_class(self):
        code = "class Foo: pass"
        tree = parse_to_ast(code)
        classes = extract_classes(tree)
        assert len(classes) == 1
        assert classes[0].name == "Foo"
    
    def test_extracts_base_classes(self):
        code = "class Child(Parent): pass"
        tree = parse_to_ast(code)
        classes = extract_classes(tree)
        assert "Parent" in classes[0].bases


class TestExtractImports:
    """Tests for import extraction."""
    
    def test_extracts_import(self):
        code = "import os"
        tree = parse_to_ast(code)
        imports = extract_imports(tree)
        assert len(imports) >= 1
        assert any(i.module == "os" for i in imports)
    
    def test_extracts_from_import(self):
        code = "from pathlib import Path"
        tree = parse_to_ast(code)
        imports = extract_imports(tree)
        assert any(i.module == "pathlib" and "Path" in i.names for i in imports)


class TestGetAstSummary:
    """Tests for AST summary."""
    
    def test_counts_correctly(self):
        code = '''
import os

class Foo:
    def method(self):
        pass

def bar():
    pass
'''
        tree = parse_to_ast(code)
        summary = get_ast_summary(tree)
        
        assert summary["function_count"] >= 2  # method + bar
        assert summary["class_count"] == 1
        assert summary["import_count"] >= 1
