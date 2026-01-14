"""Tests for structural feature extraction."""

import pytest
from src.features.structural import (
    extract_structural_features,
    StructuralFeatures,
    calculate_cyclomatic_complexity,
)
from src.preprocessor.ast_parser import parse_to_ast


class TestStructuralFeatures:
    """Tests for structural feature extraction."""
    
    def test_extracts_features(self):
        code = '''
def hello():
    for i in range(10):
        print(i)
'''
        features = extract_structural_features(code)
        assert features is not None
        assert features.loop_count >= 1
        assert features.for_loop_count >= 1
    
    def test_detects_recursion(self):
        code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
        features = extract_structural_features(code)
        assert features.has_recursion is True
    
    def test_counts_try_except(self):
        code = '''
def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0
'''
        features = extract_structural_features(code)
        assert features.try_except_count >= 1
    
    def test_detects_bare_except(self):
        code = '''
def risky():
    try:
        do_something()
    except:
        pass
'''
        features = extract_structural_features(code)
        assert features.bare_except_count >= 1
    
    def test_counts_comprehensions(self):
        code = '''
squares = [x**2 for x in range(10)]
evens = {x for x in range(10) if x % 2 == 0}
'''
        features = extract_structural_features(code)
        assert features.comprehension_count >= 2
    
    def test_nesting_depth(self):
        code = '''
def deep():
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i > j:
                    print(k)
'''
        features = extract_structural_features(code)
        assert features.max_nesting_depth >= 4
    
    def test_feature_vector_length(self):
        code = "x = 1"
        features = extract_structural_features(code)
        vector = features.to_vector()
        names = StructuralFeatures.feature_names()
        assert len(vector) == len(names)


class TestCyclomaticComplexity:
    """Tests for cyclomatic complexity calculation."""
    
    def test_simple_function(self):
        code = "def foo(): pass"
        tree = parse_to_ast(code)
        cc = calculate_cyclomatic_complexity(tree)
        assert cc == 1  # Base complexity
    
    def test_with_if(self):
        code = '''
def foo(x):
    if x > 0:
        return True
    return False
'''
        tree = parse_to_ast(code)
        cc = calculate_cyclomatic_complexity(tree)
        assert cc >= 2  # 1 base + 1 if
    
    def test_with_loop(self):
        code = '''
def foo(items):
    for item in items:
        process(item)
'''
        tree = parse_to_ast(code)
        cc = calculate_cyclomatic_complexity(tree)
        assert cc >= 2  # 1 base + 1 for
