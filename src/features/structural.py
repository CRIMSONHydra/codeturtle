"""
Structural Feature Extraction

Extracts AST-based features that capture code structure,
complexity, and potential risk indicators.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessor.ast_parser import parse_to_ast, extract_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StructuralFeatures:
    """Container for all structural features extracted from code."""
    
    # Basic counts
    loop_count: int = 0
    for_loop_count: int = 0
    while_loop_count: int = 0
    if_count: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    
    # Complexity metrics
    max_nesting_depth: int = 0
    avg_function_length: float = 0.0
    max_function_length: int = 0
    cyclomatic_complexity: int = 0
    
    # Pattern indicators
    has_recursion: bool = False
    recursion_count: int = 0
    try_except_count: int = 0
    bare_except_count: int = 0
    global_var_count: int = 0
    assertion_count: int = 0
    
    # Advanced patterns
    lambda_count: int = 0
    comprehension_count: int = 0
    generator_count: int = 0
    decorator_count: int = 0
    
    # Code characteristics
    total_lines: int = 0
    code_lines: int = 0
    has_main_guard: bool = False
    has_type_hints: bool = False
    
    def to_vector(self) -> List[float]:
        """Convert features to a numerical vector for ML."""
        return [
            float(self.loop_count),
            float(self.for_loop_count),
            float(self.while_loop_count),
            float(self.if_count),
            float(self.function_count),
            float(self.class_count),
            float(self.import_count),
            float(self.max_nesting_depth),
            float(self.avg_function_length),
            float(self.max_function_length),
            float(self.cyclomatic_complexity),
            float(self.has_recursion),
            float(self.recursion_count),
            float(self.try_except_count),
            float(self.bare_except_count),
            float(self.global_var_count),
            float(self.assertion_count),
            float(self.lambda_count),
            float(self.comprehension_count),
            float(self.generator_count),
            float(self.decorator_count),
            float(self.total_lines),
            float(self.code_lines),
            float(self.has_main_guard),
            float(self.has_type_hints),
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get names of all features in vector order."""
        return [
            "loop_count", "for_loop_count", "while_loop_count",
            "if_count", "function_count", "class_count", "import_count",
            "max_nesting_depth", "avg_function_length", "max_function_length",
            "cyclomatic_complexity", "has_recursion", "recursion_count",
            "try_except_count", "bare_except_count", "global_var_count",
            "assertion_count", "lambda_count", "comprehension_count",
            "generator_count", "decorator_count", "total_lines", "code_lines",
            "has_main_guard", "has_type_hints",
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class NestingDepthVisitor(ast.NodeVisitor):
    """Visitor to calculate maximum nesting depth."""
    
    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0
    
    def _visit_nesting(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_For(self, node):
        self._visit_nesting(node)
    
    def visit_While(self, node):
        self._visit_nesting(node)
    
    def visit_If(self, node):
        self._visit_nesting(node)
    
    def visit_With(self, node):
        self._visit_nesting(node)
    
    def visit_Try(self, node):
        self._visit_nesting(node)
    
    def visit_FunctionDef(self, node):
        self._visit_nesting(node)
    
    def visit_AsyncFunctionDef(self, node):
        self._visit_nesting(node)


class RecursionDetector(ast.NodeVisitor):
    """Detect recursive function calls."""
    
    def __init__(self):
        self.function_names: Set[str] = set()
        self.recursive_calls: Set[str] = set()
        self.current_function: Optional[str] = None
    
    def visit_FunctionDef(self, node):
        self.function_names.add(node.name)
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func
    
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
    
    def visit_Call(self, node):
        if self.current_function:
            if isinstance(node.func, ast.Name):
                if node.func.id == self.current_function:
                    self.recursive_calls.add(self.current_function)
        self.generic_visit(node)


def calculate_cyclomatic_complexity(tree: ast.AST) -> int:
    """
    Calculate McCabe cyclomatic complexity.
    
    CC = E - N + 2P where:
    - E = edges (decision points)
    - N = nodes  
    - P = connected components (usually 1)
    
    Simplified: Count decision points + 1
    """
    complexity = 1  # Base complexity
    
    for node in ast.walk(tree):
        # Each decision point adds 1
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # Each boolean operator adds paths
            complexity += len(node.values) - 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Comprehensions with conditions
            for generator in node.generators:
                complexity += len(generator.ifs)
    
    return complexity


def extract_structural_features(code: str) -> Optional[StructuralFeatures]:
    """
    Extract all structural features from Python code.
    
    Args:
        code: Python source code string
        
    Returns:
        StructuralFeatures object or None if parsing fails
    """
    tree = parse_to_ast(code)
    if tree is None:
        return None
    
    features = StructuralFeatures()
    
    # Count lines
    lines = code.split('\n')
    features.total_lines = len(lines)
    features.code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith('#'))
    
    # Basic node counts
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            features.for_loop_count += 1
            features.loop_count += 1
        elif isinstance(node, ast.While):
            features.while_loop_count += 1
            features.loop_count += 1
        elif isinstance(node, ast.If):
            features.if_count += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            features.function_count += 1
            # Check for type hints
            if node.returns or any(arg.annotation for arg in node.args.args):
                features.has_type_hints = True
        elif isinstance(node, ast.ClassDef):
            features.class_count += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            features.import_count += 1
        elif isinstance(node, ast.Try):
            features.try_except_count += 1
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                features.bare_except_count += 1
        elif isinstance(node, ast.Global):
            features.global_var_count += len(node.names)
        elif isinstance(node, ast.Assert):
            features.assertion_count += 1
        elif isinstance(node, ast.Lambda):
            features.lambda_count += 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            features.comprehension_count += 1
        elif isinstance(node, ast.GeneratorExp):
            features.generator_count += 1
    
    # Count decorators
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            features.decorator_count += len(node.decorator_list)
    
    # Calculate nesting depth
    depth_visitor = NestingDepthVisitor()
    depth_visitor.visit(tree)
    features.max_nesting_depth = depth_visitor.max_depth
    
    # Detect recursion
    recursion_detector = RecursionDetector()
    recursion_detector.visit(tree)
    features.has_recursion = len(recursion_detector.recursive_calls) > 0
    features.recursion_count = len(recursion_detector.recursive_calls)
    
    # Calculate cyclomatic complexity
    features.cyclomatic_complexity = calculate_cyclomatic_complexity(tree)
    
    # Function length statistics
    functions = extract_functions(tree)
    if functions:
        lengths = [f.end_lineno - f.lineno + 1 for f in functions]
        features.avg_function_length = sum(lengths) / len(lengths)
        features.max_function_length = max(lengths)
    
    # Check for main guard
    features.has_main_guard = 'if __name__' in code
    
    return features


def extract_features_from_file(filepath: Path) -> Optional[StructuralFeatures]:
    """
    Extract structural features from a Python file.
    
    Args:
        filepath: Path to Python file
        
    Returns:
        StructuralFeatures or None if extraction fails
    """
    filepath = Path(filepath)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        return extract_structural_features(code)
    except Exception as e:
        logger.error(f"Failed to extract features from {filepath}: {e}")
        return None


if __name__ == "__main__":
    # Test feature extraction
    test_code = '''
def fibonacci(n):
    """Calculate fibonacci recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def process_data(items):
    results = []
    for item in items:
        try:
            if item > 0:
                for i in range(item):
                    results.append(i ** 2)
        except:
            pass
    return results

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add(self, x):
        self.data.append(x)

squares = [x**2 for x in range(10) if x % 2 == 0]

if __name__ == "__main__":
    print(fibonacci(10))
'''
    
    features = extract_structural_features(test_code)
    if features:
        print("Extracted Features:")
        for name, value in features.to_dict().items():
            print(f"  {name}: {value}")
        print(f"\nFeature Vector ({len(features.to_vector())} dims): {features.to_vector()[:10]}...")
