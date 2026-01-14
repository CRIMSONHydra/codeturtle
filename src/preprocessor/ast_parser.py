"""
AST Parser Module

Parses Python code into Abstract Syntax Trees and extracts
structural information for feature extraction.
"""

import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function in the AST."""
    name: str
    lineno: int
    end_lineno: int
    args: List[str]
    num_args: int
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    has_return: bool = False


@dataclass
class ClassInfo:
    """Information about a class in the AST."""
    name: str
    lineno: int
    end_lineno: int
    bases: List[str]
    methods: List[FunctionInfo] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Information about imports."""
    module: str
    names: List[str]
    is_from_import: bool


def parse_to_ast(code: str) -> Optional[ast.AST]:
    """
    Parse Python code string into an AST.
    
    Args:
        code: Python source code
        
    Returns:
        AST tree or None if parsing fails
    """
    try:
        return ast.parse(code)
    except SyntaxError as e:
        logger.warning(f"Syntax error in code: {e}")
        return None


def parse_file(filepath: Path) -> Optional[ast.AST]:
    """
    Parse a Python file into an AST.
    
    Args:
        filepath: Path to Python file
        
    Returns:
        AST tree or None if parsing fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        return parse_to_ast(code)
    except Exception as e:
        logger.error(f"Failed to parse {filepath}: {e}")
        return None


def extract_functions(tree: ast.AST) -> List[FunctionInfo]:
    """
    Extract all function definitions from an AST.
    
    Args:
        tree: AST tree
        
    Returns:
        List of FunctionInfo objects
    """
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get argument names
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            for arg in node.args.kwonlyargs:
                args.append(arg.arg)
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")
            
            # Get decorator names
            decorators = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
            
            # Get function calls within this function
            calls = []
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Call):
                    if isinstance(subnode.func, ast.Name):
                        calls.append(subnode.func.id)
                    elif isinstance(subnode.func, ast.Attribute):
                        calls.append(subnode.func.attr)
            
            # Check for return statements
            has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
            
            func_info = FunctionInfo(
                name=node.name,
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                args=args,
                num_args=len(node.args.args) + len(node.args.kwonlyargs),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                decorators=decorators,
                calls=calls,
                has_return=has_return,
            )
            functions.append(func_info)
    
    return functions


def extract_classes(tree: ast.AST) -> List[ClassInfo]:
    """
    Extract all class definitions from an AST.
    
    Args:
        tree: AST tree
        
    Returns:
        List of ClassInfo objects
    """
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get base class names
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            
            # Get decorators
            decorators = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
            
            # Get methods
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Simplified method info
                    methods.append(FunctionInfo(
                        name=item.name,
                        lineno=item.lineno,
                        end_lineno=item.end_lineno or item.lineno,
                        args=[arg.arg for arg in item.args.args],
                        num_args=len(item.args.args),
                        is_async=isinstance(item, ast.AsyncFunctionDef),
                    ))
            
            class_info = ClassInfo(
                name=node.name,
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                bases=bases,
                methods=methods,
                decorators=decorators,
            )
            classes.append(class_info)
    
    return classes


def extract_imports(tree: ast.AST) -> List[ImportInfo]:
    """
    Extract all imports from an AST.
    
    Args:
        tree: AST tree
        
    Returns:
        List of ImportInfo objects
    """
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportInfo(
                    module=alias.name,
                    names=[alias.asname or alias.name],
                    is_from_import=False,
                ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names = [alias.name for alias in node.names]
            imports.append(ImportInfo(
                module=module,
                names=names,
                is_from_import=True,
            ))
    
    return imports


def get_node_counts(tree: ast.AST) -> Dict[str, int]:
    """
    Count occurrences of different AST node types.
    
    Args:
        tree: AST tree
        
    Returns:
        Dictionary mapping node type names to counts
    """
    counts: Dict[str, int] = {}
    
    for node in ast.walk(tree):
        node_type = type(node).__name__
        counts[node_type] = counts.get(node_type, 0) + 1
    
    return counts


def get_ast_summary(tree: ast.AST) -> Dict[str, Any]:
    """
    Get a comprehensive summary of an AST.
    
    Args:
        tree: AST tree
        
    Returns:
        Dictionary with various AST statistics
    """
    functions = extract_functions(tree)
    classes = extract_classes(tree)
    imports = extract_imports(tree)
    counts = get_node_counts(tree)
    
    return {
        "function_count": len(functions),
        "class_count": len(classes),
        "import_count": len(imports),
        "functions": [f.name for f in functions],
        "classes": [c.name for c in classes],
        "node_counts": counts,
        "total_nodes": sum(counts.values()),
        "has_async": any(f.is_async for f in functions),
    }


if __name__ == "__main__":
    # Test AST parsing
    test_code = '''
import os
from pathlib import Path

class Calculator:
    def add(self, a, b):
        return a + b
    
    async def async_compute(self, x):
        return x * 2

def main():
    calc = Calculator()
    print(calc.add(1, 2))

if __name__ == "__main__":
    main()
'''
    
    tree = parse_to_ast(test_code)
    if tree:
        print("Functions:", [f.name for f in extract_functions(tree)])
        print("Classes:", [c.name for c in extract_classes(tree)])
        print("Imports:", [i.module for i in extract_imports(tree)])
        print("\nSummary:", get_ast_summary(tree))
