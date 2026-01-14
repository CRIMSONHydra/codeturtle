"""Preprocessor package for code cleaning and AST parsing."""
from .cleaner import clean_code, remove_comments, remove_docstrings
from .ast_parser import parse_to_ast, extract_functions, extract_classes
