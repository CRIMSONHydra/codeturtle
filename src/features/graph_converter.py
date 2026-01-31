"""
AST to Graph Converter for GNN Processing.

Converts Python source code into a graph structure suitable for PyTorch Geometric.
Nodes represent AST elements, and edges represent hierarchical/control-flow relationships.
"""

import ast
import logging
from typing import List, Dict, Tuple, Optional, Set
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

logger = logging.getLogger(__name__)

# Standard Python AST node types to track
# We assign a unique ID to each for one-hot encoding
AST_NODE_TYPES = [
    'Module', 'FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'Return', 'Delete',
    'Assign', 'AugAssign', 'AnnAssign', 'For', 'AsyncFor', 'While', 'If', 'With',
    'AsyncWith', 'Match', 'Raise', 'Try', 'Assert', 'Import', 'ImportFrom',
    'Global', 'Nonlocal', 'Expr', 'Pass', 'Break', 'Continue', 'BoolOp',
    'NamedExpr', 'BinOp', 'UnaryOp', 'Lambda', 'IfExp', 'Dict', 'Set',
    'ListComp', 'SetComp', 'DictComp', 'GeneratorExp', 'Await', 'Yield',
    'YieldFrom', 'Compare', 'Call', 'FormattedValue', 'JoinedStr', 'Constant',
    'Attribute', 'Subscript', 'Starred', 'Name', 'List', 'Tuple', 'Slice',
    'Load', 'Store', 'Del', 'arg', 'arguments', 'alias', 'ExceptHandler'
]

NODE_TYPE_MAP = {name: i for i, name in enumerate(AST_NODE_TYPES)}
NUM_NODE_TYPES = len(AST_NODE_TYPES) + 1  # +1 for "Unknown"


class ASTGraphConverter:
    """Converts Python code to PyTorch Geometric graphs."""

    def __init__(self, use_gpu: bool = False):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    def code_to_graph(self, code: str) -> Optional[Data]:
        """
        Convert source code string to graph data object.
        
        Args:
            code: Python source code
            
        Returns:
            torch_geometric.data.Data object or None if parsing fails
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        except Exception as e:
            logger.warning(f"AST parse failed: {e}")
            return None

        # Tracking nodes and edges
        node_features = []  # List of integer IDs
        edge_index = [[], []]  # [source_indices, target_indices]
        
        # Helper to recursively traverse
        # Returns index of the processed node
        node_count = 0
        
        def process_node(node: ast.AST) -> int:
            nonlocal node_count
            current_idx = node_count
            node_count += 1
            
            # 1. Feature Extraction (Node Type)
            type_name = type(node).__name__
            type_id = NODE_TYPE_MAP.get(type_name, NUM_NODE_TYPES - 1)
            node_features.append(type_id)
            
            # 2. Traverse Children
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            child_idx = process_node(item)
                            # Add Edge: Parent -> Child
                            edge_index[0].append(current_idx)
                            edge_index[1].append(child_idx)
                            # Add Edge: Child -> Parent (Undirected/Bidirectional flow)
                            edge_index[0].append(child_idx)
                            edge_index[1].append(current_idx)
                elif isinstance(value, ast.AST):
                    child_idx = process_node(value)
                    edge_index[0].append(current_idx)
                    edge_index[1].append(child_idx)
                    edge_index[0].append(child_idx)
                    edge_index[1].append(current_idx)
                    
            return current_idx

        # Build graph
        try:
            process_node(tree)
        except RecursionError:
            logger.warning("Recursion depth exceeded during AST traversal")
            return None

        if node_count == 0:
            return None
        
        # Size limits to prevent CUDA memory issues
        MAX_NODES = 5000
        MAX_EDGES = 50000
        
        if node_count > MAX_NODES:
            logger.debug(f"Graph too large: {node_count} nodes (max {MAX_NODES})")
            return None
        
        if len(edge_index[0]) > MAX_EDGES:
            logger.debug(f"Too many edges: {len(edge_index[0])} (max {MAX_EDGES})")
            return None

        # Convert to Tensors
        # x: Node features [num_nodes, num_features] -> We use One-Hot encoding conceptually,
        # but for embedding layers we typically just pass the integer indices if using an Embedding layer.
        # Alternatively, we can output one-hot directly.
        # Let's use One-Hot for GCN compatibility locally.
        
        x_indices = torch.tensor(node_features, dtype=torch.long)
        x = torch.nn.functional.one_hot(x_indices, num_classes=NUM_NODE_TYPES).float()
        
        # Handle empty edge case
        if len(edge_index[0]) == 0:
            # Add self-loop for single node
            edge_index_tensor = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        # Validate edge indices are within bounds
        if edge_index_tensor.numel() > 0:
            max_idx = edge_index_tensor.max().item()
            if max_idx >= node_count:
                logger.warning(f"Invalid edge index: {max_idx} >= {node_count}")
                return None
        
        data = Data(x=x, edge_index=edge_index_tensor)
        
        # Validate
        if data.validate():
            return data
        else:
            return None

    def visualize(self, data: Data):
        """Vizualize graph using networkx (for debug)."""
        g = to_networkx(data, to_undirected=True)
        nx.draw(g)
