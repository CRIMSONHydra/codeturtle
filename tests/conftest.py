import pytest
import sys
# Pre-import torch to avoid double-loading issues with pytest assertion rewriting
# and torch internal registry (specifically _inductor_test)
try:
    import torch
except ImportError:
    pass
