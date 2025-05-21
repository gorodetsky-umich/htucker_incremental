"""Htucker.

Htucker is a package for hierarchical tucker decomposition.
"""

__version__ = "0.1.0"

# Import core classes
from htucker.core import TuckerCore, TuckerLeaf
from htucker.decomposition import hosvd, hosvd_only_for_dimensions, truncated_svd
from htucker.ht import HTucker
from htucker.tree import Node, Tree, createDimensionTree
from htucker.utils import (
    NotFoundError,
    convert_to_base2,
    create_permutations,
    mode_n_product,
    mode_n_unfolding,
    split_dimensions,
)

__all__ = [
    "HTucker",
    "TuckerCore",
    "TuckerLeaf",
    "hosvd",
    "truncated_svd",
    "hosvd_only_for_dimensions",
    "create_permutations",
    "split_dimensions",
    "mode_n_unfolding",
    "mode_n_product",
    "createDimensionTree",
    "Tree",
    "Node",
    "NotFoundError",
    "convert_to_base2",
]
