"""Core classes for Hierarchical Tucker decomposition."""

import numpy as np


class TuckerCore:
    """Object for tucker cores in hierarchical Tucker decomposition.
    
    This class represents internal nodes in the hierarchical Tucker tree structure.
    """
    
    def __init__(self, core=None, parent=None, dims=None, idx=None):
        """Initialize a TuckerCore object.
        
        Args:
            core: Core tensor
            parent: Parent node
            dims: Dimensions
            idx: Index in the tree
        """
        self.parent = parent
        self.core = core
        self.left = None
        self.right = None
        self.dims = dims
        self.core_idx = idx
        self.ranks = []
        self.children = []

        if parent is None:
            self._isroot = True
        else:
            self._isroot = False

        self._isexpanded = False

    def get_ranks(self):
        """Get ranks from the core tensor shape.
        
        Updates the ranks attribute based on the shape of the core tensor.
        """
        if self.core is not None:
            self.ranks = list(self.core.shape)
    
    def contract_children(self):
        """Contract children nodes with the current core.
        
        This function is for binary tree structure.
        """
        # Need to write another contraction code for n-ary splits
        _ldims = len(self.left.dims) + 1
        _rdims = len(self.right.dims) + 1
        
        left_str = ''.join([chr(idx) for idx in range(97, 97+_ldims)])
        right_str = ''.join([chr(idx) for idx in range(97+_ldims, 97+_ldims+_rdims)])
        
        if self._isroot:
            core_str = left_str[-1] + right_str[-1]
        else:    
            core_str = left_str[-1] + right_str[-1] + chr(97+_ldims+_rdims)

        result_str = core_str.replace(left_str[-1], left_str[:-1])
        result_str = result_str.replace(right_str[-1], right_str[:-1])
        
        self.core = np.einsum(
            ','.join([left_str, right_str, core_str]) + '->' + result_str,
            self.left.core, self.right.core, self.core, optimize=True
        )

    def contract_children_dimension_tree(self):
        """Contract children nodes in a dimension tree structure.
        
        This supports an n-ary tree structure.
        """
        dimensions = []
        matrices = []
        
        for chld in self.children:
            if isinstance(chld, TuckerLeaf):
                dimensions.append(2)
            else:
                dimensions.append(len(chld.dims) + 1)
            matrices.append(chld.core)
            
        matrices.append(self.core)
        strings = []
        core_string = ""
        last_char = 97
        
        for dims in dimensions:
            strings.append(''.join([chr(idx) for idx in range(last_char, last_char+dims)]))
            last_char += dims
            core_string += strings[-1][-1]
            
        core_string += chr(last_char)
        result_string = ""
        
        for stri in strings:
            result_string += stri[:-1]
            
        result_string += core_string[-1]
        
        if self.parent is not None:
            pass
        else:  # We are contracting the root node
            # We need to adjust the einstein summation strings for root node
            result_string, core_string = result_string[:-1], core_string[:-1]
            
        self.core = eval(
            "np.einsum(" +
            "'" +
            ",".join([
                ','.join(strings+[core_string]) + '->' + result_string + "'",
                ",".join([f"matrices[{idx}]" for idx in range(len(matrices))]),
                'optimize=True,order="F"']
            ) +
            ")"
        )

    @property
    def shape(self):
        """Get the shape of the core tensor."""
        return self.core.shape


class TuckerLeaf:
    """Object for tucker leaf nodes in hierarchical Tucker decomposition.
    
    This class represents leaf nodes in the hierarchical Tucker tree structure.
    """
    
    def __init__(self, matrix=None, parent=None, dims=None, idx=None):
        """Initialize a TuckerLeaf object.
        
        Args:
            matrix: Factor matrix
            parent: Parent node
            dims: Dimensions
            idx: Leaf index
        """
        self.parent = parent
        self.core = matrix  # Refactoring this as core for consistency with TuckerCore
        self.dims = dims
        self.leaf_idx = idx
        
        if matrix is not None: 
            self.rank = matrix.shape[1]
            
    @property
    def shape(self):
        """Get the shape of the factor matrix."""
        return self.core.shape
    
    def get_ranks(self):
        """Get ranks from the core tensor shape.
        
        Updates the rank attribute based on the shape of the factor matrix.
        """
        if self.core is not None:
            self.rank = self.core.shape[-1]
