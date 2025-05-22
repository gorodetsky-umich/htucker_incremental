"""Hierarchical Tucker tensor decomposition implementation.

This module implements the Hierarchical Tucker tensor format and decomposition methods as described
in the literature, particularly by Grasedyck 2010, Oseledets 2009, and Kressner 2014. The format
provides an efficient representation of high-dimensional tensors through a hierarchical structure.

The implementation includes:
- Hierarchical Tucker tensor representation
- Various compression methods (leaf-to-root, root-to-leaf approaches)
- Batch processing capabilities for multiple tensors
- Tensor operations (addition, multiplication, contraction)
- Incremental update procedures
- Utility functions for saving/loading the decomposition

References:
    - Grasedyck 2010: https://doi.org/10.1137/090764189
    - Oseledets 2009: https://doi.org/10.1137/090748330
    - Kressner 2014: https://sma.epfl.ch/~anchpcommon/publications/htucker_manual.pdf
"""

import os
import pickle as pckl
from warnings import warn

import numpy as np

import htucker as ht
from htucker.core import TuckerCore, TuckerLeaf
from htucker.decomposition import hosvd, hosvd_only_for_dimensions, truncated_svd
from htucker.tree import createDimensionTree
from htucker.utils import NotFoundError, mode_n_product, split_dimensions


class HTucker:
    """Hierarchical Tucker tensor decomposition.

    This class implements the hierarchical Tucker (HT) decomposition with various
    compression and reconstruction methods for high-dimensional tensors.

    The HT format represents tensors through a hierarchical binary tree structure
    where each node contains a transfer tensor that connects its children. Leaf nodes
    represent the original tensor's dimensions, and the root node represents the
    compressed representation of the entire tensor.

    The class supports:
    - Different compression approaches (leaf-to-root and root-to-leaf)
    - Batch processing for multiple tensors
    - Various tensor operations (addition, multiplication, contraction)
    - Error-controlled truncation
    - Incremental updates to existing decompositions
    - Serialization and deserialization for persistence

    Attributes:
        _leaf_count: Number of leaf nodes in the decomposition
        leaves: List of leaf nodes, each representing a tensor dimension
        transfer_nodes: List of intermediate (transfer) nodes
        root: Root node of the decomposition
        _iscompressed: Flag indicating if tensor is compressed
        _dimension_tree: Tree structure defining the dimension hierarchy
        batch_dimension: Index of batch dimension (for batch processing)
        rtol: Relative tolerance for compression error
    """

    def __init__(self):
        """Initialize the HTucker object.

        The initialization creates necessary placeholder variables that will be
        populated later during compression. No actual tensor decomposition happens
        during initialization - this happens when calling one of the compression methods.

        After initialization, call methods like `initialize()` and then one of the
        compression methods (`compress_leaf2root()`, `compress_root2leaf()`, etc.)
        to perform the actual decomposition.
        """
        self._leaf_count = None  # Number of leaf nodes in the decomposition
        self.leaves = None  # List of leaf nodes (TuckerLeaf objects)
        self.transfer_nodes = None  # List of transfer nodes (TuckerCore objects)
        self.root = None  # Root node of the decomposition
        self._iscompressed = False  # Flag indicating compression status
        self._dimension_tree = None  # Tree defining dimension hierarchy
        self.batch_dimension = None  # Index of batch dimension (for batch processing)
        self.rtol = None  # Relative tolerance for compression error

    def initialize(self, tensor, dimension_tree=None, batch=False, batch_dimension=None):
        """Initialize the HTucker object with a tensor.

        This method sets up the HTucker object with the provided tensor's information
        but does not perform the actual decomposition. After calling this method,
        you should call one of the compression methods to perform the decomposition.

        Args:
            tensor (ndarray): Input tensor to be decomposed.
            dimension_tree: Dimension tree defining the hierarchical structure for decomposition.
                If None, a dimension tree will be created automatically during compression.
            batch (bool): Whether the tensor has a batch dimension (for processing multiple tensors).
            batch_dimension (int, optional): Specifies which dimension is the batch dimension.
                If None and batch=True, defaults to the last dimension.

        Note:
            The batch dimension is handled specially - it's excluded from the dimension tree
            and decomposition processes but tracked so that batch operations can be performed.
        """
        self.original_shape = list(tensor.shape)
        if batch:
            self._leaf_count = len(self.original_shape) - 1
            if batch_dimension is None:
                batch_dimension = len(self.original_shape) - 1
            self.batch_dimension = batch_dimension
            self.original_shape = (
                self.original_shape[:batch_dimension] + self.original_shape[batch_dimension + 1 :]
            )
            self.batch_count = tensor.shape[batch_dimension]
        else:
            self.batch_count = 1
            self._leaf_count = len(self.original_shape)

        self._dimension_tree = dimension_tree
        self.leaves = [None] * self._leaf_count
        self.transfer_nodes = [None] * (self._leaf_count - 2)  # Root node not included here
        self.root = None
        self.nodes2Expand = []
        self._iscompressed = False
        self.allowed_error = 0

    def compress_root2leaf(self, tensor=None, isroot=False):
        """Compress a tensor using the root-to-leaf approach.

        This method performs hierarchical Tucker decomposition using a top-down approach,
        starting from the root of the dimension tree and working toward the leaves.
        The process uses recursive Higher Order Singular Value Decomposition (HOSVD).

        The root-to-leaf approach first decomposes the full tensor into a core tensor
        and factor matrices, then recursively applies SVD to these components moving
        down the tree structure.

        Args:
            tensor (ndarray): Input tensor to be decomposed.
            isroot (bool): Whether this is the root compression (primarily for internal use).

        Returns:
            None

        Raises:
            NotFoundError: If no tensor is provided.
            AssertionError: If the tensor is already compressed.

        Note:
            The results of compression are stored directly in the HTucker object's attributes
            (root, transfer_nodes, leaves) and can be accessed after this method completes.
            The _iscompressed flag is set to True upon successful completion.
        """
        assert self._iscompressed is False
        _leaf_counter = 0
        _node_counter = 0

        # This if check looks unnecessary
        # if self.root is None:
        #     isroot = True

        if tensor is None:
            raise NotFoundError("No tensor is given. Please check if you provided correct input(s)")

        dims = list(tensor.shape)

        # Initial split for the tensor
        left, right = split_dimensions(dims)

        self.root = TuckerCore(dims=dims)
        self.root.left = TuckerCore(parent=self.root, dims=left)
        self.root.right = TuckerCore(parent=self.root, dims=right)

        # Reshape initial tensor into a matrix for the first split
        tensor = tensor.reshape(np.prod(left), np.prod(right), order="F")

        self.root.core, [self.root.left.core, self.root.right.core] = hosvd(tensor)
        self.root.get_ranks()

        self.nodes2Expand.append(self.root.left)
        self.nodes2Expand.append(self.root.right)

        while self.nodes2Expand:
            node = self.nodes2Expand.pop(0)
            left, right = split_dimensions(node.dims)

            node.core = node.core.reshape((np.prod(left), np.prod(right), -1), order="F")
            node.core, [lsv1, lsv2, lsv3] = hosvd(node.core)
            # Contract the third leaf with the tucker core for now
            node.core = np.einsum("ijk,lk->ijl", node.core, lsv3, optimize=True)
            node._isexpanded = True
            node.core_idx = _node_counter
            self.transfer_nodes[_node_counter] = node
            _node_counter += 1

            if len(left) == 1:
                # i.e we have a leaf
                node.left = TuckerLeaf(matrix=lsv1, parent=node, dims=left, idx=_leaf_counter)
                self.leaves[_leaf_counter] = node.left
                _leaf_counter += 1
            else:
                node.left = TuckerCore(core=lsv1, parent=node, dims=left)
                self.nodes2Expand.append(node.left)

            if len(right) == 1:
                node.right = TuckerLeaf(matrix=lsv2, parent=node, dims=right, idx=_leaf_counter)
                self.leaves[_leaf_counter] = node.right
                _leaf_counter += 1
            else:
                node.right = TuckerCore(core=lsv2, parent=node, dims=right)
                self.nodes2Expand.append(node.right)

        self._iscompressed = True
        return None

    def compress_leaf2root(self, tensor=None, dimension_tree=None):
        """Compress a tensor using the leaf-to-root approach.

        This method performs hierarchical Tucker decomposition using a bottom-up approach,
        starting from the leaves of the dimension tree and working toward the root.
        The process uses recursive Higher Order Singular Value Decomposition (HOSVD)
        applied in ascending order through the tree.

        The leaf-to-root approach generally provides better error control than root-to-leaf,
        as the approximation errors can be tracked and managed throughout the process.
        If rtol (relative tolerance) is set, this method will adaptively adjust the
        approximation to meet the error tolerance.

        Args:
            tensor (ndarray): Input tensor to be decomposed.
            dimension_tree: Dimension tree defining the hierarchical structure. If None,
                a binary tree will be created automatically using createDimensionTree().

        Returns:
            None

        Raises:
            NotFoundError: If no tensor is provided.
            AssertionError: If the tensor is already compressed.

        Note:
            The results of compression are stored directly in the HTucker object's attributes
            (root, transfer_nodes, leaves) and can be accessed after this method completes.
            The _iscompressed flag is set to True upon successful completion.
        """
        assert self._iscompressed is False
        if tensor is None:
            raise NotFoundError("No tensor is given. Please check if you provided correct input(s)")

        if (self._dimension_tree is None) and (dimension_tree is None):
            warn("No dimension tree is given, creating one with binary splitting now...")
            self._dimension_tree = createDimensionTree(tensor, 2, 1)
        else:
            self._dimension_tree = dimension_tree

        if self.rtol is not None:
            num_svds = 2 * len(tensor.shape) - 3
            tenNorm = np.linalg.norm(tensor)
            cur_norm = tenNorm
            total_allowed_error = tenNorm * self.rtol
            self.allowed_error = tenNorm * self.rtol / np.sqrt(num_svds)  # allowed error per SVD

        # Start with initial HOSVD to get the initial set of leaves
        existing_leaves = []
        # existing_nodes = []
        node_idx = self._dimension_tree._nodeCount - 1

        _, leafs = hosvd(tensor, tol=self.allowed_error)

        for li in self._dimension_tree._level_items[-1]:
            if li._isleaf:
                leaf_idx = li._dimension_index[0]
                self.leaves[leaf_idx] = ht.TuckerLeaf(
                    matrix=leafs[leaf_idx], dims=li.val[0], idx=leaf_idx
                )
                li._ranks[0] = self.leaves[leaf_idx].shape[-1]
                li.real_node = self.leaves[leaf_idx]
                li.parent._ranks[li.parent._ranks.index(None)] = li.real_node.rank

        for lf in self.leaves:
            if (lf is not None) and (lf not in existing_leaves):
                existing_leaves.append(lf)
                tensor = mode_n_product(tensor, lf.core, [lf.leaf_idx, 0])

        # Calculate tensor norm
        tenNorm = np.linalg.norm(tensor)
        cur_norm = tenNorm
        # last_layer = self._dimension_tree._level_items[-1]

        num_svds -= len(self._dimension_tree._level_items[-1])

        # Process intermediate layers
        for layer in self._dimension_tree._level_items[1:-1][::-1]:
            self.allowed_error = np.sqrt(
                (total_allowed_error ** (2)) - max(((tenNorm ** (2)) - (cur_norm ** (2))), 0)
            )
            self.allowed_error = self.allowed_error / np.sqrt(num_svds)

            new_shape = []
            deneme = []

            for item in layer:
                deneme.extend(item._dimension_index)
                if item._isleaf:
                    new_shape.append(item.val[0])
                else:
                    for chld in item.children:
                        item.real_children.append(chld.real_node)
                    new_shape.append(
                        np.prod(list(filter(lambda item: item is not None, item._ranks)))
                    )

            tensor = tensor.reshape(new_shape, order="F")
            _, leafs = hosvd(tensor, tol=self.allowed_error)

            for item_idx, item in enumerate(layer):
                tensor = mode_n_product(tensor, leafs[item_idx], [item_idx, 0])

                if item._isleaf:
                    # The current item is a leaf
                    leaf_idx = item._dimension_index[0]
                    lf = ht.TuckerLeaf(matrix=leafs[item_idx], dims=item.val[0], idx=leaf_idx)
                    self.leaves[leaf_idx] = lf
                    item.real_node = lf
                    item._ranks[0] = lf.rank
                    item.parent._ranks[item.parent._ranks.index(None)] = lf.rank
                else:
                    # The current item is a transfer node
                    item._ranks[item._ranks.index(None)] = leafs[item_idx].shape[-1]
                    item.parent._ranks[item.parent._ranks.index(None)] = leafs[item_idx].shape[-1]

                    node = ht.TuckerCore(
                        core=leafs[item_idx].reshape(item._ranks, order="F"),
                        dims=item.val.copy(),
                        idx=item._dimension_index.copy(),
                    )

                    self.transfer_nodes[node_idx] = node
                    node._isexpanded = True
                    node_idx -= 1

                    if len(layer) != 1:
                        node._isroot = False

                    item.real_node = node

                    for chld in item.real_children:
                        node.children.append(chld)
                        chld.parent = node

                    for chld in item.children:
                        chld.real_parent = node

                    node.get_ranks()

            cur_norm = min(np.linalg.norm(tensor), tenNorm)
            num_svds -= len(layer)
            # last_layer = layer

        # Process root layer
        layer = self._dimension_tree._level_items[0]
        item = layer[0]

        for chld in item.children:
            item.real_children.append(chld.real_node)

        node = ht.TuckerCore(core=tensor, dims=item.val.copy(), idx=item._dimension_index.copy())

        self.root = node
        node._isexpanded = True
        item.real_node = node

        for chld in item.real_children:
            node.children.append(chld)
            chld.parent = node

        for chld in item.children:
            chld.real_parent = node

        node.get_ranks()
        self._iscompressed = True

        return None

    def compress_leaf2root_batch(self, tensor=None, dimension_tree=None, batch_dimension=None):
        """Compress a batch of tensors using the leaf-to-root approach.

        This method performs hierarchical Tucker decomposition on a batch of tensors
        using a bottom-up approach. It processes multiple tensors simultaneously by
        treating one dimension as a batch dimension, enabling efficient decomposition
        of tensor sequences or collections.

        The batch dimension is handled specially - it's excluded from the decomposition
        process and preserved as a separate dimension in the final representation.

        Args:
            tensor (ndarray): Input tensor with a batch dimension.
            dimension_tree: Dimension tree defining the hierarchical structure. If None,
                a binary tree will be created automatically using createDimensionTree().
            batch_dimension (int, optional): Index specifying which dimension is the
                batch dimension. If None, defaults to the last dimension.

        Returns:
            None

        Raises:
            NotFoundError: If no tensor is provided.
            AssertionError: If the tensor is already compressed.

        Note:
            This method is especially useful for applications involving sequences of
            tensors, such as video data or time-evolving simulations, as it can exploit
            common structure across the batch.
        """
        assert self._iscompressed is False
        if tensor is None:
            raise NotFoundError("No tensor is given. Please check if you provided correct input(s)")
        if batch_dimension is None:
            warn("No batch dimension is given, assuming last dimension as batch dimension!")
            batch_dimension = len(tensor.shape) - 1
        batch_count = tensor.shape[batch_dimension]
        if (self._dimension_tree is None) and (dimension_tree is None):
            warn("No dimension tree is given, creating one with binary splitting now...")
            tree_shape = (
                list(tensor.shape)[:batch_dimension] + list(tensor.shape)[batch_dimension + 1 :]
            )
            self._dimension_tree = createDimensionTree(tree_shape, 2, 1)
        else:
            self._dimension_tree = dimension_tree

        if self.rtol is not None:
            num_svds = 2 * (len(tensor.shape) - 1) - 2
            tenNorm = np.linalg.norm(tensor)
            cur_norm = tenNorm
            total_allowed_error = tenNorm * self.rtol
            self.allowed_error = tenNorm * self.rtol / np.sqrt(num_svds)  # allowed error per svd

            # TODO: here, the allowed error does not take the fact that one of the
            # dimensions is the batch dimension and therefore will be ignored

        # print(num_svds,total_allowed_error,self.allowed_error,cur_norm,tenNorm)
        node_idx = self._dimension_tree._nodeCount - 1

        hosvd_dimensions = [
            item._dimension_index[0]
            for item in self._dimension_tree._level_items[-1]
            if item._isleaf
        ]
        leafs = hosvd_only_for_dimensions(
            tensor,
            tol=self.allowed_error,
            dims=hosvd_dimensions,
            # batch_dimension = batch_dimension,
            contract=False,
        )
        for leaf_idx, li, leaf in zip(
            hosvd_dimensions, self._dimension_tree._level_items[-1], leafs
        ):
            if li._isleaf:
                # leaf_idx=li._dimension_index[0]
                self.leaves[leaf_idx] = ht.TuckerLeaf(matrix=leaf, dims=li.val[0], idx=leaf_idx)
                li._ranks[0] = self.leaves[leaf_idx].shape[-1]
                li.real_node = self.leaves[leaf_idx]
                li.parent._ranks[li.parent._ranks.index(None)] = li.real_node.rank
            tensor = mode_n_product(tensor=tensor, matrix=leaf, modes=[leaf_idx, 0])
        cur_norm = min(np.linalg.norm(tensor), tenNorm)
        # print()
        # print(
        #     num_svds,
        #     len(self._dimension_tree._level_items[-1]),
        #     tenNorm,
        #     cur_norm,
        #     total_allowed_error,
        #     self.allowed_error,
        #     (cur_norm**2/tenNorm**2),
        #     np.sqrt(1-(cur_norm**2/tenNorm**2)),
        #     max(((tenNorm**(2))-(cur_norm**(2))),0),
        #     )
        # print()

        num_svds -= len(self._dimension_tree._level_items[-1])
        for layer in self._dimension_tree._level_items[1:-1][::-1]:
            # print(cur_norm)
            self.allowed_error = np.sqrt(
                (total_allowed_error ** (2)) - max(((tenNorm ** (2)) - (cur_norm ** (2))), 0)
            )
            # print(self.allowed_error)
            self.allowed_error = self.allowed_error / np.sqrt(num_svds)
            # print(self.allowed_error)
            # print(num_svds,self.allowed_error,cur_norm,tenNorm)

            hosvd_dimensions = [item._dimension_index[0] for item in layer if item._isleaf]
            # _ , leafs = hosvd(tensor,tol=self.allowed_error)
            if hosvd_dimensions:
                # Compute missing leaves (if any)
                leafs = hosvd_only_for_dimensions(
                    tensor,
                    tol=self.allowed_error,
                    dims=hosvd_dimensions,
                    # batch_dimension = batch_dimension,
                    contract=False,
                )
                leaf_ctr = 0
                # for item_idxx, item in enumerate(layer):
                for item in layer:
                    if item._isleaf:
                        leaf_idx = item._dimension_index[0]
                        self.leaves[leaf_idx] = ht.TuckerLeaf(
                            matrix=leafs[leaf_ctr], dims=item.val[0], idx=leaf_idx
                        )
                        item._ranks[0] = self.leaves[leaf_idx].shape[-1]
                        item.real_node = self.leaves[leaf_idx]
                        # item.parent._ranks[item.parent._ranks.index(None)]=item.real_node.rank
                        item.parent._ranks[item.parent.children.index(item)] = item.real_node.rank
                        leaf_ctr += 1
            # hosvd_dimensions = [item._dimension_index[0] for item in layer if not item._isleaf]

            new_shape = []
            # inter_shape = []
            dimension_shift = 0
            for item in layer:
                child_list = []
                if item.children:
                    if item._dimension_index[0] < batch_dimension:
                        # print(len(item.children)-1)
                        dimension_shift += len(item.children) - 1
                    for chld in item.children:
                        child_list.append(chld.shape[-1])
                else:
                    child_list.append(item.shape[0])
                new_shape.append(np.prod(child_list))
                # inter_shape.append(np.prod(child_list))
            # print(batch_dimension-dimension_shift)

            batch_dimension -= dimension_shift
            new_shape.insert(batch_dimension, batch_count)
            tensor = tensor.reshape(new_shape, order="F")
            hosvd_dimensions = [item_idx for item_idx, item in enumerate(layer) if not item._isleaf]
            nodes = hosvd_only_for_dimensions(
                tensor, tol=self.allowed_error, dims=hosvd_dimensions, contract=False
            )
            node_ctr = 0
            leaf_ctr = 0
            for item_idx, item in enumerate(layer):
                if not item._isleaf:
                    # The current item is a node
                    item._ranks[item._ranks.index(None)] = nodes[node_ctr].shape[-1]
                    # item.parent._ranks[item.parent._ranks.index(None)]=nodes[node_ctr].shape[-1]
                    item.parent._ranks[item.parent.children.index(item)] = nodes[node_ctr].shape[-1]
                    node = ht.TuckerCore(
                        core=nodes[node_ctr].reshape(item._ranks, order="F"),
                        dims=item.val.copy(),
                        idx=item._dimension_index.copy(),
                    )
                    self.transfer_nodes[node_idx] = node

                    tensor = mode_n_product(tensor, nodes[node_ctr], modes=[item_idx, 0])
                    node._isexpanded = True
                    node_idx -= 1
                    if len(layer) != 1:
                        node._isroot = False
                    item.real_node = node
                    for chld in item.real_children:
                        node.children.append(chld)
                        chld.parent = node
                    for chld in item.children:
                        chld.real_parent = node
                    node.get_ranks()

                    node_ctr += 1
                else:
                    # The current item is a leaf
                    tensor = mode_n_product(tensor, leafs[leaf_ctr], modes=[item_idx, 0])
                    leaf_ctr += 1
            cur_norm = min(np.linalg.norm(tensor), tenNorm)
            num_svds -= len(layer)
        layer = self._dimension_tree._level_items[0]
        item = layer[0]
        for chld in item.children:
            item.real_children.append(chld.real_node)
        node = ht.TuckerCore(core=tensor, dims=item.val.copy(), idx=item._dimension_index.copy())
        self.root = node
        node._isexpanded = True
        item.real_node = node
        for chld in item.real_children:
            node.children.append(chld)
            chld.parent = node
        for chld in item.children:
            chld.real_parent = node
        node.get_ranks()
        self._iscompressed = True

        return None

    def reconstruct_all(self):
        """Reconstruct the full tensor from the hierarchical decomposition.

        This method performs a full reconstruction of the original tensor from its
        hierarchical Tucker representation. The reconstruction process traverses
        the dimension tree and applies tensor contractions between cores and
        factor matrices to recover the original tensor.

        For decompositions created with a dimension tree (leaf-to-root approach),
        the reconstruction follows the tree structure. For decompositions created
        without a dimension tree (root-to-leaf approach), it uses a bottom-up
        binary pairing strategy.

        Returns:
            ndarray: The reconstructed full tensor with the original dimensions.

        Raises:
            AssertionError: If the tensor has not been compressed yet.

        Note:
            For large tensors, this operation can be memory-intensive as it
            reconstructs the full tensor. Consider using partial reconstruction
            methods if the full tensor would exceed available memory.
        """
        assert self._iscompressed
        # _transfer_nodes = self.transfer_nodes.copy()

        if self._dimension_tree is None:
            # Use old rebuild method for root-to-leaf compression
            current_level = []
            for leaf in self.leaves:
                current_level.append(leaf)

            next_level = []
            while len(current_level) > 1:
                for i in range(0, len(current_level), 2):
                    if i + 1 < len(current_level):
                        parent = current_level[i].parent
                        parent.contract_children()
                        next_level.append(parent)
                    else:
                        next_level.append(current_level[i])

                current_level = next_level
                next_level = []

            tensor = current_level[0].core
        else:
            # Use dimension tree rebuild
            for layer in self._dimension_tree._level_items[:-1]:
                for item in layer:
                    if hasattr(item, "real_node") and item.real_node is not None:
                        item.real_node.contract_children_dimension_tree()

            tensor = self.root.core
            if len(self.root.core.shape) > len(self.original_shape):
                # We need to reshape back to original shape
                tensor = tensor.reshape(self.original_shape, order="F")

        self._iscompressed = False
        return tensor

    def project(self, new_tensor, batch=False, batch_dimension=None):
        """Project a tensor onto the basis of the hierarchical Tucker decomposition.

        This method projects a new tensor onto the basis vectors of an existing
        hierarchical Tucker decomposition. The projection effectively represents
        the new tensor using the same subspaces as the original decomposition,
        which is useful for dimensionality reduction, comparison, or as part of
        incremental update procedures.

        Args:
            new_tensor (ndarray): The tensor to project onto the hierarchical Tucker basis.
                Must be compatible with the original tensor shape.
            batch (bool, optional): Whether to treat the tensor as a batch. Default is False.
            batch_dimension (int, optional): Which dimension to treat as the batch dimension.
                Only used if batch is True. If None, uses the batch dimension from initialization.

        Returns:
            ndarray: The projected tensor in the hierarchical Tucker basis.

        Raises:
            AssertionError: If the decomposition has not been compressed yet.
            ValueError: If the new tensor shape is incompatible with the original tensor shape.

        Note:
            Projection is a key operation in incremental algorithms, as it allows new data
            to be efficiently represented in terms of the existing decomposition's structure.
        """
        assert self._iscompressed is True
        new_tensor_shape = new_tensor.shape
        if batch:
            if (
                list(new_tensor_shape[:batch_dimension] + new_tensor_shape[batch_dimension + 1 :])
                != self.original_shape
            ):
                try:
                    shape = np.arange(len(new_tensor_shape)).tolist()
                    shape.pop(batch_dimension)
                    shape = shape + [batch_dimension]
                    new_tensor.transpose(shape)
                    new_tensor = new_tensor.reshape(self.original_shape + [-1], order="F")
                except ValueError:
                    warn(
                        f"Presented tensor has shape {new_tensor.shape}, which is not\
                            compatible with {tuple(self.original_shape)}!"
                    )
        else:
            if list(new_tensor_shape) != self.original_shape:
                try:
                    new_tensor = new_tensor.reshape(self.original_shape, order="F")
                    # 2+2
                except ValueError:
                    warn(
                        f"Presented tensor has shape {new_tensor.shape}, which is not\
                            compatible with {tuple(self.original_shape)}!"
                    )

        for layer in self._dimension_tree._level_items[::-1][:-1]:
            idxCtr = 0
            strings = []
            last_char = 97
            dims = len(new_tensor.shape)
            coreString = [chr(idx) for idx in range(last_char, last_char + dims)]
            strings.append("".join(coreString))
            last_char += dims
            # for itemIdx, item in enumerate(layer):
            for item in layer:
                if type(item.real_node) is ht.TuckerLeaf:
                    if (item in self._dimension_tree._level_items[-1]) and (
                        item.real_node.leaf_idx != idxCtr
                    ):
                        idxCtr += 1
                    strings.append(strings[0][idxCtr] + chr(last_char))
                    coreString[idxCtr] = chr(last_char)
                    last_char += 1
                    idxCtr += 1
                elif type(item.real_node) is ht.TuckerCore:
                    # keep a separate small counter inside, increment by 1 for leaf and
                    # by 2 for core, and then just finish
                    # also reset the counter in each layer.
                    contractionDims = len(item.shape) - 1
                    strings.append(strings[0][idxCtr : idxCtr + contractionDims] + chr(last_char))
                    coreString[idxCtr] = chr(last_char)
                    last_char += 1
                    for stringIdx in range(1, contractionDims):
                        coreString[idxCtr + stringIdx] = ""
                    # last_char += 1
                    idxCtr += contractionDims
                else:
                    ValueError(f"Unknown node type! {type(item)} is not known!")
            try:
                new_tensor = eval(
                    "np.einsum("
                    + "'"
                    + ",".join(
                        [
                            ",".join(strings) + "->" + "".join(coreString) + "'",
                            "new_tensor",
                            ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"',
                        ]  # Check here if there's a problem order="F" was added later
                    )
                    + ")"
                )
            except ValueError:
                for ii, string in enumerate(strings):
                    tempstr = [*string]
                    for jj, chrs in enumerate(tempstr):
                        if ord(chrs) > ord("z"):
                            strings[ii] = strings[ii].replace(
                                chrs, chr(ord(chrs) - ord("z") + ord("A") - 1), jj
                            )
                for jj, chrs in enumerate(coreString):
                    if ord(chrs) > ord("z"):
                        coreString[jj] = chr(ord(chrs) - ord("z") + ord("A") - 1)
                new_tensor = eval(
                    "np.einsum("
                    + "'"
                    + ",".join(
                        [
                            ",".join(strings) + "->" + "".join(coreString) + "'",
                            "new_tensor",
                            ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"',
                        ]  # Check here if there's a problem. order="F" was added later
                    )
                    + ")"
                )
        return new_tensor

    def reconstruct(self, core, batch=False):
        """Reconstruct a tensor from a given core and the hierarchical decomposition.

        This method performs reconstruction starting from a provided core tensor,
        contracting it with the factor matrices stored in the hierarchical Tucker
        representation. This allows for flexible reconstructions, such as using
        a modified core tensor while keeping the same basis.

        Args:
            core (ndarray): The core tensor to use for reconstruction.
            batch (bool): Whether to handle the reconstruction as a batch operation.

        Returns:
            ndarray: The reconstructed tensor.

        Note:
            This method is particularly useful for operations that modify the core
            while preserving the hierarchical structure, such as tensor algebra
            operations or parameter studies.
        """
        for layer in self._dimension_tree._level_items[1:]:
            # idxCtr = 0
            strings = []
            last_char = 97
            dims = len(core.shape)
            coreString = [chr(idx) for idx in range(last_char, last_char + dims)]
            strings.append("".join(coreString))
            last_char += dims
            for itemIdx, item in enumerate(layer):
                if type(item.real_node) is ht.TuckerLeaf:
                    tempStr = chr(last_char)
                    if (item in self._dimension_tree._level_items[-1]) and (
                        item.real_node.leaf_idx != itemIdx
                    ):
                        strings.append(tempStr + strings[0][item.real_node.leaf_idx])
                        coreString[item.real_node.leaf_idx] = tempStr
                        # idxCtr+=1
                    else:
                        strings.append(tempStr + strings[0][itemIdx])
                        coreString[itemIdx] = tempStr
                    last_char += 1
                    # idxCtr+=1
                elif type(item.real_node) is ht.TuckerCore:
                    contractionDims = len(item.shape) - 1
                    tempStr = "".join(
                        [chr(last_char + stringIdx) for stringIdx in range(0, contractionDims)]
                    )  # Sorun olursa 0i 1 yap
                    strings.append(
                        tempStr
                        + strings[0][itemIdx]
                        # strings[0][idxCtr:idxCtr+contractionDims]+chr(last_char)
                    )
                    coreString[itemIdx] = tempStr
                    last_char += contractionDims
                    # idxCtr += contractionDims
                else:
                    ValueError(f"Unknown node type! {type(item)} is not known!")
            try:
                core = eval(
                    "np.einsum("
                    + "'"
                    + ",".join(
                        [
                            ",".join(strings) + "->" + "".join(coreString) + "'",
                            "core",
                            ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"',
                        ]  # Check here if there's a problem. order="F" was added later
                    )
                    + ")"
                )
            except ValueError:
                for ii, string in enumerate(strings):
                    tempstr = [*string]
                    for jj, chrs in enumerate(tempstr):
                        if ord(chrs) > ord("z"):
                            tempstr[jj] = chr(ord(chrs) - ord("z") + ord("A") - 1)
                            strings[ii] = "".join(tempstr)
                for jj, chrs in enumerate(coreString):
                    if ord(chrs) > ord("z"):
                        coreString[jj] = chr(ord(chrs) - ord("z") + ord("A") - 1)
                core = eval(
                    "np.einsum("
                    + "'"
                    + ",".join(
                        [
                            ",".join(strings) + "->" + "".join(coreString) + "'",
                            "core",
                            ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"',
                        ]  # Check here if there's a problem. order="F" was added later
                    )
                    + ")"
                )

        return core

    def incremental_update(self, new_tensor):
        """Update the hierarchical Tucker decomposition incrementally with a new tensor.

        This method efficiently updates an existing hierarchical Tucker decomposition
        to incorporate a new tensor without performing a full recomputation. It uses
        the structure and basis of the existing decomposition to rapidly integrate
        the new information.

        The method is particularly useful for scenarios where tensors arrive sequentially
        or where the data evolves over time, as it preserves most of the computational
        work done in previous decompositions.

        Args:
            new_tensor (ndarray): New tensor to incorporate into the decomposition.
                Must have the same shape as the original tensor.

        Returns:
            ndarray: The updated core tensor after incorporating the new information.

        Raises:
            ValueError: If the new tensor cannot be reshaped to match the original tensor shape.

        Note:
            This is an implementation of the HT-RISE (Hierarchical Tucker - Rapid Incremental
            Subspace Expansion) algorithm which provides mathematically proven approximation
            error upper bounds.
        """
        if list(new_tensor.shape) != self.original_shape:
            try:
                new_tensor = new_tensor.reshape(self.original_shape, order="F")
            except ValueError:
                warn(
                    f"Presented tensor has shape {new_tensor.shape}, which is not compatible\
                    with {tuple(self.original_shape)}!"
                )

        core = self.project(new_tensor)
        reconstruction = self.reconstruct(core)
        tenNorm = np.linalg.norm(new_tensor)

        if (np.linalg.norm(new_tensor - reconstruction) / tenNorm) <= self.rtol:
            warn("Current tensor network is sufficient, no need to update the cores.")
            return core

        allowed_error = tenNorm * self.rtol / np.sqrt(2 * len(new_tensor.shape) - 3)

        for layer in self._dimension_tree._level_items[::-1][:-1]:
            idxCtr = 0
            # for itemIdx, item in enumerate(layer):
            for item in layer:
                # Find orthonormal vectors
                if type(item.real_node) is ht.TuckerLeaf:
                    if (item in self._dimension_tree._level_items[-1]) and (
                        item.real_node.leaf_idx != idxCtr
                    ):
                        idxCtr += 1

                    tempTens = np.tensordot(new_tensor, item.real_node.core, axes=[idxCtr, 0])
                    tempTens = np.tensordot(tempTens, item.real_node.core, axes=[idxCtr, 0])
                    tempTens = tempTens - new_tensor

                    u, s, _ = np.linalg.svd(
                        ht.mode_n_unfolding(tempTens, idxCtr), full_matrices=False
                    )
                    idxCtr += 1

                elif type(item.real_node) is ht.TuckerCore:
                    contractionDims = len(item.shape) - 1

                    # Create proper axes for tensordot
                    axes_1 = list(range(idxCtr, idxCtr + contractionDims))
                    axes_2 = list(range(0, contractionDims))
                    tempTens = np.tensordot(new_tensor, item.real_node.core, axes=(axes_1, axes_2))
                    tempTens = np.tensordot(
                        tempTens, item.real_node.core, axes=[idxCtr, contractionDims]
                    )
                    tempTens = tempTens - new_tensor

                    new_shape = list(new_tensor.shape)
                    new_shape = (
                        new_shape[:idxCtr]
                        + [np.prod(new_shape[idxCtr : idxCtr + contractionDims])]
                        + new_shape[idxCtr + contractionDims :]
                    )
                    u, s, _ = np.linalg.svd(
                        ht.mode_n_unfolding(tempTens.reshape(new_shape, order="F"), idxCtr),
                        full_matrices=False,
                    )
                    idxCtr += contractionDims

                else:
                    raise ValueError(f"Unknown node type! {type(item)} is not known!")

                # Core updating
                u = u[:, np.cumsum((s**2)[::-1])[::-1] > (allowed_error) ** 2]
                u_shape = list(item.shape)[:-1] + [-1]
                item.real_node.core = np.concatenate(
                    (item.real_node.core, u.reshape(u_shape, order="F")), axis=-1
                )
                item.real_node.get_ranks()

                # Rank matching
                if item.parent._dimension_index.index(item._dimension_index[0]) == 0:
                    ranks = item.real_parent.ranks
                    item.real_parent.core = np.concatenate(
                        (item.real_parent.core, np.zeros([u.shape[-1]] + ranks[1:])), axis=0
                    )
                else:
                    ranks = item.real_parent.ranks
                    item.real_parent.core = np.concatenate(
                        (item.real_parent.core, np.zeros([ranks[0], u.shape[-1]] + ranks[2:])),
                        axis=1,
                    )

                item.real_parent.get_ranks()

            # Project through layer
            new_tensor = self.project(new_tensor)

        return new_tensor

    def incremental_update_batch(self, new_tensor, batch_dimension=None, append=True):
        """Incrementally update the decomposition with a new batch tensor.

        This method efficiently updates a batch hierarchical Tucker decomposition
        with new tensor data. It can either append the new data to the existing
        batch or replace portions of it, making it useful for sequence data,
        streaming data applications, or time-evolving tensor data.

        The batch version of incremental updates maintains the efficiency advantages
        of the standard incremental update while handling multiple tensors in a
        batch-oriented manner.

        Args:
            new_tensor (ndarray): New tensor data to incorporate into the decomposition.
                Must have compatible dimensions with the original tensor.
            batch_dimension (int, optional): The dimension to treat as the batch dimension.
                If None, defaults to the batch dimension specified during initialization.
            append (bool): Whether to append the new tensor to the existing batch (True)
                or replace existing items in the batch (False).

        Returns:
            bool: True if the decomposition was successfully updated, False otherwise.

        Raises:
            AssertionError: If the tensor has not been compressed yet.
            ValueError: If the new tensor cannot be reshaped to match dimensions.

        Note:
            This is an extension of the HT-RISE algorithm for batch processing, enabling
            efficient updates to sequences of tensors without full recomputation.
        """
        assert self._iscompressed is True
        new_tensor_shape = new_tensor.shape

        if (
            list(new_tensor_shape[:batch_dimension] + new_tensor_shape[batch_dimension + 1 :])
            != self.original_shape
        ):
            try:
                shape = np.arange(len(new_tensor_shape)).tolist()
                shape.pop(batch_dimension)
                shape = shape + [batch_dimension]
                new_tensor.transpose(shape)
                new_tensor = new_tensor.reshape(self.original_shape + [-1], order="F")
            except ValueError:
                pass

        core = self.project(new_tensor, batch=True, batch_dimension=batch_dimension)

        tenNorm = np.linalg.norm(new_tensor)

        if np.linalg.norm(core) >= tenNorm * np.sqrt(1 - self.rtol**2):
            self.batch_count += new_tensor_shape[batch_dimension]
            if append:
                self.root.core = np.concatenate((self.root.core, core), axis=-1)
                self.root.get_ranks()
                return False
            else:
                return False

        cur_norm = tenNorm
        total_allowed_error = tenNorm * self.rtol
        allowed_error = total_allowed_error
        num_svds = 2 * (len(new_tensor.shape) - 1) - 2

        for layer in self._dimension_tree._level_items[::-1][:-1]:
            allowed_error = allowed_error / np.sqrt(num_svds)
            idxCtr = 0

            # for itemIdx, item in enumerate(layer):
            for item in layer:
                strings = []
                last_char = 97
                dims = len(new_tensor.shape)
                coreString = [chr(idx) for idx in range(last_char, last_char + dims)]
                strings.append("".join(coreString))
                last_char += dims

                contractionDims = len(item.shape) - 1

                # Find orthonormal vectors
                if type(item.real_node) is ht.TuckerLeaf:
                    if (item in self._dimension_tree._level_items[-1]) and (
                        item.real_node.leaf_idx != idxCtr
                    ):
                        idxCtr += 1
                    strings.append(strings[0][idxCtr] + chr(last_char))
                    strings.append(chr(last_char + contractionDims) + chr(last_char))
                    coreString[idxCtr] = chr(last_char + contractionDims)

                    tempTens = eval(
                        "np.einsum("
                        + "'"
                        + ",".join(
                            [
                                ",".join(strings) + "->" + "".join(coreString) + "'",
                                "new_tensor",
                                "item.real_node.core,item.real_node.core",
                                'optimize=True,order="F"',
                            ]
                        )
                        + ")"
                    )

                    tempTens = tempTens - new_tensor
                    u, s, _ = np.linalg.svd(
                        ht.mode_n_unfolding(tempTens, idxCtr), full_matrices=False
                    )
                    idxCtr += 1

                elif type(item.real_node) is ht.TuckerCore:
                    strings.append(strings[0][idxCtr : idxCtr + contractionDims] + chr(last_char))
                    strings.append(
                        "".join(
                            [
                                chr(stringIdx + 1)
                                for stringIdx in range(last_char, last_char + contractionDims)
                            ]
                        )
                        + chr(last_char)
                    )

                    for stringIdx in range(contractionDims):
                        coreString[idxCtr + stringIdx] = strings[-1][stringIdx]

                    tempTens = eval(
                        "np.einsum("
                        + "'"
                        + ",".join(
                            [
                                ",".join(strings) + "->" + "".join(coreString) + "'",
                                "new_tensor",
                                "item.real_node.core,item.real_node.core",
                                'optimize=True,order="F"',
                            ]
                        )
                        + ")"
                    )

                    tempTens = tempTens - new_tensor
                    new_shape = list(new_tensor.shape)
                    new_shape = (
                        new_shape[:idxCtr]
                        + [np.prod(new_shape[idxCtr : idxCtr + contractionDims])]
                        + new_shape[idxCtr + contractionDims :]
                    )

                    try:
                        u, s, _ = np.linalg.svd(
                            ht.mode_n_unfolding(tempTens.reshape(new_shape, order="F"), idxCtr),
                            full_matrices=False,
                        )
                    except np.linalg.LinAlgError:
                        print("Numpy SVD did not converge, using QR+SVD")
                        q, r = np.linalg.qr(
                            ht.mode_n_unfolding(tempTens.reshape(new_shape, order="F"), idxCtr)
                        )
                        u, s, _ = np.linalg.svd(r, full_matrices=False)
                        u = q @ u

                    idxCtr += contractionDims

                else:
                    raise ValueError(f"Unknown node type! {type(item)} is not known!")

                # Core updating
                u = u[:, np.cumsum((s**2)[::-1])[::-1] > (allowed_error) ** 2]
                u_shape = list(item.shape)[:-1] + [-1]
                item.real_node.core = np.concatenate(
                    (item.real_node.core, u.reshape(u_shape, order="F")), axis=-1
                )
                item.real_node.get_ranks()

                # Rank matching
                if item.parent._dimension_index.index(item._dimension_index[0]) == 0:
                    ranks = item.real_parent.ranks
                    item.real_parent.core = np.concatenate(
                        (item.real_parent.core, np.zeros([u.shape[-1]] + ranks[1:])), axis=0
                    )
                else:
                    ranks = item.real_parent.ranks
                    item.real_parent.core = np.concatenate(
                        (item.real_parent.core, np.zeros([ranks[0], u.shape[-1]] + ranks[2:])),
                        axis=1,
                    )

                item.real_parent.get_ranks()

            # Project through layer
            idxCtr = 0
            strings = []
            last_char = 97
            dims = len(new_tensor.shape)
            coreString = [chr(idx) for idx in range(last_char, last_char + dims)]
            strings.append("".join(coreString))
            last_char += dims

            # for itemIdx, item in enumerate(layer):
            for item in layer:
                if type(item.real_node) is ht.TuckerLeaf:
                    if (item in self._dimension_tree._level_items[-1]) and (
                        item.real_node.leaf_idx != idxCtr
                    ):
                        idxCtr += 1
                    strings.append(strings[0][idxCtr] + chr(last_char))
                    coreString[idxCtr] = chr(last_char)
                    last_char += 1
                    idxCtr += 1

                elif type(item.real_node) is ht.TuckerCore:
                    contractionDims = len(item.shape) - 1
                    strings.append(strings[0][idxCtr : idxCtr + contractionDims] + chr(last_char))
                    coreString[idxCtr] = chr(last_char)
                    last_char += 1

                    for stringIdx in range(1, contractionDims):
                        coreString[idxCtr + stringIdx] = ""

                    idxCtr += contractionDims

                else:
                    raise ValueError(f"Unknown node type! {type(item)} is not known!")

            try:
                new_tensor = eval(
                    "np.einsum("
                    + "'"
                    + ",".join(
                        [
                            ",".join(strings) + "->" + "".join(coreString) + "'",
                            "new_tensor",
                            ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"',
                        ]
                    )
                    + ")"
                )
            except ValueError:
                # Handle character overflow by remapping to uppercase
                for ii, string in enumerate(strings):
                    tempstr = [*string]
                    for jj, chrs in enumerate(tempstr):
                        if ord(chrs) > ord("z"):
                            strings[ii] = strings[ii].replace(
                                chrs, chr(ord(chrs) - ord("z") + ord("A") - 1), jj
                            )

                for jj, chrs in enumerate(coreString):
                    if chrs:
                        temp_chrs = [*chrs]
                        for kk, ch in enumerate(temp_chrs):
                            if ord(ch) > ord("z"):
                                temp_chrs[kk] = chr(ord(ch) - ord("z") + ord("A") - 1)
                        coreString[jj] = "".join(temp_chrs)

                new_tensor = eval(
                    "np.einsum("
                    + "'"
                    + ",".join(
                        [
                            ",".join(strings) + "->" + "".join(coreString) + "'",
                            "new_tensor",
                            ",".join([f"layer[{idx}].real_node.core" for idx in range(len(layer))]),
                            'optimize=True,order="F"',
                        ]
                    )
                    + ")"
                )

            num_svds -= len(layer)
            cur_norm = np.linalg.norm(new_tensor)
            allowed_error = np.sqrt(
                (total_allowed_error ** (2)) - max(((tenNorm ** (2)) - (cur_norm ** (2))), 0)
            )

        self.batch_count += new_tensor_shape[batch_dimension]
        if append:
            self.root.core = np.concatenate((self.root.core, new_tensor), axis=-1)
            self.root.get_ranks()
            return True
        else:
            return True

    def compress_sanity_check(self, tensor):
        """Perform a sanity check on the compression process.

        Args:
            tensor (ndarray): Input tensor

        Returns:
            tuple: Components of the hierarchical Tucker decomposition
        """
        mat_tensor = np.reshape(
            tensor,
            (tensor.shape[0] * tensor.shape[1], tensor.shape[2] * tensor.shape[3]),
            order="F",
        )

        [u, s, v] = truncated_svd(mat_tensor, 1e-8, full_matrices=False)
        [core_l, lsv_l] = hosvd(u.reshape(tensor.shape[0], tensor.shape[1], -1, order="F"))
        [core_r, lsv_r] = hosvd(v.reshape(-1, tensor.shape[2], tensor.shape[3], order="F"))

        core_l = np.einsum("ijk,lk->ijl", core_l, lsv_l[-1], optimize=True)
        core_r = np.einsum("ijk,li->ljk", core_r, lsv_r[0], optimize=True)
        top = np.diag(s)

        return (lsv_l[0], lsv_l[1], lsv_r[1], lsv_r[2], core_l, core_r, top)

    @property
    def compression_ratio(self):
        """Calculate the compression ratio.

        Returns:
            float: Compression ratio
        """
        num_entries = np.prod(self.root.shape[:-1]) * self.batch_count
        for tf in self.transfer_nodes:
            if tf is not None:
                num_entries += np.prod(tf.shape)
        for lf in self.leaves:
            if lf is not None:
                num_entries += np.prod(lf.shape)
        return np.prod(self.original_shape) * self.batch_count / num_entries

    def save(self, fileName, fileType="hto", directory="./"):
        """Save the HTucker object to a file for later retrieval.

        This method serializes the entire hierarchical Tucker decomposition to a file
        for persistence. The default format is a pickle-based representation with the
        .hto extension (hierarchical Tucker object), which preserves all properties
        and components of the decomposition.

        The method supports different file formats, though currently only the .hto
        format is fully implemented.

        Args:
            fileName (str): Name of the file to save the decomposition to. If the file
                extension is included, it will override the fileType parameter.
            fileType (str, optional): Type of file to save as. Default is "hto"
                (hierarchical Tucker object), which uses Python's pickle format.
            directory (str, optional): Directory to save the file in. Default is the
                current directory.

        Raises:
            NameError: If the file name contains more than one dot (.) or the file
                extension is unknown.
            NotImplementedError: If the specified file type is not yet supported.

        Note:
            The .hto format stores the complete object using Python's pickle module,
            which ensures all attributes and methods are preserved.
        """
        if len(fileName.split(".")) == 2:
            # File extension is given in the file name
            fileType = fileName.split(".")[1]
        elif len(fileName.split(".")) >= 2:
            raise NameError(f"Filename {fileName} can not have more than 1 '.'!")
        else:
            fileName = fileName + "." + fileType

        if fileType == "hto":
            # Save to a hierarchical tucker object file
            with open(os.path.join(directory, fileName), "wb") as f:
                pckl.dump(self, f)
        elif fileType == "npy":
            # Save htucker object to numpy arrays
            raise NotImplementedError("This function is not implemented yet")
        else:
            raise NameError(f"Unknown file extension {fileType}")

    @staticmethod
    def load(file, directory="./"):
        """Load a previously saved HTucker object from a file.

        This static method deserializes a hierarchical Tucker decomposition from a file
        that was previously saved using the save() method. It reconstructs the complete
        HTucker object with all its components and properties.

        The method supports different file formats, though currently only the .hto
        format (hierarchical Tucker object) is fully implemented.

        Args:
            file (str): Name of the file to load the HTucker object from. Should include
                the file extension (e.g., "my_decomposition.hto").
            directory (str, optional): Directory to load the file from. Default is the
                current directory.

        Returns:
            HTucker: A fully reconstructed HTucker object with all components and properties.

        Raises:
            AssertionError: If the file name contains directory separators.
            NameError: If the file name contains more than one dot (.) or if the file
                extension is unknown.
            NotImplementedError: If the specified file type is not yet supported.
            FileNotFoundError: If the specified file does not exist.

        Note:
            This is a static method, so it should be called on the class itself
            (HTucker.load()) rather than on an instance.
        """
        # File address should be given as directory variable
        assert len(file.split(os.sep)) == 1, "Please give address as directory variable."

        if len(file.split(".")) == 2:
            # File extension is given in the file name
            _, fileType = file.split(".")
        elif len(file.split(".")) >= 2:
            raise NameError(f"Filename {file} can not have more than 1 '.'!")

        if fileType == "hto":
            # File is a hierarchical tucker object file
            with open(os.path.join(directory, file), "rb") as f:
                return pckl.load(f)
        elif fileType == "npy":
            # Load htucker object using numpy arrays
            raise NotImplementedError("This function is not implemented yet")
        else:
            raise NameError(f"Unknown file extension {fileType}")

    def get_ranks(self):
        """Get the ranks of the hierarchical Tucker decomposition.

        This method returns the ranks at each node of the hierarchical Tucker
        representation. The ranks determine the dimensionality of the transfer
        tensors and are critical for understanding the compression level and
        approximation quality of the decomposition.

        Lower ranks lead to higher compression but may introduce higher approximation
        error. In an optimal decomposition, the ranks are chosen to balance these factors.

        Returns:
            dict: A dictionary with node identifiers as keys and their ranks as values.
                Keys include "root" for the root node, "node{i}" for transfer nodes,
                and "leaf{i}" for leaf nodes.

        Note:
            Returns None if the tensor has not been compressed yet.
            The rank of a node indicates the number of basis vectors used at that node
            to represent the tensor subspace.
        """
        if not self._iscompressed:
            return None

        ranks = {}
        if self.root is not None:
            ranks["root"] = self.root.U.shape[1]  # Transfer rank of the root node

        if self.transfer_nodes is not None:
            for i, node in enumerate(self.transfer_nodes):
                if node is not None:
                    ranks[f"node{i}"] = node.U.shape[1]  # Transfer rank of each node

        if self.leaves is not None:
            for i, leaf in enumerate(self.leaves):
                if leaf is not None:
                    ranks[f"leaf{i}"] = leaf.B.shape[-1]  # Rank of each leaf

        return ranks

    def get_memory_size(self):
        """Calculate the memory size of the hierarchical Tucker decomposition in bytes.

        This method computes the total memory consumption of the hierarchical Tucker
        representation by summing the memory used by all components: the root node,
        transfer nodes, leaf nodes, and associated metadata. The calculation is based
        on the actual numpy array sizes used in the representation.

        This is useful for understanding the compression efficiency of the decomposition
        compared to storing the full tensor, and for memory requirement analysis.

        Returns:
            float: Total memory size in bytes used by the decomposition.

        Note:
            Returns 0 if the tensor has not been compressed yet.
            The calculation includes all arrays (U matrices, B matrices, etc.) but uses
            an estimate for the dimension tree structure.
        """
        if not self._iscompressed:
            return 0

        memory_size = 0

        # Add root node memory
        if self.root is not None:
            if hasattr(self.root, "U") and self.root.U is not None:
                memory_size += self.root.U.nbytes
            if hasattr(self.root, "B") and self.root.B is not None:
                memory_size += self.root.B.nbytes

        # Add transfer nodes memory
        if self.transfer_nodes is not None:
            for node in self.transfer_nodes:
                if node is not None:
                    if hasattr(node, "U") and node.U is not None:
                        memory_size += node.U.nbytes
                    if hasattr(node, "B") and node.B is not None:
                        memory_size += node.B.nbytes

        # Add leaves memory
        if self.leaves is not None:
            for leaf in self.leaves:
                if leaf is not None:
                    if hasattr(leaf, "U") and leaf.U is not None:
                        memory_size += leaf.U.nbytes
                    if hasattr(leaf, "B") and leaf.B is not None:
                        memory_size += leaf.B.nbytes

        # Add dimension tree metadata (rough estimate)
        if self._dimension_tree is not None:
            # Estimate memory for tree structure
            memory_size += 1000  # Rough estimate for tree structure overhead

        return memory_size
