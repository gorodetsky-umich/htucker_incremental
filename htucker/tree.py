"""Tree implementation for Hierarchical Tucker decomposition."""

from warnings import warn

import numpy as np

from .utils import NotFoundError

# class Node:
#     """Node class for dimension tree implementation."""

#     def __init__(self, val, children=None, parent=None):
#         """Initialize a Node in the dimension tree.

#         Args:
#             val: Value of the node
#             children: Child nodes (default: None)
#             parent: Parent node (default: None)
#         """
#         self.val = val
#         self.parent = parent
#         self.children = [] if children is None else children
#         self._isleaf = True
#         self._ranks = []
#         self._dimension_index = []
#         self.real_node = None
#         self.real_parent = None
#         self.real_children = []

#     def __str__(self):
#         """String representation of the node."""
#         return str(self.val)

#     def adjust_ranks(self):
#         """Adjust ranks based on children."""
#         if hasattr(self, "_ranks") and len(self._ranks) > 0:
#             for i, rank in enumerate(self._ranks):
#                 if rank is None:
#                     continue
#                 if i < len(self.children):
#                     self.children[i]._ranks[self.children[i]._ranks.index(None)] = rank

#     @property
#     def shape(self):
#         """Get the shape of the node."""
#         return self.val


# class Tree:
#     """Tree implementation for dimension hierarchy."""

#     def __init__(self):
#         """Initialize an empty tree."""
#         self.root = None
#         self._size = 0
#         self._leafCount = 0
#         self._depth = 0
#         self._max_depth = 0
#         self._level_items = []
#         self._nodeCount = 0

#     def findNode(self, node, key, check_propagation=False):
#         """Find a node in the tree.

#         Args:
#             node: Starting node for search
#             key: Value to find
#             check_propagation (bool): Whether to check propagation

#         Returns:
#             Node: Found node or None
#         """
#         if node is None:
#             return None

#         if node.val == key:
#             return node

#         for child in node.children:
#             found = self.findNode(child, key, check_propagation)
#             if found is not None:
#                 return found

#         return None

#     def isEmpty(self):
#         """Check if the tree is empty.

#         Returns:
#             bool: True if empty, False otherwise
#         """
#         return self.root is None

#     def initializeTree(self, vals):
#         """Initialize the tree with values.

#         Args:
#             vals: Values to initialize the tree with
#         """
#         for val in vals:
#             self.insertNode(val)

#     def insertNode(self, val, parent=None, dim_index=None):
#         """Insert a node into the tree.

#         Args:
#             val: Value of the node
#             parent: Parent node (default: None)
#             dim_index: Dimension index (default: None)

#         Returns:
#             Node: Inserted node
#         """
#         newNode = Node(val, parent=parent)
#         if dim_index is not None:
#             newNode._dimension_index = dim_index

#         if parent is not None:
#             parent.children.append(newNode)
#             parent._isleaf = False
#             newNode._ranks = [None] * (len(parent._ranks) - 1)
#         else:
#             # This is the root node
#             if self.root is None:
#                 self.root = newNode
#                 self._size = 1
#                 # initialize ranks for root node
#                 newNode._ranks = [None] * len(val)
#             else:
#                 raise ValueError("Tree already has a root")

#         return newNode

#     def get_max_depth(self):
#         """Calculate and store the maximum depth of the tree."""
#         def _get_depth(node, depth):
#             if node is None:
#                 return 0

#             if not node.children:
#                 self._leafCount += 1
#                 return depth

#             max_depth = 0
#             for child in node.children:
#                 child_depth = _get_depth(child, depth + 1)
#                 max_depth = max(max_depth, child_depth)

#             return max_depth

#         self._depth = _get_depth(self.root, 0)
#         self._max_depth = self._depth
#         self.get_items_from_level()
#         return self._depth

#     def get_items_from_level(self):
#         """Get items from each level of the tree."""
#         self._level_items = [[] for _ in range(self._depth + 1)]

#         def _traverse(node, level):
#             if node is None:
#                 return

#             self._level_items[level].append(node)
#             for child in node.children:
#                 _traverse(child, level + 1)

#         _traverse(self.root, 0)
#         return self._level_items

#     def toList(self):
#         """Convert tree to list representation."""
#         if self.isEmpty():
#             return []

#         result = []

#         def _traverse(node):
#             if node is None:
#                 return

#             result.append(node.val)
#             for child in node.children:
#                 _traverse(child)

#         _traverse(self.root)
#         return result


class Node:
    """Node class for dimension tree implementation in Hierarchical Tucker decomposition.

    This class represents a node in the dimension tree, storing information about the node's
    value (tensor dimensions), parent-child relationships, ranks, and associated tensor data.

    Attributes:
        children: List of child nodes.
        val: Value of the node (typically a list of dimensions or an integer).
        parent: Reference to the parent node.
        real_children: References to real (tensor) children nodes.
        real_parent: References to real (tensor) parent nodes.
        real_node: Reference to the associated tensor node.
        _ranks: List of ranks for tensor contractions.
        _propagated: Flag indicating if the node has been processed for propagation.
        _isleaf: Flag indicating if the node is a leaf node.
        _level: The level of the node in the tree (root is 0).
        _dimension_index: List of dimension indices this node represents.
    """

    def __init__(self, val, children=None, parent=None) -> None:
        """Initialize a Node in the dimension tree.

        Args:
            val: Value of the node (typically dimensions of a tensor).
            children: List of child nodes (default: None).
            parent: Reference to parent node (default: None).
        """
        self.children: list["Node"] = children or []
        self.val: list[int] | int = val
        self.parent: Node | None = parent
        self.real_children: list["Node"] = []
        self.real_parent: list["Node"] = []
        self.real_node = None
        self._ranks: list[int | None] = []
        self._propagated: bool = False
        self._isleaf: bool = False
        self._level: int | None = None
        self._dimension_index: list[int] | None = None

    def __str__(self) -> str:
        """Return string representation of the node.

        Returns:
            str: String representation of the node's value.
        """
        return str(self.val)

    def adjust_ranks(self) -> None:
        """Adjust the ranks list based on the number of children.

        This method ensures that the _ranks list has appropriate length:
        - For root nodes: at least as many elements as there are children
        - For other nodes: at least as many elements as (children + 1)

        If the _ranks list is too short, it's extended with None values.
        """
        if self.parent is None:
            # This is the root node
            if len(self._ranks) < len(self.children):
                diff = len(self.children) - len(self._ranks)
                self._ranks += [None] * diff
        else:
            # This is any node (incl. leaves)
            if len(self._ranks) < len(self.children) + 1:
                diff = len(self.children) - len(self._ranks) + 1
                self._ranks += [None] * diff

    @property
    def shape(self):
        """Get the shape of the associated tensor.

        Returns:
            The shape of the real node's tensor if it exists, otherwise a warning.
        """
        if self.real_node:
            return self.real_node.shape
        else:
            return warn("No real node is defined.")


class Tree:
    """Tree implementation for dimension hierarchy in Hierarchical Tucker decomposition.

    This class represents a tree structure used for organizing tensor dimensions
    in the hierarchical Tucker format. It manages a collection of nodes representing
    different dimension groupings in a hierarchical manner.

    Attributes:
        root: The root node of the tree.
        _depth: Maximum depth of the tree.
        _size: Total number of nodes in the tree.
        _leafCount: Number of leaf nodes in the tree.
        _nodeCount: Number of internal (non-leaf, non-root) nodes.
        _leaves: List of all leaf nodes.
        _level_items: List of lists containing nodes at each level.
    """

    def __init__(self) -> None:
        """Initialize an empty dimension tree."""
        self.root: Node | None = None
        self._depth: int = 0
        self._size: int = 0
        self._leafCount: int = 0
        self._nodeCount: int = 0
        self._leaves: list[Node] = []
        self._level_items: list[list[Node]] | None = None

    def findNode(self, node, key, check_propagation=False):
        """Find a node in the tree by its value.

        This method recursively searches for a node with the given value, starting
        from the specified node. When check_propagation is True, it will only return
        unpropagated nodes if they match the key.

        Args:
            node: Starting node for the search.
            key: Value to search for.
            check_propagation: Whether to check the propagation status (default: False).

        Returns:
            Node: The found node, or None if no matching node is found.
        """
        if (node is None) or (node.val == key):
            return node
        for child in node.children:
            return_node = self.findNode(child, key, check_propagation=check_propagation)
            if return_node:
                if check_propagation and (not return_node._propagated):
                    return return_node
                elif check_propagation and return_node._propagated:
                    pass
                else:
                    return return_node
        return None

    def isEmpty(self):
        """Check if the tree is empty.

        Returns:
            bool: True if the tree has no nodes, False otherwise.
        """
        return self._size == 0

    def initializeTree(self, vals):
        """Initialize the tree with a root node containing the given values.

        Args:
            vals: Values for the root node, must be a list.

        Raises:
            TypeError: If vals is not a list.
        """
        if self.root is None:
            if isinstance(vals, list):
                self.root = vals
            else:
                raise TypeError(f"Type: {type(vals)} is not known!")
        else:
            warn("Root node already implemented! Doing nothing.")

    def insertNode(self, val, parent=None, dim_index=None):
        """Insert a new node into the tree.

        This method creates a new node with the given value and attaches it to the
        specified parent. If no parent is specified, the new node becomes the root.

        Args:
            val: Value of the new node.
            parent: Parent node or parent key/dimensions (default: None).
            dim_index: Dimension indices associated with this node (default: None).

        Returns:
            Node: The newly created node.

        Raises:
            NotFoundError: If the specified parent cannot be found in the tree.
        """
        newNode = Node(val)
        newNode._dimension_index = dim_index
        if parent is None:  # No parent is given, i.e. Root node
            self.root = newNode
            self._depth = 1
            self._size = 1
            newNode._level = 0
            newNode.adjust_ranks()
        elif type(parent) is Node:  # Parent is given directly as a node object
            parent.children.append(newNode)
            parent._propagated = True
            # parent._ranks+=[None]
            self._size += 1
            newNode.parent = parent
            newNode._level = parent._level + 1
            parent.adjust_ranks()
        else:  # Key/dimensions of the parent is given as input
            parentNode = self.findNode(self.root, parent, check_propagation=True)
            if not (parentNode):
                raise NotFoundError(f"No parent was found for parent name: {parent}")
            parentNode.children.append(newNode)
            parentNode._propagated = True
            # parentNode._ranks+=[None]
            self._size += 1
            newNode.parent = parentNode
            newNode._level = parentNode._level + 1
            parentNode.adjust_ranks()
        if len(val) == 1:
            newNode._isleaf = True
            newNode.adjust_ranks()
            self._leaves.append(newNode)
            self._leafCount += 1

        return newNode

    def get_max_depth(self):
        """Calculate and store the maximum depth of the tree.

        This method calculates the maximum depth of the tree by finding the
        longest path from any leaf to the root node.

        Returns:
            None: The depth is stored in the _depth attribute.
        """
        self._depth = 0
        for leaf in self._leaves:
            depth = 0
            node = leaf
            while node.parent is not None:
                depth += 1
                node = node.parent
            if depth > self._depth:
                self._depth = depth
        return None

    def get_items_from_level(self):
        """Organize nodes by their level in the tree.

        This method populates the _level_items attribute with lists of nodes
        organized by their level in the tree (distance from root).
        """
        self._level_items = []
        for _ in range(self._depth + 1):
            self._level_items.append([])
        # for depth,items in enumerate(level_items):
        nodes2expand = [self.root]
        while nodes2expand:
            node = nodes2expand.pop(0)
            nodes2expand.extend(node.children)
            self._level_items[node._level].append(node)

    def toList(self) -> list:
        """Convert tree to list representation.

        Returns:
            list: Currently returns an empty list as this method is not fully implemented.
        """
        # Returns a list from the tree
        if not self.root:
            return []
        return []


def createDimensionTree(inp, numSplits, minSplitSize):
    """Create a dimension tree from input dimensions.

    This function creates a hierarchical tree structure that organizes tensor dimensions
    by recursively splitting them into groups. It is used to establish the structure
    for hierarchical Tucker decomposition.

    Args:
        inp: Input tensor (numpy.ndarray) or dimensions (tuple/list).
        numSplits: Number of splits to create at each level.
        minSplitSize: Minimum size of a dimension group to consider for splitting.

    Returns:
        Tree: A dimension tree with nodes representing dimension groups.

    Raises:
        TypeError: If inp is not a numpy array, tuple, or list.

    Notes:
        - FIXME: Dimension tree returns wrong case where dimension order repeats itself.
        - BUG: Searching just with node values may return wrong results
        - The current implementation may need improvement for better n-ary splits
    """
    if type(inp) is np.ndarray:
        dims = np.array(inp.shape)
    elif (type(inp) is tuple) or (type(inp) is list):
        dims = np.array(inp)  # NOQA
    else:
        raise TypeError(f"Type: {type(inp)} is unsupported!!")
    dimensionTree = Tree()
    dimensionTree.insertNode(dims.tolist())
    # print(np.array(dimensionTree.root.val))
    dimensionTree.root._dimension_index = [idx for idx, _ in enumerate(dimensionTree.root.val)]
    nodes2expand = []
    nodes2expand.append(dimensionTree.root.val.copy())
    while nodes2expand:
        # BUG: searching just with node values return wrong results
        # FIXME: change dimensions in nodes2expand to a tuple of (dimensions,parent node)
        #        to avoid confusion while searching in the dimension tree.
        #        Or maybe come up with a new createDimensionTree that suits n-ary splits better.
        # print(leaves)
        node2expand = nodes2expand.pop(0)
        node = dimensionTree.findNode(dimensionTree.root, node2expand, check_propagation=True)
        dim_split = np.array_split(np.array(node.val), numSplits)
        idx_split = np.array_split(np.array(node._dimension_index), numSplits)
        if (not node._propagated) and (len(node.val) > minSplitSize + 1):
            # for split in [data[x:x+10] for x in xrange(0, len(data), 10)]:
            for dims, indices in zip(dim_split, idx_split):  # place zip here
                # print(dims)
                # tree.insertNode(split,node.val)
                # leaves.append(split)
                # dimensionTree.insertNode(dims.tolist(), node.val,dim_index=indices.tolist())
                dimensionTree.insertNode(dims.tolist(), node, dim_index=indices.tolist())
                nodes2expand.append(dims.tolist())
        elif (not node._propagated) and (len(node.val) > minSplitSize):
            # i.e. the node is a leaf
            # print(node.val)
            for dims, indices in zip(dim_split, idx_split):  # place zip here
                # dimensionTree.insertNode(dims.tolist(), node.val, dim_index=indices.tolist())
                dimensionTree.insertNode(dims.tolist(), node, dim_index=indices.tolist())
    dimensionTree.get_max_depth()
    dimensionTree._nodeCount = (
        dimensionTree._size - dimensionTree._leafCount - 1
    )  # last -1 is to subtract root node
    return dimensionTree


# def createDimensionTree(inp, numSplits, minSplitSize):
#     """Create a dimension tree from input dimensions.

#     Args:
#         inp: Input tensor or dimensions
#         numSplits: Number of splits
#         minSplitSize: Minimum size of split

#     Returns:
#         Tree: Dimension tree
#     """
#     if isinstance(inp, np.ndarray):
#         dims = np.array(inp.shape)
#     elif isinstance(inp, (tuple, list)):
#         dims = np.array(inp)
#     else:
#         raise ValueError("Input must be a numpy array or a list/tuple of dimensions")

#     dimensionTree = Tree()
#     dimensionTree.insertNode(dims.tolist())
#     dimensionTree.root._dimension_index = [idx for idx, _ in enumerate(dimensionTree.root.val)]

#     nodes2expand = []
#     nodes2expand.append(dimensionTree.root)

#     while nodes2expand:
#         curr_node = nodes2expand.pop(0)
#         curr_dims = curr_node.val
#         curr_indices = curr_node._dimension_index

#         if len(curr_dims) <= minSplitSize:
#             continue

#         for split_idx in range(1, min(numSplits+1, len(curr_dims))):
#             split_size = len(curr_dims) // numSplits if numSplits > 0 else 1
#             if split_size < minSplitSize:
#                 continue

#             split_point = split_idx * split_size
#             if split_point == 0 or split_point >= len(curr_dims):
#                 continue

#             left_dims = curr_dims[:split_point]
#             right_dims = curr_dims[split_point:]

#             left_indices = curr_indices[:split_point]
#             right_indices = curr_indices[split_point:]

#             left_node = dimensionTree.insertNode(left_dims, curr_node, left_indices)
#             right_node = dimensionTree.insertNode(right_dims, curr_node, right_indices)

#             nodes2expand.append(left_node)
#             nodes2expand.append(right_node)

#     dimensionTree.get_max_depth()
#     dimensionTree._nodeCount = dimensionTree._size - dimensionTree._leafCount - 1
#     # -1 to subtract root node
#     return dimensionTree
