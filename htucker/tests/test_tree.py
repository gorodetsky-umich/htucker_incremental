"""Tests for the tree.py module."""

import unittest
import numpy as np
from htucker.tree import Node, Tree, createDimensionTree

class TestNode(unittest.TestCase):
    """Test cases for the Node class."""

    def setUp(self):
        """Set up test fixture."""
        self.node = Node([0, 1], 1)
        
    def test_init(self):
        """Test initialization of Node."""
        self.assertEqual(self.node._dimension_index, [0, 1])
        self.assertEqual(self.node.val, 1)
        self.assertEqual(self.node.children, [])
        self.assertEqual(self.node.parent, None)
        self.assertEqual(self.node._isleaf, False)
        self.assertEqual(self.node._level, 0)
        self.assertEqual(self.node.real_node, None)
        self.assertEqual(self.node.real_parent, None)
        self.assertEqual(self.node.real_children, [])

    def test_add_child(self):
        """Test adding a child to a node."""
        child = Node([2, 3], 2)
        self.node.add_child(child)
        
        self.assertEqual(len(self.node.children), 1)
        self.assertIs(self.node.children[0], child)
        self.assertIs(child.parent, self.node)
        
    def test_is_leaf(self):
        """Test is_leaf property."""
        # Initially not a leaf
        self.assertFalse(self.node._isleaf)
        
        # Set node as leaf
        self.node._isleaf = True
        self.assertTrue(self.node._isleaf)
        
class TestTree(unittest.TestCase):
    """Test cases for the Tree class."""

    def setUp(self):
        """Set up test fixture."""
        self.root = Node([0, 1, 2, 3], "root")
        self.tree = Tree(self.root)
        
    def test_init(self):
        """Test initialization of Tree."""
        self.assertIs(self.tree.root, self.root)
        self.assertEqual(self.tree._nodeCount, 1)
        self.assertEqual(self.tree.leaves, [])
        
    def test_add_node(self):
        """Test adding nodes to the tree."""
        node1 = Node([0, 1], "node1")
        node2 = Node([2, 3], "node2")
        
        # Add nodes as children of root
        self.tree.add_node(node1, self.root)
        self.tree.add_node(node2, self.root)
        
        # Check node count
        self.assertEqual(self.tree._nodeCount, 3)
        
        # Check parent-child relationships
        self.assertIs(node1.parent, self.root)
        self.assertIs(node2.parent, self.root)
        self.assertIn(node1, self.root.children)
        self.assertIn(node2, self.root.children)
        
    def test_add_leaf(self):
        """Test adding a leaf node."""
        leaf = Node([0], "leaf")
        leaf._isleaf = True
        
        # Add leaf
        self.tree.add_leaf(leaf)
        
        # Check if added to leaves list
        self.assertIn(leaf, self.tree.leaves)
        
    def test_get_items_from_level(self):
        """Test getting items from each level."""
        # Create a simple tree with 3 levels
        node1 = Node([0, 1], "level1_node1")
        node2 = Node([2, 3], "level1_node2")
        self.tree.add_node(node1, self.root)
        self.tree.add_node(node2, self.root)
        
        leaf1 = Node([0], "leaf1")
        leaf1._isleaf = True
        leaf2 = Node([1], "leaf2")
        leaf2._isleaf = True
        leaf3 = Node([2], "leaf3")
        leaf3._isleaf = True
        leaf4 = Node([3], "leaf4")
        leaf4._isleaf = True
        
        self.tree.add_node(leaf1, node1)
        self.tree.add_node(leaf2, node1)
        self.tree.add_node(leaf3, node2)
        self.tree.add_node(leaf4, node2)
        
        # Update leaf nodes
        for leaf in [leaf1, leaf2, leaf3, leaf4]:
            self.tree.add_leaf(leaf)
        
        # Get items from level
        self.tree.get_items_from_level()
        
        # Check level assignments
        self.assertEqual(self.root._level, 0)
        self.assertEqual(node1._level, 1)
        self.assertEqual(node2._level, 1)
        self.assertEqual(leaf1._level, 2)
        self.assertEqual(leaf2._level, 2)
        self.assertEqual(leaf3._level, 2)
        self.assertEqual(leaf4._level, 2)
        
        # Check level items
        self.assertEqual(len(self.tree._level_items), 3)  # 3 levels
        self.assertEqual(len(self.tree._level_items[0]), 1)  # 1 node at level 0
        self.assertEqual(len(self.tree._level_items[1]), 2)  # 2 nodes at level 1
        self.assertEqual(len(self.tree._level_items[2]), 4)  # 4 nodes at level 2

class TestCreateDimensionTree(unittest.TestCase):
    """Test cases for the createDimensionTree function."""

    def test_create_dimension_tree_with_tensor(self):
        """Test creating a dimension tree from a tensor."""
        tensor = np.random.rand(4, 6, 8, 10)
        tree = createDimensionTree(tensor, 2, 1)
        
        # Check that tree is created
        self.assertIsInstance(tree, Tree)
        
        # Check root node dimensions
        self.assertEqual(tree.root._dimension_index, [0, 1, 2, 3])
        
        # Check leaves
        self.assertEqual(len(tree.leaves), 4)  # One leaf per dimension
        
        # All leaves should have a single dimension index
        for i, leaf in enumerate(tree.leaves):
            self.assertEqual(len(leaf._dimension_index), 1)
            
    def test_create_dimension_tree_with_shape(self):
        """Test creating a dimension tree from a shape tuple."""
        shape = (4, 6, 8, 10)
        tree = createDimensionTree(shape, 2, 1)
        
        # Check that tree is created
        self.assertIsInstance(tree, Tree)
        
        # Check root node dimensions
        self.assertEqual(tree.root._dimension_index, [0, 1, 2, 3])
        
        # Check leaves
        self.assertEqual(len(tree.leaves), 4)  # One leaf per dimension
        
        # All leaves should have a single dimension index
        for i, leaf in enumerate(tree.leaves):
            self.assertEqual(len(leaf._dimension_index), 1)

if __name__ == '__main__':
    unittest.main()
