"""Tests for the tree.py module."""

import unittest

import numpy as np

from htucker.tree import Node, Tree, createDimensionTree


class TestNode(unittest.TestCase):
    """Test cases for the Node class."""

    def setUp(self):
        """Set up test fixture."""
        self.node = Node([0, 1])

    def test_init(self):
        """Test initialization of Node."""
        self.assertEqual(self.node.val, [0, 1])
        self.assertEqual(self.node.children, [])
        self.assertEqual(self.node.parent, None)
        self.assertEqual(self.node._isleaf, False)
        self.assertEqual(self.node.real_node, None)
        self.assertEqual(self.node.real_children, [])

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
        self.tree = Tree()
        self.tree.insertNode([0, 1, 2, 3])  # This creates the root

    def test_init(self):
        """Test initialization of Tree."""
        tree = Tree()
        self.assertIsNone(tree.root)
        self.assertEqual(tree._size, 0)
        self.assertEqual(tree._leafCount, 0)
        self.assertEqual(tree._nodeCount, 0)

    def test_insert_node(self):
        """Test inserting nodes to the tree."""
        # Root node was already created in setUp
        self.assertIsNotNone(self.tree.root)
        self.assertEqual(self.tree._size, 1)

        # Insert child nodes
        self.tree.insertNode([0, 1], self.tree.root)
        self.tree.insertNode([2, 3], self.tree.root)

        # Check size
        self.assertEqual(self.tree._size, 3)

        # Check parent-child relationships
        self.assertEqual(len(self.tree.root.children), 2)

    def test_insert_leaf(self):
        """Test inserting a leaf node."""
        # Insert a leaf node (single dimension)
        self.tree.insertNode([0], self.tree.root)

        # Should be added to leaves
        self.assertTrue(any(node._isleaf for node in self.tree.root.children))
        self.assertEqual(self.tree._leafCount, 1)

    def test_get_items_from_level(self):
        """Test getting items from each level."""
        # Create a simple tree with 3 levels
        parent1 = self.tree.insertNode([0, 1], self.tree.root)
        parent2 = self.tree.insertNode([2, 3], self.tree.root)

        self.tree.insertNode([0], parent1)
        self.tree.insertNode([1], parent1)
        self.tree.insertNode([2], parent2)
        self.tree.insertNode([3], parent2)

        # Get items from level
        self.tree.get_items_from_level()

        # Check level items exist
        self.assertIsNotNone(self.tree._level_items)


class TestCreateDimensionTree(unittest.TestCase):
    """Test cases for the createDimensionTree function."""

    def test_create_dimension_tree_with_tensor(self):
        """Test creating a dimension tree from a tensor."""
        tensor = np.random.rand(4, 6, 8, 10)
        tree = createDimensionTree(tensor, 2, 1)

        # Check that tree is created
        self.assertIsInstance(tree, Tree)

        # Check some basic properties
        self.assertIsNotNone(tree.root)
        self.assertEqual(tree.root.val, [4, 6, 8, 10])

    def test_create_dimension_tree_with_shape(self):
        """Test creating a dimension tree from a shape tuple."""
        shape = (4, 6, 8, 10)
        tree = createDimensionTree(shape, 2, 1)

        # Check that tree is created
        self.assertIsInstance(tree, Tree)

        # Check some basic properties
        self.assertIsNotNone(tree.root)
        self.assertEqual(tree.root.val, [4, 6, 8, 10])


if __name__ == "__main__":
    unittest.main()
