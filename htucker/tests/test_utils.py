"""Tests for the utils.py module."""

import unittest

import numpy as np

from htucker.utils import (
    NotFoundError,
    convert_to_base2,
    create_permutations,
    mode_n_product,
    mode_n_unfolding,
    split_dimensions,
)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self):
        """Set up test fixture."""
        # Create a sample tensor for testing
        self.tensor_3d = np.array(
            [
                [[1, 13], [4, 16], [7, 19], [10, 22]],
                [[2, 14], [5, 17], [8, 20], [11, 23]],
                [[3, 15], [6, 18], [9, 21], [12, 24]],
            ]
        )

        # Expected mode-n unfoldings for the test tensor (based on Kolda's definition)
        self.mode0_unfolding = np.array(
            [
                [1, 4, 7, 10, 13, 16, 19, 22],
                [2, 5, 8, 11, 14, 17, 20, 23],
                [3, 6, 9, 12, 15, 18, 21, 24],
            ]
        )

        self.mode1_unfolding = np.array(
            [
                [1, 2, 3, 13, 14, 15],
                [4, 5, 6, 16, 17, 18],
                [7, 8, 9, 19, 20, 21],
                [10, 11, 12, 22, 23, 24],
            ]
        )

        self.mode2_unfolding = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            ]
        )

    def test_mode_n_unfolding(self):
        """Test mode-n unfolding function."""
        # Test mode-0 unfolding
        mode0 = mode_n_unfolding(self.tensor_3d, 0)
        self.assertTrue(np.allclose(mode0, self.mode0_unfolding))

        # Test mode-1 unfolding
        mode1 = mode_n_unfolding(self.tensor_3d, 1)
        self.assertTrue(np.allclose(mode1, self.mode1_unfolding))

        # Test mode-2 unfolding
        mode2 = mode_n_unfolding(self.tensor_3d, 2)
        self.assertTrue(np.allclose(mode2, self.mode2_unfolding))

    def test_mode_n_product(self):
        """Test mode-n product function."""
        # Create a matrix for the n-mode product
        matrix = np.random.rand(5, self.tensor_3d.shape[1])

        # Compute n-mode product
        result = mode_n_product(self.tensor_3d, matrix, [1, 1])

        # Verify result dimensions
        expected_shape = list(self.tensor_3d.shape)
        expected_shape[1] = matrix.shape[0]
        self.assertEqual(result.shape, tuple(expected_shape))

    def test_split_dimensions(self):
        """Test split_dimensions function for basic splitting."""
        # Test with a list (original functionality)
        dims = [1, 2, 3, 4, 5]
        split1, split2 = split_dimensions(dims)

        # Verify the splits
        self.assertEqual(split1, [1, 2, 3])
        self.assertEqual(split2, [4, 5])

    def test_create_permutations(self):
        """Test create_permutations function."""
        # Test with a simple list
        result = create_permutations([1, 2, 3])

        # Verify permutations structure
        self.assertEqual(len(result), 3)  # Should create 3 permutations with 3 elements

        # Check the permutations generated
        self.assertIn([1, 2, 3], result)
        self.assertIn([2, 1, 3], result)
        self.assertIn([3, 1, 2], result)

    def test_convert_to_base2(self):
        """Test convert_to_base2 function."""
        result = convert_to_base2(10, 4)
        self.assertEqual(result, [1, 0, 1, 0])

        result = convert_to_base2(15, 4)
        self.assertEqual(result, [1, 1, 1, 1])

        result = convert_to_base2(7, 3)
        self.assertEqual(result, [1, 1, 1])

    def test_notfound_error(self):
        """Test NotFoundError exception."""
        with self.assertRaises(NotFoundError):
            raise NotFoundError("Test error message")


if __name__ == "__main__":
    unittest.main()
