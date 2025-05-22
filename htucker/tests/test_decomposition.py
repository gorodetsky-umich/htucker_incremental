"""Tests for the decomposition.py module."""

import unittest

import numpy as np

from htucker.decomposition import hosvd, hosvd_only_for_dimensions, truncated_svd


class TestDecomposition(unittest.TestCase):
    """Test cases for the decomposition functions."""

    def setUp(self):
        """Set up test fixture."""
        # Create a sample tensor for testing
        self.tensor_4d = np.random.rand(5, 4, 3, 2)
        self.tol = 1e-10

    def test_truncated_svd(self):
        """Test truncated_svd function."""
        # Test with a matrix
        matrix = np.random.rand(10, 8)
        u, s, v = truncated_svd(matrix, self.tol)

        # Verify dimensions
        self.assertEqual(u.shape[0], matrix.shape[0])
        self.assertEqual(v.shape[1], matrix.shape[1])

        # Verify reconstruction
        reconstructed = u @ np.diag(s) @ v
        rel_error = np.linalg.norm(reconstructed - matrix) / np.linalg.norm(matrix)
        self.assertLess(rel_error, self.tol * 10)  # Allow for numerical errors

    def test_hosvd(self):
        """Test hosvd function."""
        # Apply HOSVD
        core, matrices = hosvd(self.tensor_4d)

        # Check core and matrices dimensions
        self.assertEqual(len(core.shape), len(self.tensor_4d.shape))
        self.assertEqual(len(matrices), len(self.tensor_4d.shape))

        # Reconstruct tensor using einsum
        einsum_str = ""
        core_indices = ""
        matrices_indices = []

        # Build einsum string for reconstruction
        for i in range(len(self.tensor_4d.shape)):
            core_indices += chr(97 + i)
            matrices_indices.append(chr(97 + len(self.tensor_4d.shape) + i) + chr(97 + i))

        einsum_str = (
            ",".join([core_indices] + matrices_indices)
            + "->"
            + "".join(
                [chr(97 + len(self.tensor_4d.shape) + i) for i in range(len(self.tensor_4d.shape))]
            )
        )

        args = [core] + matrices
        reconstructed = np.einsum(einsum_str, *args)

        # Check reconstruction error
        rel_error = np.linalg.norm(reconstructed - self.tensor_4d) / np.linalg.norm(self.tensor_4d)
        self.assertLess(rel_error, 1e-10)  # HOSVD should give exact reconstruction

    def test_hosvd_only_for_dimensions(self):
        """Test hosvd_only_for_dimensions function."""
        # Select dimensions to decompose
        dims = [0, 2]

        # Apply selective HOSVD
        matrices = hosvd_only_for_dimensions(
            self.tensor_4d, tol=self.tol, dims=dims, contract=False
        )

        # Check that we get matrices for selected dimensions
        self.assertEqual(len(matrices), len(dims))

        # Check matrices dimensions
        for i, idx in enumerate(dims):
            self.assertEqual(matrices[i].shape[0], self.tensor_4d.shape[idx])

        # Test with contract=True
        result = hosvd_only_for_dimensions(self.tensor_4d, tol=self.tol, dims=dims, contract=True)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
