"""
HTucker Incremental Update Example

This script demonstrates the incremental update functionality of the HTucker package,
which is useful for streaming data and online tensor decomposition.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import htucker as ht

# Set random seed for reproducibility
np.random.seed(42)


def main():
    # Create an initial random tensor
    print("Creating initial tensor...")
    tensor_shape = (10, 8, 6, 4)
    initial_tensor = np.random.rand(*tensor_shape)

    print(f"Tensor shape: {initial_tensor.shape}")

    # Create dimension tree
    print("\nCreating dimension tree...")
    dim_tree = ht.createDimensionTree(initial_tensor, 2, 1)

    # Initialize HTucker object with dimension tree
    print("\nInitializing HTucker with dimension tree...")
    htd = ht.HTucker()
    htd.initialize(initial_tensor, dimension_tree=dim_tree)
    htd.rtol = 1e-6  # Relative tolerance for SVD truncation

    # Compress using leaf-to-root approach
    print("\nCompressing initial tensor using leaf-to-root approach...")
    start_time = time.time()
    htd.compress_leaf2root(initial_tensor, dim_tree)
    compress_time = time.time() - start_time
    print(f"Initial compression time: {compress_time:.4f} seconds")

    # Reconstruct and check error
    reconstructed = htd.reconstruct_all()
    rel_error = np.linalg.norm(reconstructed - initial_tensor) / np.linalg.norm(initial_tensor)
    print(f"Initial relative reconstruction error: {rel_error:.8e}")

    # Store ranks for comparison
    initial_ranks = htd.get_ranks()
    print(f"Initial ranks: {initial_ranks}")

    # Now create a new tensor to simulate streaming data
    print("\nCreating new tensor for incremental update...")

    # Let's create a slightly different pattern in the new tensor
    new_tensor = np.random.rand(*tensor_shape)
    # Add some structure to make it different from the initial tensor
    for i in range(tensor_shape[0]):
        for j in range(tensor_shape[1]):
            new_tensor[i, j, :, :] += 0.5 * np.sin(i / 2) * np.cos(j / 3)

    # Perform incremental update
    print("\nPerforming incremental update...")
    start_time = time.time()
    htd.incremental_update(new_tensor)
    update_time = time.time() - start_time
    print(f"Incremental update time: {update_time:.4f} seconds")

    # Check reconstruction quality on new tensor
    reconstructed_new = htd.reconstruct(htd.project(new_tensor))
    rel_error_new = np.linalg.norm(reconstructed_new - new_tensor) / np.linalg.norm(new_tensor)
    print(f"New tensor relative reconstruction error: {rel_error_new:.8e}")

    # Check updated ranks
    updated_ranks = htd.get_ranks()
    print(f"Updated ranks: {updated_ranks}")

    # Now try reconstructing original tensor after update
    reconstructed_original = htd.reconstruct(htd.project(initial_tensor))
    rel_error_original = np.linalg.norm(reconstructed_original - initial_tensor) / np.linalg.norm(
        initial_tensor
    )
    print(f"Original tensor relative reconstruction error after update: {rel_error_original:.8e}")

    # Performance comparison
    print("\nPerformance comparison:")
    print(f"Initial compression time: {compress_time:.4f} seconds")
    print(f"Incremental update time: {update_time:.4f} seconds")
    print(f"Speed-up ratio: {compress_time/update_time:.2f}x")

    # Plot comparison of original, new and reconstructed tensors
    slice_idx = 0
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(initial_tensor[slice_idx, :, :, 0], cmap="viridis")
    plt.title("Original Tensor Slice")
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(reconstructed_original[slice_idx, :, :, 0], cmap="viridis")
    plt.title("Reconstructed Original")
    plt.colorbar()

    plt.subplot(2, 3, 3)
    diff_original = initial_tensor[slice_idx, :, :, 0] - reconstructed_original[slice_idx, :, :, 0]
    plt.imshow(diff_original, cmap="RdBu_r")
    plt.title(f"Original Difference (Max: {np.max(np.abs(diff_original)):.2e})")
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.imshow(new_tensor[slice_idx, :, :, 0], cmap="viridis")
    plt.title("New Tensor Slice")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(reconstructed_new[slice_idx, :, :, 0], cmap="viridis")
    plt.title("Reconstructed New")
    plt.colorbar()

    plt.subplot(2, 3, 6)
    diff_new = new_tensor[slice_idx, :, :, 0] - reconstructed_new[slice_idx, :, :, 0]
    plt.imshow(diff_new, cmap="RdBu_r")
    plt.title(f"New Difference (Max: {np.max(np.abs(diff_new)):.2e})")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("htd_incremental_update_results.png")
    plt.close()

    print("\nResults visualization saved as 'htd_incremental_update_results.png'")


if __name__ == "__main__":
    main()
