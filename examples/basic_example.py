"""
HTucker Basic Example

This script demonstrates the basic usage of the HTucker package for 
tensor decomposition using Hierarchical Tucker format.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import htucker as ht

# Set random seed for reproducibility
np.random.seed(42)

def main():
    # Create a random tensor
    print("Creating test tensor...")
    tensor_shape = (10, 8, 6, 4)
    tensor = np.random.rand(*tensor_shape)
    
    # Print tensor information
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor size: {tensor.size} elements")
    print(f"Memory usage: {tensor.nbytes / 1024:.2f} KB")
    
    # Create HTucker object
    print("\nInitializing HTucker object...")
    htd = ht.HTucker()
    htd.initialize(tensor)
    
    # Compress using root-to-leaf approach
    print("\nCompressing using root-to-leaf approach...")
    start_time = time.time()
    htd.compress_root2leaf(tensor)
    r2l_time = time.time() - start_time
    print(f"Compression time: {r2l_time:.4f} seconds")
    
    # Reconstruct and check error
    print("\nReconstructing tensor...")
    reconstructed = htd.reconstruct_all()
    rel_error = np.linalg.norm(reconstructed - tensor) / np.linalg.norm(tensor)
    print(f"Relative reconstruction error: {rel_error:.8e}")
    
    # Now try leaf-to-root compression with dimension tree
    print("\nCreating dimension tree...")
    dim_tree = ht.createDimensionTree(tensor, 2, 1)
    print(f"Dimension tree created with {dim_tree._leafCount} leaves")
    
    # Print dimension tree structure
    print("Dimension tree structure:")
    for node in dim_tree.nodes:
        if node.parent is not None:
            parent_dims = node.parent.dimensions
        else:
            parent_dims = "None (root)"
        print(f"Node dimensions: {node.dimensions}, Parent: {parent_dims}")
    
    # Initialize HTucker with dimension tree
    print("\nInitializing HTucker with dimension tree...")
    htd = ht.HTucker()
    htd.initialize(tensor, dimension_tree=dim_tree)
    
    # Set relative tolerance for compression
    htd.rtol = 1e-6
    print(f"Setting relative tolerance: {htd.rtol}")
    
    # Compress using leaf-to-root approach
    print("\nCompressing using leaf-to-root approach...")
    start_time = time.time()
    htd.compress_leaf2root(tensor, dim_tree)
    l2r_time = time.time() - start_time
    print(f"Compression time: {l2r_time:.4f} seconds")
    
    # Reconstruct and check error
    print("\nReconstructing tensor...")
    reconstructed = htd.reconstruct_all()
    rel_error = np.linalg.norm(reconstructed - tensor) / np.linalg.norm(tensor)
    print(f"Relative reconstruction error: {rel_error:.8e}")
    
    # Memory savings
    htd_memory = htd.get_memory_size()
    original_memory = tensor.nbytes
    compression_ratio = original_memory / htd_memory
    print(f"\nCompression Ratio: {compression_ratio:.2f}x")
    print(f"Original tensor: {original_memory / 1024:.2f} KB")
    print(f"Compressed representation: {htd_memory / 1024:.2f} KB")
    
    # Plot comparison of original vs reconstructed tensor slice
    slice_idx = 0
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(tensor[slice_idx, :, :, 0], cmap='viridis')
    plt.title('Original Tensor Slice')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed[slice_idx, :, :, 0], cmap='viridis')
    plt.title('Reconstructed Tensor Slice')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    diff = tensor[slice_idx, :, :, 0] - reconstructed[slice_idx, :, :, 0]
    plt.imshow(diff, cmap='RdBu_r')
    plt.title(f'Difference (Max: {np.max(np.abs(diff)):.2e})')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('htd_compression_results.png')
    plt.close()
    
    print("\nResults visualization saved as 'htd_compression_results.png'")
    
if __name__ == "__main__":
    main()
