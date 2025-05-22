#!/usr/bin/env python3
"""
HTucker Package Test Script

This script tests the basic functionality of the HTucker package.
"""

import numpy as np
import htucker as ht
import time
import argparse

def main():
    """Run a simple test of the HTucker package."""
    parser = argparse.ArgumentParser(description='Test HTucker package functionality')
    parser.add_argument('--dim', type=int, default=4, 
                        help='Number of tensor dimensions (default: 4)')
    parser.add_argument('--size', type=int, default=10, 
                        help='Size of each dimension (default: 10)')
    parser.add_argument('--tol', type=float, default=1e-6, 
                        help='Relative tolerance for compression (default: 1e-6)')
    args = parser.parse_args()

    print(f"HTucker Package Test (version {ht.__version__})")
    print("=" * 50)
    
    # Create a random tensor
    dims = [args.size] * args.dim
    print(f"Creating random tensor with shape {dims}...")
    tensor = np.random.rand(*dims)
    
    print(f"Tensor size: {tensor.size} elements")
    print(f"Memory usage: {tensor.nbytes / 1024:.2f} KB")
    
    # Create dimension tree
    print("\nCreating dimension tree...")
    start_time = time.time()
    dim_tree = ht.createDimensionTree(tensor, 2, 1)
    tree_time = time.time() - start_time
    print(f"Tree creation time: {tree_time:.4f} seconds")
    print(f"Tree has {len(dim_tree.leaves)} leaves")

    # Test root-to-leaf compression
    print("\nTesting root-to-leaf compression...")
    htd_r2l = ht.HTucker()
    htd_r2l.initialize(tensor)
    
    start_time = time.time()
    htd_r2l.compress_root2leaf(tensor)
    r2l_time = time.time() - start_time
    print(f"Compression time: {r2l_time:.4f} seconds")
    
    # Reconstruct and check error
    start_time = time.time()
    reconstr_r2l = htd_r2l.reconstruct_all()
    rec_time = time.time() - start_time
    print(f"Reconstruction time: {rec_time:.4f} seconds")
    
    # Calculate error
    rel_error = np.linalg.norm(reconstr_r2l - tensor) / np.linalg.norm(tensor)
    print(f"Relative error: {rel_error:.8e}")
    
    # Memory statistics
    memory_size = htd_r2l.get_memory_size()
    compression_ratio = tensor.nbytes / memory_size if memory_size > 0 else 0
    print(f"Compressed memory size: {memory_size / 1024:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Try to get the ranks (if the function is implemented)
    try:
        ranks = htd_r2l.get_ranks()
        if ranks:
            print("\nHT ranks:")
            for node, rank in ranks.items():
                print(f"  {node}: {rank}")
    except (AttributeError, NotImplementedError):
        print("\nRank information not available")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
