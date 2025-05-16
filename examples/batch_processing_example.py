"""
HTucker Batch Processing Example

This script demonstrates the batch processing functionality of the HTucker package,
which is useful for handling tensors with a batch dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import htucker as ht

# Set random seed for reproducibility
np.random.seed(42)

def main():
    # Create a tensor with a batch dimension
    print("Creating batch tensor...")
    base_shape = (10, 8, 6)  # Base tensor shape
    batch_size = 5          # Number of batch samples
    
    # Create a tensor with batch dimension at the end
    batch_tensor = np.random.rand(*base_shape, batch_size)
    
    print(f"Batch tensor shape: {batch_tensor.shape}")
    print(f"Batch dimension: {len(batch_tensor.shape) - 1} (last dimension)")
    
    # Create dimension tree for the base tensor shape (excluding batch dimension)
    print("\nCreating dimension tree...")
    dim_tree = ht.createDimensionTree(base_shape, 2, 1)
    
    # Initialize HTucker object with batch awareness
    print("\nInitializing HTucker for batch processing...")
    htd = ht.HTucker()
    batch_dimension = len(batch_tensor.shape) - 1  # Last dimension is batch
    htd.initialize(batch_tensor, dimension_tree=dim_tree, 
                  batch=True, batch_dimension=batch_dimension)
    htd.rtol = 1e-6  # Relative tolerance for SVD truncation
    
    # Compress using leaf-to-root batch approach
    print("\nCompressing batch tensor...")
    start_time = time.time()
    htd.compress_leaf2root_batch(batch_tensor, dim_tree, batch_dimension=batch_dimension)
    compress_time = time.time() - start_time
    print(f"Batch compression time: {compress_time:.4f} seconds")
    
    # Verify compression
    print(f"Batch count in HTucker: {htd.batch_count}")
    print(f"Expected batch count: {batch_size}")
    
    # Project and reconstruct individual samples from the batch
    print("\nProjecting and reconstructing individual batch samples...")
    
    # Store reconstruction errors for all batch samples
    reconstruction_errors = []
    
    # Visualization setup
    plt.figure(figsize=(15, 10))
    
    # Process each batch sample
    for batch_idx in range(batch_size):
        print(f"\nProcessing batch sample {batch_idx}...")
        
        # Extract this batch sample as a tensor
        # Note: Need to keep dimension for proper indexing
        sample_tensor = batch_tensor[..., batch_idx:batch_idx+1]
        
        # Project the sample tensor
        start_time = time.time()
        projected = htd.project(sample_tensor, batch=True, batch_dimension=batch_dimension)
        project_time = time.time() - start_time
        
        # Reconstruct from the projection
        start_time = time.time()
        reconstructed = htd.reconstruct(projected, batch=True, batch_dimension=batch_dimension)
        reconstruct_time = time.time() - start_time
        
        # Remove singleton batch dimension for comparison
        reconstructed = reconstructed[..., 0]
        sample_tensor = sample_tensor[..., 0]
        
        # Calculate reconstruction error
        rel_error = np.linalg.norm(reconstructed - sample_tensor) / np.linalg.norm(sample_tensor)
        reconstruction_errors.append(rel_error)
        
        print(f"Sample {batch_idx} relative error: {rel_error:.8e}")
        print(f"Project time: {project_time:.4f}s, Reconstruct time: {reconstruct_time:.4f}s")
        
        # Visualize the results for this batch sample
        # We'll show a 2D slice of the tensor
        slice_idx = 0
        
        # Plot original
        plt.subplot(batch_size, 3, 3*batch_idx + 1)
        plt.imshow(sample_tensor[slice_idx, :, :], cmap='viridis')
        plt.title(f'Batch {batch_idx} Original')
        plt.colorbar()
        
        # Plot reconstructed
        plt.subplot(batch_size, 3, 3*batch_idx + 2)
        plt.imshow(reconstructed[slice_idx, :, :], cmap='viridis')
        plt.title(f'Batch {batch_idx} Reconstructed')
        plt.colorbar()
        
        # Plot difference
        plt.subplot(batch_size, 3, 3*batch_idx + 3)
        diff = sample_tensor[slice_idx, :, :] - reconstructed[slice_idx, :, :]
        plt.imshow(diff, cmap='RdBu_r')
        plt.title(f'Diff (Max: {np.max(np.abs(diff)):.2e})')
        plt.colorbar()
    
    # Overall statistics
    print("\nBatch processing summary:")
    print(f"Mean reconstruction error: {np.mean(reconstruction_errors):.8e}")
    print(f"Max reconstruction error: {np.max(reconstruction_errors):.8e}")
    print(f"Min reconstruction error: {np.min(reconstruction_errors):.8e}")
    
    # Save visualization
    plt.tight_layout()
    plt.savefig('htd_batch_processing_results.png')
    plt.close()
    
    print("\nResults visualization saved as 'htd_batch_processing_results.png'")
    
if __name__ == "__main__":
    main()
