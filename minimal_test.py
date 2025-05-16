#!/usr/bin/env python3
"""Minimal HTucker test"""

import numpy as np
import htucker as ht
import sys
import traceback

# Configure output to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

try:
    print(f"HTucker version: {ht.__version__}")
    
    # Create a small tensor
    tensor = np.random.rand(5, 4, 3)
    print(f"Created tensor with shape {tensor.shape}")
    
    # Create HTucker object
    htd = ht.HTucker()
    htd.initialize(tensor)
    print("Initialized HTucker object")

    # Compress
    try:
        htd.compress_root2leaf(tensor)
        print("Compressed tensor")
    except Exception as e:
        print(f"Compression error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Reconstruct
    try:
        reconstructed = htd.reconstruct_all()
        print(f"Reconstructed tensor with shape {reconstructed.shape}")
    except Exception as e:
        print(f"Reconstruction error: {e}")
        traceback.print_exc()
        sys.exit(2)

    # Error
    rel_error = np.linalg.norm(reconstructed - tensor) / np.linalg.norm(tensor)
    print(f"Relative error: {rel_error:.8e}")

    print("Test completed successfully!")
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()
