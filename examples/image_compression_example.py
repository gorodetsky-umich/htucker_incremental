"""
HTucker Image Compression Example

This script demonstrates how to use HTucker for compressing and reconstructing image data,
showing its application in real-world scenarios.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import htucker as ht


def load_image(image_path):
    """Load an image and convert it to a numpy array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path)
    img_array = np.array(img)

    return img_array


def main():
    # Check if sample image exists, otherwise create a synthetic test image
    image_path = "sample_image.jpg"

    if os.path.exists(image_path):
        print(f"Loading image: {image_path}")
        img_array = load_image(image_path)
    else:
        print("Sample image not found, creating synthetic test image...")
        # Create a synthetic test image with some patterns
        img_size = (512, 512, 3)  # RGB image
        x = np.linspace(0, 1, img_size[0])
        y = np.linspace(0, 1, img_size[1])
        xx, yy = np.meshgrid(x, y)

        # Create RGB channels with different patterns
        r_channel = np.sin(10 * xx) * np.cos(10 * yy)
        g_channel = np.sin(20 * xx * yy)
        b_channel = np.cos(15 * xx + 15 * yy)

        # Normalize to [0, 1] range
        r_channel = (r_channel - r_channel.min()) / (r_channel.max() - r_channel.min())
        g_channel = (g_channel - g_channel.min()) / (g_channel.max() - g_channel.min())
        b_channel = (b_channel - b_channel.min()) / (b_channel.max() - b_channel.min())

        # Combine channels
        img_array = np.stack([r_channel, g_channel, b_channel], axis=2)
        img_array = (img_array * 255).astype(np.uint8)

        # Save the synthetic image
        Image.fromarray(img_array).save("synthetic_image.jpg")
        image_path = "synthetic_image.jpg"
        print(f"Saved synthetic test image as {image_path}")

    # Print image information
    print(f"Image shape: {img_array.shape}")

    # Convert to float for numerical stability in decomposition
    img_tensor = img_array.astype(np.float32) / 255.0

    # Create dimension tree
    print("\nCreating dimension tree...")
    dim_tree = ht.createDimensionTree(img_tensor, 2, 1)

    # Compression with different tolerance levels
    tolerances = [1e-1, 1e-2, 1e-3, 1e-4]
    results = []

    plt.figure(figsize=(15, 12))
    plt.subplot(len(tolerances) + 1, 3, 1)
    plt.imshow(img_array)
    plt.title("Original Image")
    plt.axis("off")

    # Plot information placeholder
    plt.subplot(len(tolerances) + 1, 3, 2)
    plt.text(
        0.5,
        0.5,
        f"Original Size: {img_array.nbytes / 1024:.1f} KB",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
    )
    plt.axis("off")

    for i, tol in enumerate(tolerances):
        print(f"\nCompressing with tolerance: {tol}")

        # Initialize HTucker object
        htd = ht.HTucker()
        htd.initialize(img_tensor, dimension_tree=dim_tree)
        htd.rtol = tol

        # Compress using leaf-to-root approach
        start_time = time.time()
        htd.compress_leaf2root(img_tensor, dim_tree)
        compress_time = time.time() - start_time

        # Reconstruct
        start_time = time.time()
        reconstructed = htd.reconstruct_all()
        reconstruct_time = time.time() - start_time

        # Convert back to uint8 for display
        reconstructed_img = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)

        # Calculate error metrics
        mse = np.mean((img_tensor - reconstructed) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

        # Calculate compressed size (approximate)
        htd_memory = htd.get_memory_size()
        compression_ratio = img_array.nbytes / htd_memory

        # Store results
        results.append(
            {
                "tolerance": tol,
                "compression_ratio": compression_ratio,
                "psnr": psnr,
                "compress_time": compress_time,
                "reconstruct_time": reconstruct_time,
                "memory_kb": htd_memory / 1024,
                "ranks": htd.get_ranks(),
            }
        )

        # Plot reconstructed image
        plt.subplot(len(tolerances) + 1, 3, (i + 1) * 3 + 1)
        plt.imshow(reconstructed_img)
        plt.title(f"Reconstructed (tol={tol})")
        plt.axis("off")

        # Plot compression info
        plt.subplot(len(tolerances) + 1, 3, (i + 1) * 3 + 2)
        info_text = (
            f"Tolerance: {tol}\n"
            f"Compression Ratio: {compression_ratio:.1f}x\n"
            f"PSNR: {psnr:.2f} dB\n"
            f"Size: {htd_memory / 1024:.1f} KB\n"
            f"Time: {compress_time:.2f}s"
        )
        plt.text(
            0.5,
            0.5,
            info_text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
        )
        plt.axis("off")

        # Plot error map
        plt.subplot(len(tolerances) + 1, 3, (i + 1) * 3 + 3)
        error = np.abs(img_tensor - reconstructed)
        error_img = np.mean(error, axis=2)  # Average error across color channels
        plt.imshow(error_img, cmap="hot")
        plt.title(f"Error Map (max={np.max(error):.4f})")
        plt.colorbar()
        plt.axis("off")

    # Print summary table
    print("\nCompression Results Summary:")
    print("-" * 80)
    print(
        f"{'Tolerance':<10} {'Comp. Ratio':<12} {'PSNR (dB)':<12} \
            {'Size (KB)':<12} {'Time (s)':<12}"
    )
    print("-" * 80)
    for res in results:
        print(
            f"{res['tolerance']:<10.1e} {res['compression_ratio']:<12.2f} {res['psnr']:<12.2f} "
            f"{res['memory_kb']:<12.2f} {res['compress_time']:<12.2f}"
        )

    # Save visualization
    plt.tight_layout()
    plt.savefig("htd_image_compression_results.png")
    plt.close()

    print("\nResults visualization saved as 'htd_image_compression_results.png'")

    # Plot compression ratio vs quality
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot([r["compression_ratio"] for r in results], [r["psnr"] for r in results], "o-")
    plt.xlabel("Compression Ratio")
    plt.ylabel("PSNR (dB)")
    plt.title("Quality vs Compression")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([r["tolerance"] for r in results], [r["compression_ratio"] for r in results], "o-")
    plt.xlabel("Tolerance")
    plt.xscale("log")
    plt.ylabel("Compression Ratio")
    plt.title("Tolerance vs Compression")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("htd_compression_analysis.png")
    plt.close()

    print("Compression analysis saved as 'htd_compression_analysis.png'")


if __name__ == "__main__":
    main()
