"""
Edge prediction pretext task.

Uses Canny/Sobel edge detection to generate automatic labels.
This teaches the encoder to detect boundaries across different art styles,
which directly transfers to wall detection.

Inspired by auto-wall (ThreeHats/auto-wall) which uses Canny edge detection
as a primary wall detection method.
"""

import cv2
import numpy as np
import torch


def generate_edge_labels(image, low_threshold=50, high_threshold=150, blur_kernel=5):
    """Generate Canny edge labels for an image.

    Args:
        image: RGB image as numpy array (H, W, 3) with values 0-255
               or torch tensor (C, H, W) with values 0-1
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold
        blur_kernel: Gaussian blur kernel size (0 to disable)

    Returns:
        Binary edge mask as numpy array (H, W) with values 0.0-1.0
    """
    # Handle torch tensor input
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            # (C, H, W) -> (H, W, C)
            image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Optional blur to reduce noise
    if blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Normalize to 0-1
    return edges.astype(np.float32) / 255.0


def generate_edge_labels_multi_scale(image, scales=[(30, 100), (50, 150), (100, 200)]):
    """Generate multi-scale edge labels by combining multiple Canny thresholds.

    This captures both fine and coarse edges, similar to how walls can be
    drawn with different line weights.

    Args:
        image: RGB image as numpy array (H, W, 3)
        scales: List of (low_threshold, high_threshold) pairs

    Returns:
        Combined edge mask as numpy array (H, W) with values 0.0-1.0
    """
    # Handle torch tensor input
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    # Convert to grayscale once
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Combine edges at different scales
    combined = np.zeros(gray.shape, dtype=np.float32)

    for low, high in scales:
        edges = cv2.Canny(gray, low, high)
        combined = np.maximum(combined, edges.astype(np.float32))

    return combined / 255.0


def generate_edge_labels_sobel(image, ksize=3):
    """Generate edge labels using Sobel operator.

    Alternative to Canny, produces gradient magnitude which
    gives softer edges (useful as data augmentation).

    Args:
        image: RGB image as numpy array (H, W, 3)
        ksize: Sobel kernel size (1, 3, 5, or 7)

    Returns:
        Edge magnitude as numpy array (H, W) with values 0.0-1.0
    """
    # Handle torch tensor input
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to 0-1
    magnitude = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude

    return magnitude.astype(np.float32)


def batch_generate_edge_labels(images, method='canny', **kwargs):
    """Generate edge labels for a batch of images.

    Args:
        images: Batch of images as torch tensor (B, C, H, W) with values 0-1
        method: 'canny', 'multi_scale', or 'sobel'
        **kwargs: Arguments passed to the edge detection function

    Returns:
        Batch of edge labels as torch tensor (B, 1, H, W)
    """
    batch_size = images.shape[0]
    height, width = images.shape[2], images.shape[3]

    edges = torch.zeros(batch_size, 1, height, width, dtype=torch.float32)

    for i in range(batch_size):
        img = images[i]

        if method == 'canny':
            edge = generate_edge_labels(img, **kwargs)
        elif method == 'multi_scale':
            edge = generate_edge_labels_multi_scale(img, **kwargs)
        elif method == 'sobel':
            edge = generate_edge_labels_sobel(img, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        edges[i, 0] = torch.from_numpy(edge)

    return edges


class EdgePredictionTask:
    """Wrapper class for edge prediction pretext task.

    Handles label generation and loss computation.
    """

    def __init__(self, method='canny', low_threshold=50, high_threshold=150,
                 multi_scale_thresholds=None, random_thresholds=False):
        """
        Args:
            method: 'canny', 'multi_scale', or 'sobel'
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold
            multi_scale_thresholds: Thresholds for multi-scale method
            random_thresholds: If True, randomly vary thresholds for augmentation
        """
        self.method = method
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.multi_scale_thresholds = multi_scale_thresholds or [(30, 100), (50, 150), (100, 200)]
        self.random_thresholds = random_thresholds

    def generate_labels(self, images):
        """Generate edge labels for a batch.

        Args:
            images: Batch of images (B, C, H, W)

        Returns:
            Edge labels (B, 1, H, W)
        """
        if self.random_thresholds and self.method == 'canny':
            # Randomly vary thresholds for augmentation
            low = self.low_threshold + np.random.randint(-20, 20)
            high = self.high_threshold + np.random.randint(-30, 30)
            low = max(10, low)
            high = max(low + 20, high)
            return batch_generate_edge_labels(images, method='canny',
                                             low_threshold=low, high_threshold=high)

        elif self.method == 'canny':
            return batch_generate_edge_labels(images, method='canny',
                                             low_threshold=self.low_threshold,
                                             high_threshold=self.high_threshold)
        elif self.method == 'multi_scale':
            return batch_generate_edge_labels(images, method='multi_scale',
                                             scales=self.multi_scale_thresholds)
        elif self.method == 'sobel':
            return batch_generate_edge_labels(images, method='sobel')
        else:
            raise ValueError(f"Unknown method: {self.method}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create test image with some edges
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[50:200, 50:200] = [200, 180, 150]  # Room
    cv2.rectangle(img, (50, 50), (200, 200), (50, 50, 50), 3)  # Walls
    cv2.rectangle(img, (100, 195), (150, 205), (100, 80, 60), -1)  # Door

    # Test all methods
    edges_canny = generate_edge_labels(img)
    edges_multi = generate_edge_labels_multi_scale(img)
    edges_sobel = generate_edge_labels_sobel(img)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[1].imshow(edges_canny, cmap='gray')
    axes[1].set_title('Canny')
    axes[2].imshow(edges_multi, cmap='gray')
    axes[2].set_title('Multi-scale')
    axes[3].imshow(edges_sobel, cmap='gray')
    axes[3].set_title('Sobel')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('edge_test.png')
    print("Saved edge_test.png")

    # Test batch processing
    batch = torch.rand(4, 3, 256, 256)
    task = EdgePredictionTask(method='canny')
    labels = task.generate_labels(batch)
    print(f"Batch input: {batch.shape}")
    print(f"Labels output: {labels.shape}")
