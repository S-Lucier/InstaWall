"""
Colorization pretext task.

Input: Grayscale battlemap
Target: Original RGB battlemap

This teaches the encoder to understand material differences:
- Walls often have different colors than floors
- Wood vs stone vs dirt have different color profiles
- The model learns to distinguish regions by texture/pattern

Similar to how auto-wall uses color clustering to separate walls from floors.
"""

import cv2
import numpy as np
import torch


def rgb_to_grayscale(image):
    """Convert RGB image to grayscale.

    Args:
        image: RGB image as numpy array (H, W, 3) with values 0-255
               or torch tensor (C, H, W) or (B, C, H, W) with values 0-1

    Returns:
        Grayscale image in same format as input
    """
    if isinstance(image, torch.Tensor):
        # Use standard luminance weights
        if image.dim() == 3:
            # (C, H, W)
            r, g, b = image[0], image[1], image[2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.unsqueeze(0)  # (1, H, W)
        elif image.dim() == 4:
            # (B, C, H, W)
            r, g, b = image[:, 0], image[:, 1], image[:, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.unsqueeze(1)  # (B, 1, H, W)
    else:
        # Numpy array
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return gray
        return image


def generate_colorization_pair(image):
    """Generate input-target pair for colorization task.

    Args:
        image: RGB image as numpy array (H, W, 3) with values 0-255
               or torch tensor (C, H, W) with values 0-1

    Returns:
        Tuple of (grayscale_input, rgb_target)
        - grayscale_input: (1, H, W) or (H, W) depending on input type
        - rgb_target: Same as input image
    """
    gray = rgb_to_grayscale(image)
    return gray, image


def batch_generate_colorization_pairs(images):
    """Generate colorization pairs for a batch.

    Args:
        images: Batch of RGB images as torch tensor (B, 3, H, W) with values 0-1

    Returns:
        Tuple of (grayscale_inputs, rgb_targets)
        - grayscale_inputs: (B, 1, H, W)
        - rgb_targets: (B, 3, H, W) - same as input
    """
    gray = rgb_to_grayscale(images)
    return gray, images.clone()


class ColorizationTask:
    """Wrapper class for colorization pretext task.

    Handles input/target generation with optional augmentations.
    """

    def __init__(self, add_noise=False, noise_std=0.02):
        """
        Args:
            add_noise: If True, add noise to grayscale input
            noise_std: Standard deviation of noise
        """
        self.add_noise = add_noise
        self.noise_std = noise_std

    def generate_pairs(self, images):
        """Generate colorization pairs for a batch.

        Args:
            images: Batch of RGB images (B, 3, H, W)

        Returns:
            Tuple of (grayscale_input, rgb_target)
        """
        gray, target = batch_generate_colorization_pairs(images)

        if self.add_noise:
            noise = torch.randn_like(gray) * self.noise_std
            gray = torch.clamp(gray + noise, 0, 1)

        return gray, target


class LABColorization:
    """Alternative colorization using LAB color space.

    Instead of predicting RGB from grayscale, predict AB channels from L.
    This is often easier to learn and produces better results.
    """

    @staticmethod
    def rgb_to_lab(image):
        """Convert RGB to LAB color space.

        Args:
            image: RGB image as numpy array (H, W, 3) with values 0-255

        Returns:
            LAB image as numpy array (H, W, 3)
        """
        # OpenCV expects BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab

    @staticmethod
    def lab_to_rgb(lab):
        """Convert LAB to RGB color space.

        Args:
            lab: LAB image as numpy array (H, W, 3)

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def generate_pairs(self, images):
        """Generate L input and AB target pairs.

        Args:
            images: Batch of RGB images (B, 3, H, W) with values 0-1

        Returns:
            Tuple of (L_input, AB_target)
            - L_input: (B, 1, H, W) normalized to 0-1
            - AB_target: (B, 2, H, W) normalized to 0-1
        """
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]

        L_batch = torch.zeros(batch_size, 1, height, width)
        AB_batch = torch.zeros(batch_size, 2, height, width)

        for i in range(batch_size):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)

            lab = self.rgb_to_lab(img)

            # Normalize L (0-100) to 0-1
            L = lab[:, :, 0].astype(np.float32) / 100.0
            # Normalize AB (-128 to 127) to 0-1
            A = (lab[:, :, 1].astype(np.float32) + 128) / 255.0
            B = (lab[:, :, 2].astype(np.float32) + 128) / 255.0

            L_batch[i, 0] = torch.from_numpy(L)
            AB_batch[i, 0] = torch.from_numpy(A)
            AB_batch[i, 1] = torch.from_numpy(B)

        return L_batch, AB_batch


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create test image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[50:200, 50:200] = [200, 180, 150]  # Beige room
    img[30:220, 30:50] = [80, 70, 60]  # Dark wall
    img[30:220, 200:220] = [80, 70, 60]  # Dark wall
    img[30:50, 30:220] = [80, 70, 60]  # Dark wall
    img[200:220, 30:220] = [80, 70, 60]  # Dark wall
    img[100:150, 195:225] = [139, 90, 43]  # Brown door

    # Test basic colorization
    gray, target = generate_colorization_pair(img)
    print(f"Grayscale shape: {gray.shape}")
    print(f"Target shape: {target.shape}")

    # Test batch processing
    batch = torch.rand(4, 3, 256, 256)
    task = ColorizationTask()
    gray_batch, target_batch = task.generate_pairs(batch)
    print(f"Batch grayscale: {gray_batch.shape}")
    print(f"Batch target: {target_batch.shape}")

    # Test LAB colorization
    lab_task = LABColorization()
    L_batch, AB_batch = lab_task.generate_pairs(batch)
    print(f"L batch: {L_batch.shape}")
    print(f"AB batch: {AB_batch.shape}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Original RGB')
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale Input')
    axes[2].imshow(img)
    axes[2].set_title('Target RGB')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('colorization_test.png')
    print("Saved colorization_test.png")
