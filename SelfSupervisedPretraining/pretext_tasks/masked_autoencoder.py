"""
Masked Autoencoder (MAE) pretext task.

Input: Image with random patches masked out
Target: Original image (reconstruct masked regions)

This teaches the encoder to understand spatial structure:
- Room layouts and shapes
- Wall continuity
- Spatial relationships between elements

The model must learn to "fill in" missing parts, requiring
understanding of the overall structure.
"""

import numpy as np
import torch
import torch.nn.functional as F


def create_patch_mask(height, width, patch_size=16, mask_ratio=0.75):
    """Create a random patch mask.

    Args:
        height: Image height
        width: Image width
        patch_size: Size of each patch
        mask_ratio: Fraction of patches to mask

    Returns:
        Binary mask (H, W) where 1 = masked, 0 = visible
    """
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size
    n_patches = n_patches_h * n_patches_w
    n_masked = int(n_patches * mask_ratio)

    # Randomly select patches to mask
    patch_indices = np.random.permutation(n_patches)
    masked_indices = patch_indices[:n_masked]

    # Create patch-level mask
    patch_mask = np.zeros(n_patches, dtype=np.float32)
    patch_mask[masked_indices] = 1.0
    patch_mask = patch_mask.reshape(n_patches_h, n_patches_w)

    # Upscale to pixel level
    mask = np.repeat(np.repeat(patch_mask, patch_size, axis=0), patch_size, axis=1)

    # Handle size mismatch due to non-divisible dimensions
    if mask.shape[0] != height or mask.shape[1] != width:
        mask_full = np.zeros((height, width), dtype=np.float32)
        mask_full[:mask.shape[0], :mask.shape[1]] = mask
        mask = mask_full

    return mask


def create_block_mask(height, width, num_blocks=4, block_size_range=(0.1, 0.3)):
    """Create a random block mask (alternative to patches).

    Masks random rectangular blocks of varying sizes.
    More similar to how parts of a map might be occluded.

    Args:
        height: Image height
        width: Image width
        num_blocks: Number of blocks to mask
        block_size_range: (min, max) fraction of image size for blocks

    Returns:
        Binary mask (H, W) where 1 = masked, 0 = visible
    """
    mask = np.zeros((height, width), dtype=np.float32)

    for _ in range(num_blocks):
        # Random block size
        bh = int(height * np.random.uniform(*block_size_range))
        bw = int(width * np.random.uniform(*block_size_range))

        # Random position
        y = np.random.randint(0, height - bh + 1)
        x = np.random.randint(0, width - bw + 1)

        mask[y:y+bh, x:x+bw] = 1.0

    return mask


def apply_mask(image, mask, mask_value=0.0):
    """Apply mask to image.

    Args:
        image: Image tensor (C, H, W) or (B, C, H, W)
        mask: Binary mask (H, W) or (B, H, W) or (B, 1, H, W)
        mask_value: Value to fill masked regions (default: 0 / black)

    Returns:
        Masked image with same shape as input
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    # Ensure mask has correct dimensions
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
    if mask.dim() == 3 and image.dim() == 4:
        mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

    # Expand mask to match image channels
    if image.dim() == 3:
        mask = mask.expand(image.shape[0], -1, -1)
    elif image.dim() == 4:
        mask = mask.expand(-1, image.shape[1], -1, -1)

    # Apply mask
    masked_image = image.clone()
    masked_image[mask.bool()] = mask_value

    return masked_image


def generate_mae_input(image, patch_size=16, mask_ratio=0.75, mask_type='patch'):
    """Generate MAE input and targets.

    Args:
        image: Image tensor (C, H, W) or numpy array (H, W, C)
        patch_size: Size of patches (for patch masking)
        mask_ratio: Fraction to mask
        mask_type: 'patch' or 'block'

    Returns:
        Tuple of (masked_image, original_image, mask)
    """
    if isinstance(image, np.ndarray):
        # Convert (H, W, C) -> (C, H, W)
        if image.ndim == 3:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if image.max() > 1.0:
                image = image / 255.0
        else:
            image = torch.from_numpy(image).float()

    height, width = image.shape[-2:]

    if mask_type == 'patch':
        mask = create_patch_mask(height, width, patch_size, mask_ratio)
    elif mask_type == 'block':
        mask = create_block_mask(height, width)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    mask_tensor = torch.from_numpy(mask)
    masked_image = apply_mask(image, mask_tensor, mask_value=0.0)

    return masked_image, image, mask_tensor


def batch_generate_mae_inputs(images, patch_size=16, mask_ratio=0.75, mask_type='patch'):
    """Generate MAE inputs for a batch.

    Args:
        images: Batch of images (B, C, H, W)
        patch_size: Size of patches
        mask_ratio: Fraction to mask
        mask_type: 'patch' or 'block'

    Returns:
        Tuple of (masked_images, original_images, masks)
        - masked_images: (B, C, H, W)
        - original_images: (B, C, H, W)
        - masks: (B, H, W)
    """
    batch_size = images.shape[0]
    height, width = images.shape[2], images.shape[3]

    masked_images = torch.zeros_like(images)
    masks = torch.zeros(batch_size, height, width)

    for i in range(batch_size):
        masked_img, _, mask = generate_mae_input(
            images[i], patch_size, mask_ratio, mask_type
        )
        masked_images[i] = masked_img
        masks[i] = mask

    return masked_images, images.clone(), masks


class MAETask:
    """Wrapper class for MAE pretext task."""

    def __init__(self, patch_size=16, mask_ratio=0.75, mask_type='patch',
                 variable_ratio=False, ratio_range=(0.5, 0.9)):
        """
        Args:
            patch_size: Size of patches for masking
            mask_ratio: Base fraction of image to mask
            mask_type: 'patch' or 'block'
            variable_ratio: If True, randomly vary mask ratio
            ratio_range: Range for variable mask ratio
        """
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.variable_ratio = variable_ratio
        self.ratio_range = ratio_range

    def generate_inputs(self, images):
        """Generate MAE inputs for a batch.

        Args:
            images: Batch of images (B, C, H, W)

        Returns:
            Tuple of (masked_images, target_images, masks)
        """
        if self.variable_ratio:
            ratio = np.random.uniform(*self.ratio_range)
        else:
            ratio = self.mask_ratio

        return batch_generate_mae_inputs(
            images,
            patch_size=self.patch_size,
            mask_ratio=ratio,
            mask_type=self.mask_type
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create test image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[50:200, 50:200] = [200, 180, 150]  # Room
    img[30:220, 30:50] = [80, 70, 60]  # Walls
    img[30:220, 200:220] = [80, 70, 60]
    img[30:50, 30:220] = [80, 70, 60]
    img[200:220, 30:220] = [80, 70, 60]

    # Convert to tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

    # Test patch masking
    masked_patch, target, mask_patch = generate_mae_input(
        img_tensor, patch_size=32, mask_ratio=0.75, mask_type='patch'
    )

    # Test block masking
    masked_block, _, mask_block = generate_mae_input(
        img_tensor, patch_size=32, mask_ratio=0.5, mask_type='block'
    )

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')

    axes[0, 1].imshow(masked_patch.permute(1, 2, 0).numpy())
    axes[0, 1].set_title('Patch Masked (75%)')

    axes[0, 2].imshow(mask_patch.numpy(), cmap='gray')
    axes[0, 2].set_title('Patch Mask')

    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Original')

    axes[1, 1].imshow(masked_block.permute(1, 2, 0).numpy())
    axes[1, 1].set_title('Block Masked')

    axes[1, 2].imshow(mask_block.numpy(), cmap='gray')
    axes[1, 2].set_title('Block Mask')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('mae_test.png')
    print("Saved mae_test.png")

    # Test batch processing
    batch = torch.rand(4, 3, 256, 256)
    task = MAETask(patch_size=32, mask_ratio=0.75)
    masked, target, masks = task.generate_inputs(batch)
    print(f"Masked batch: {masked.shape}")
    print(f"Target batch: {target.shape}")
    print(f"Masks batch: {masks.shape}")
