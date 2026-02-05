"""
Jigsaw puzzle pretext task.

Input: Image with patches shuffled according to a permutation
Target: Classify which permutation was used

This teaches the encoder to understand spatial relationships:
- Rooms connect in certain ways
- Walls form continuous boundaries
- Doors appear at specific locations relative to walls

The model learns global spatial structure by figuring out
how pieces fit together.
"""

import math
import numpy as np
import torch
import itertools


def generate_permutations(num_positions, num_permutations=100, seed=42):
    """Generate a fixed set of permutations for classification.

    We don't use all possible permutations (9! = 362880 for 3x3).
    Instead, we select a diverse subset.

    Args:
        num_positions: Number of patches (e.g., 9 for 3x3 grid)
        num_permutations: Number of permutations to generate
        seed: Random seed for reproducibility

    Returns:
        List of permutation tuples
    """
    np.random.seed(seed)

    # Generate candidate permutations
    all_positions = list(range(num_positions))
    permutations = []

    # Always include identity permutation
    permutations.append(tuple(all_positions))

    # Generate random permutations
    while len(permutations) < num_permutations:
        perm = tuple(np.random.permutation(all_positions))
        # Avoid duplicates and identity
        if perm not in permutations:
            permutations.append(perm)

    return permutations


def hamming_distance(perm1, perm2):
    """Compute Hamming distance between two permutations."""
    return sum(p1 != p2 for p1, p2 in zip(perm1, perm2))


def generate_diverse_permutations(num_positions, num_permutations=100, seed=42):
    """Generate diverse permutations with good separation.

    Tries to maximize minimum Hamming distance between permutations
    for more challenging and informative classification.
    """
    np.random.seed(seed)

    all_positions = list(range(num_positions))
    permutations = [tuple(all_positions)]  # Start with identity

    # Generate many candidates
    n_candidates = min(10000, math.factorial(num_positions))
    candidates = []
    while len(candidates) < n_candidates:
        perm = tuple(np.random.permutation(all_positions))
        if perm not in candidates and perm not in permutations:
            candidates.append(perm)

    # Greedily select diverse permutations
    while len(permutations) < num_permutations and candidates:
        best_perm = None
        best_min_dist = -1

        for perm in candidates:
            min_dist = min(hamming_distance(perm, p) for p in permutations)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_perm = perm

        if best_perm:
            permutations.append(best_perm)
            candidates.remove(best_perm)
        else:
            break

    return permutations


# Pre-computed permutations for 3x3 grid (9 positions)
# Using simple random permutations for faster import; diverse selection is too slow
PERMUTATIONS_3x3 = generate_permutations(9, 100, seed=42)

# Pre-computed permutations for 2x2 grid (4 positions) - easier task
PERMUTATIONS_2x2 = generate_permutations(4, 24, seed=42)  # All 24 permutations


def split_into_grid(image, grid_size=3):
    """Split image into grid of patches.

    Args:
        image: Image tensor (C, H, W) or numpy array (H, W, C)
        grid_size: Number of patches per side

    Returns:
        List of patches in row-major order
    """
    if isinstance(image, np.ndarray):
        # (H, W, C) -> (C, H, W)
        if image.ndim == 3:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if image.max() > 1.0:
                image = image / 255.0

    C, H, W = image.shape
    patch_h = H // grid_size
    patch_w = W // grid_size

    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = image[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            patches.append(patch)

    return patches


def reassemble_grid(patches, grid_size=3, original_size=None):
    """Reassemble patches into an image.

    Args:
        patches: List of patch tensors (C, pH, pW)
        grid_size: Number of patches per side
        original_size: Optional tuple (H, W) to pad result to original size

    Returns:
        Reassembled image tensor (C, H, W)
    """
    C = patches[0].shape[0]
    patch_h = patches[0].shape[1]
    patch_w = patches[0].shape[2]

    H = patch_h * grid_size
    W = patch_w * grid_size

    image = torch.zeros(C, H, W, dtype=patches[0].dtype)

    for idx, patch in enumerate(patches):
        i = idx // grid_size
        j = idx % grid_size
        image[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = patch

    # Pad to original size if needed
    if original_size is not None:
        orig_h, orig_w = original_size
        if H < orig_h or W < orig_w:
            padded = torch.zeros(C, orig_h, orig_w, dtype=patches[0].dtype)
            padded[:, :H, :W] = image
            image = padded

    return image


def apply_permutation(patches, permutation):
    """Shuffle patches according to permutation.

    Args:
        patches: List of patches
        permutation: Tuple of indices defining the shuffle

    Returns:
        List of shuffled patches
    """
    return [patches[i] for i in permutation]


def generate_jigsaw_input(image, grid_size=3, permutations=None):
    """Generate jigsaw puzzle input and target.

    Args:
        image: Image tensor (C, H, W)
        grid_size: Grid size (2 or 3)
        permutations: List of permutations to use

    Returns:
        Tuple of (shuffled_image, permutation_index)
    """
    if permutations is None:
        if grid_size == 3:
            permutations = PERMUTATIONS_3x3
        elif grid_size == 2:
            permutations = PERMUTATIONS_2x2
        else:
            permutations = generate_permutations(grid_size * grid_size)

    # Get original size for padding back
    _, orig_h, orig_w = image.shape

    # Split into patches
    patches = split_into_grid(image, grid_size)

    # Select random permutation
    perm_idx = np.random.randint(0, len(permutations))
    perm = permutations[perm_idx]

    # Apply permutation
    shuffled_patches = apply_permutation(patches, perm)

    # Reassemble with original size
    shuffled_image = reassemble_grid(shuffled_patches, grid_size, original_size=(orig_h, orig_w))

    return shuffled_image, perm_idx


def batch_generate_jigsaw_inputs(images, grid_size=3, permutations=None):
    """Generate jigsaw inputs for a batch.

    Args:
        images: Batch of images (B, C, H, W)
        grid_size: Grid size
        permutations: List of permutations

    Returns:
        Tuple of (shuffled_images, permutation_indices)
    """
    if permutations is None:
        if grid_size == 3:
            permutations = PERMUTATIONS_3x3
        elif grid_size == 2:
            permutations = PERMUTATIONS_2x2
        else:
            permutations = generate_permutations(grid_size * grid_size)

    batch_size = images.shape[0]
    shuffled_images = torch.zeros_like(images)
    perm_indices = torch.zeros(batch_size, dtype=torch.long)

    for i in range(batch_size):
        shuffled, idx = generate_jigsaw_input(images[i], grid_size, permutations)
        shuffled_images[i] = shuffled
        perm_indices[i] = idx

    return shuffled_images, perm_indices


class JigsawTask:
    """Wrapper class for jigsaw puzzle pretext task."""

    def __init__(self, grid_size=3, num_permutations=100, add_gap=False, gap_size=2):
        """
        Args:
            grid_size: Grid size (2, 3, or 4)
            num_permutations: Number of permutation classes
            add_gap: If True, add small gaps between patches
            gap_size: Size of gap in pixels
        """
        self.grid_size = grid_size
        self.num_permutations = num_permutations
        self.add_gap = add_gap
        self.gap_size = gap_size

        # Generate or use pre-computed permutations
        if grid_size == 3 and num_permutations <= 100:
            self.permutations = PERMUTATIONS_3x3[:num_permutations]
        elif grid_size == 2:
            self.permutations = PERMUTATIONS_2x2[:min(24, num_permutations)]
        else:
            self.permutations = generate_diverse_permutations(
                grid_size * grid_size, num_permutations
            )

    def generate_inputs(self, images):
        """Generate jigsaw inputs for a batch.

        Args:
            images: Batch of images (B, C, H, W)

        Returns:
            Tuple of (shuffled_images, permutation_indices)
        """
        shuffled, indices = batch_generate_jigsaw_inputs(
            images, self.grid_size, self.permutations
        )

        if self.add_gap:
            shuffled = self._add_gaps(shuffled)

        return shuffled, indices

    def _add_gaps(self, images):
        """Add gaps between patches to make task slightly easier."""
        B, C, H, W = images.shape
        patch_h = H // self.grid_size
        patch_w = W // self.grid_size

        for i in range(1, self.grid_size):
            # Horizontal gaps
            y = i * patch_h
            images[:, :, y-self.gap_size:y+self.gap_size, :] = 0
            # Vertical gaps
            x = i * patch_w
            images[:, :, :, x-self.gap_size:x+self.gap_size] = 0

        return images

    def get_num_classes(self):
        """Return number of permutation classes."""
        return len(self.permutations)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create test image with distinct regions
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Different colored quadrants
    img[0:128, 0:128] = [200, 100, 100]    # Top-left: red
    img[0:128, 128:256] = [100, 200, 100]  # Top-right: green
    img[128:256, 0:128] = [100, 100, 200]  # Bottom-left: blue
    img[128:256, 128:256] = [200, 200, 100] # Bottom-right: yellow

    # Add some structure
    cv2 = None
    try:
        import cv2
        cv2.rectangle(img, (30, 30), (100, 100), (50, 50, 50), 2)
        cv2.rectangle(img, (150, 30), (220, 100), (50, 50, 50), 2)
    except ImportError:
        pass

    # Convert to tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

    # Generate jigsaw puzzles
    task_3x3 = JigsawTask(grid_size=3, num_permutations=100)
    task_2x2 = JigsawTask(grid_size=2, num_permutations=24)

    # Single image test
    shuffled_3x3, idx_3x3 = generate_jigsaw_input(img_tensor, grid_size=3)
    shuffled_2x2, idx_2x2 = generate_jigsaw_input(img_tensor, grid_size=2)

    print(f"3x3 permutation index: {idx_3x3}")
    print(f"2x2 permutation index: {idx_2x2}")
    print(f"3x3 permutation: {PERMUTATIONS_3x3[idx_3x3]}")
    print(f"2x2 permutation: {PERMUTATIONS_2x2[idx_2x2]}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title('Original')

    axes[1].imshow(shuffled_3x3.permute(1, 2, 0).numpy())
    axes[1].set_title(f'3x3 Jigsaw (perm {idx_3x3})')

    axes[2].imshow(shuffled_2x2.permute(1, 2, 0).numpy())
    axes[2].set_title(f'2x2 Jigsaw (perm {idx_2x2})')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('jigsaw_test.png')
    print("Saved jigsaw_test.png")

    # Test batch processing
    batch = torch.rand(4, 3, 256, 256)
    shuffled_batch, indices = task_3x3.generate_inputs(batch)
    print(f"Shuffled batch: {shuffled_batch.shape}")
    print(f"Indices: {indices.shape}, values: {indices.tolist()}")
    print(f"Num classes: {task_3x3.get_num_classes()}")
