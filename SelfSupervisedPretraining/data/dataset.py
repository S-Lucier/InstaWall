"""
Dataset for self-supervised pretraining.

Loads unlabeled battlemap images and generates pretext task inputs on-the-fly.
Supports multiple data sources and augmentations.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Using basic transforms.")


class BattlemapDataset(Dataset):
    """Dataset for loading unlabeled battlemap images.

    Loads images from one or more directories and applies augmentations.
    No labels required - pretext task labels are generated on-the-fly.
    """

    def __init__(self, image_dirs, image_size=512, augment=True,
                 extensions=('.png', '.jpg', '.jpeg', '.webp')):
        """
        Args:
            image_dirs: Path or list of paths to image directories
            image_size: Target image size (square)
            augment: Whether to apply augmentations
            extensions: Valid image file extensions
        """
        self.image_size = image_size
        self.augment = augment

        # Collect all image paths
        if isinstance(image_dirs, (str, Path)):
            image_dirs = [image_dirs]

        self.image_paths = []
        for dir_path in image_dirs:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                print(f"Warning: Directory not found: {dir_path}")
                continue

            for ext in extensions:
                self.image_paths.extend(dir_path.glob(f'*{ext}'))
                self.image_paths.extend(dir_path.glob(f'*{ext.upper()}'))

        self.image_paths = sorted(set(self.image_paths))
        print(f"Found {len(self.image_paths)} images")

        # Setup transforms
        self.transform = self._build_transform()

    def _build_transform(self):
        """Build augmentation pipeline."""
        if not HAS_ALBUMENTATIONS:
            return None

        if self.augment:
            return A.Compose([
                # Resize and crop
                A.LongestMaxSize(max_size=self.image_size + 64),
                A.RandomCrop(self.image_size, self.image_size),

                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                # Color augmentations (mild - we want to learn color differences)
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=0.5
                ),

                # Quality augmentations
                A.OneOf([
                    A.GaussNoise(var_limit=(5, 20), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
                ], p=0.3),

                # Normalize and convert
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(
                    min_height=self.image_size,
                    min_width=self.image_size,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                    value=(0, 0, 0)
                ),
                A.CenterCrop(self.image_size, self.image_size),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load and transform an image.

        Returns:
            Image tensor (C, H, W) with values 0-1
        """
        img_path = self.image_paths[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid image instead
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Basic fallback transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image


class PretrainingDataset(Dataset):
    """Dataset that generates all pretext task inputs.

    Wraps BattlemapDataset and generates inputs for all tasks on-the-fly.
    """

    def __init__(self, base_dataset, tasks=None):
        """
        Args:
            base_dataset: BattlemapDataset or similar
            tasks: Dict of task name -> task object
                   If None, uses default tasks
        """
        self.base_dataset = base_dataset

        # Import tasks here to avoid circular imports
        from ..pretext_tasks import (
            EdgePredictionTask,
            ColorizationTask,
            MAETask,
            JigsawTask
        )

        self.tasks = tasks or {
            'edge': EdgePredictionTask(method='canny', random_thresholds=True),
            'color': ColorizationTask(add_noise=True),
            'mae': MAETask(patch_size=32, mask_ratio=0.75, variable_ratio=True),
            'jigsaw': JigsawTask(grid_size=3, num_permutations=100)
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """Get image and generate all pretext task inputs.

        Returns:
            Dict with:
                'image': Original image (C, H, W)
                'edge_target': Edge labels (1, H, W)
                'gray_input': Grayscale input for colorization (1, H, W)
                'color_target': RGB target for colorization (3, H, W)
                'mae_input': Masked image (C, H, W)
                'mae_target': Original image (C, H, W)
                'mae_mask': Binary mask (H, W)
                'jigsaw_input': Shuffled image (C, H, W)
                'jigsaw_target': Permutation index (scalar)
        """
        image = self.base_dataset[idx]

        # Add batch dimension for task processing
        image_batch = image.unsqueeze(0)

        result = {'image': image}

        # Edge prediction
        if 'edge' in self.tasks:
            edge_labels = self.tasks['edge'].generate_labels(image_batch)
            result['edge_target'] = edge_labels[0]  # Remove batch dim

        # Colorization
        if 'color' in self.tasks:
            gray, color = self.tasks['color'].generate_pairs(image_batch)
            result['gray_input'] = gray[0]
            result['color_target'] = color[0]

        # MAE
        if 'mae' in self.tasks:
            masked, target, mask = self.tasks['mae'].generate_inputs(image_batch)
            result['mae_input'] = masked[0]
            result['mae_target'] = target[0]
            result['mae_mask'] = mask[0]

        # Jigsaw
        if 'jigsaw' in self.tasks:
            shuffled, perm_idx = self.tasks['jigsaw'].generate_inputs(image_batch)
            result['jigsaw_input'] = shuffled[0]
            result['jigsaw_target'] = perm_idx[0]

        return result


def create_dataloader(image_dirs, batch_size=8, image_size=512,
                     num_workers=4, augment=True, shuffle=True):
    """Create a DataLoader for pretraining.

    Args:
        image_dirs: Path(s) to image directories
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        augment: Whether to apply augmentations
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    base_dataset = BattlemapDataset(
        image_dirs=image_dirs,
        image_size=image_size,
        augment=augment
    )

    dataset = PretrainingDataset(base_dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Avoid issues with batch norm on last small batch
    )


if __name__ == "__main__":
    # Test with a sample directory
    import sys

    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        # Default test path
        test_dir = r"C:\Users\shini\InstaWall\WatabouTestModel\data\train_images"

    print(f"Testing dataset with: {test_dir}")

    # Create dataset
    base_ds = BattlemapDataset(test_dir, image_size=256, augment=True)
    print(f"Base dataset size: {len(base_ds)}")

    if len(base_ds) > 0:
        # Test base dataset
        img = base_ds[0]
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")

        # Test pretraining dataset
        pretrain_ds = PretrainingDataset(base_ds)
        sample = pretrain_ds[0]

        print("\nPretraining sample keys:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")

        # Test dataloader
        loader = create_dataloader(test_dir, batch_size=2, image_size=256, num_workers=0)
        batch = next(iter(loader))

        print("\nBatch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
    else:
        print("No images found. Please provide a valid image directory.")
