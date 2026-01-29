import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    """Generic segmentation dataset for loading images and masks.

    Args:
        image_dir: Directory containing input images
        mask_dir: Directory containing mask images
        transform: Albumentations transform pipeline
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Normalize mask to [0, 1]
        mask = mask / 255.0

        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


def get_train_transforms(image_height=512, image_width=512):
    """Training data augmentation pipeline.

    All augmentation parameters are grouped here for easy tuning.
    See Text Files/Model Augmentations.txt for detailed explanations.
    """
    return A.Compose([
        # ============================================================
        # PREPROCESSING (resize/pad to target size)
        # ============================================================
        A.LongestMaxSize(max_size=max(image_height, image_width)),
        A.PadIfNeeded(min_height=image_height, min_width=image_width, border_mode=0),

        # ============================================================
        # GEOMETRIC AUGMENTATIONS (preserve grid alignment)
        # ============================================================
        A.RandomRotate90(p=1.0),        # Always rotate by 0/90/180/270 degrees (25% each)
        A.HorizontalFlip(p=0.7),        # 70% chance to flip horizontally
        A.VerticalFlip(p=0.7),          # 70% chance to flip vertically

        # ============================================================
        # COLOR/BRIGHTNESS AUGMENTATIONS (lighting variations)
        # ============================================================
        A.RandomBrightnessContrast(
            brightness_limit=0.25,      # ±25% brightness adjustment
            contrast_limit=0.25,        # ±25% contrast adjustment
            p=0.6                       # Apply to 60% of images
        ),
        A.ColorJitter(
            brightness=0.15,            # ±15% brightness jitter
            contrast=0.15,              # ±15% contrast jitter
            saturation=0.15,            # ±15% saturation jitter (handles color shifts)
            hue=0.08,                   # ±8% hue shift (slight color tinting)
            p=0.4                       # Apply to 40% of images
        ),

        # ============================================================
        # QUALITY DEGRADATION (simulate scans/photos/compression)
        # ============================================================
        A.GaussNoise(
            var_limit=(5.0, 20.0),      # Add subtle noise (low variance)
            p=0.2                       # Apply to 20% of images
        ),
        A.Blur(
            blur_limit=3,               # Small blur (1-3 pixel kernel)
            p=0.2                       # Apply to 20% of images
        ),

        # ============================================================
        # NORMALIZATION (scale to 0-1 range)
        # ============================================================
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_height=512, image_width=512):
    """Validation transforms (no augmentation, only resize and normalize)."""
    return A.Compose([
        A.LongestMaxSize(max_size=max(image_height, image_width)),
        A.PadIfNeeded(min_height=image_height, min_width=image_width, border_mode=0),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
