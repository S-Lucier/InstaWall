"""
Dataset for wall segmentation training.

Loads battlemap images and their corresponding line segment masks,
extracts tiles at consistent grid scale, and applies augmentations.
"""

import json
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import binary_dilation

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Using basic transforms.")

from .tiling import TileExtractor


class WallSegmentationDataset(Dataset):
    """
    Dataset for training wall segmentation model.

    Loads battlemap images and their corresponding masks, extracts tiles
    at consistent grid scale, and applies augmentations.

    Expected directory structure:
        image_dir/
            MapName.jpg (or .png, .webp)
        mask_dir/
            MapName_mask_lines.png

    Each mask file should have a corresponding metadata file or the grid size
    should be provided per-sample.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        metadata_file: Optional[str] = None,
        tile_grid_cells: int = 8,
        tile_size: int = 512,
        tiles_per_image: int = 4,
        mask_scale: int = 50,
        augment: bool = True,
        default_grid_size: int = 140,
        merge_terrain: bool = False,
        watabou_dir: Optional[str] = None,
        watabou_include_prob: float = 0.36,
        use_imagenet_norm: bool = False,
        global_image_size: int = 0,
        mask_dilation: int = 0,
    ):
        """
        Args:
            image_dir: Directory containing battlemap images
            mask_dir: Directory containing mask files
            metadata_file: Optional JSON file with per-image metadata (grid_size, etc.)
            tile_grid_cells: Number of grid cells per tile
            tile_size: Output tile size in pixels
            tiles_per_image: Number of random tiles to extract per image per epoch
            mask_scale: Scale factor used in mask files (class = value / scale)
            augment: Whether to apply augmentations
            default_grid_size: Default grid size if not in metadata
            merge_terrain: Whether to merge terrain class into wall class
            watabou_dir: Optional path to Watabou data (watabou_images/, watabou_edge_mask/, watabou_recessed_mask/)
            watabou_include_prob: Per-epoch inclusion probability for each Watabou map
            use_imagenet_norm: Use ImageNet normalization (for SegFormer) instead of 0-1 range
            global_image_size: If >0, also return the full image downscaled to this size (for global context)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.tile_grid_cells = tile_grid_cells
        self.tile_size = tile_size
        self.tiles_per_image = tiles_per_image
        self.mask_scale = mask_scale
        self.augment = augment
        self.default_grid_size = default_grid_size
        self.merge_terrain = merge_terrain
        self.watabou_dir = Path(watabou_dir) if watabou_dir else None
        self.watabou_include_prob = watabou_include_prob
        self.use_imagenet_norm = use_imagenet_norm
        self.global_image_size = global_image_size
        self.mask_dilation = mask_dilation

        # Pre-build dilation structuring element
        if mask_dilation > 0:
            r = mask_dilation
            y, x = np.ogrid[-r:r+1, -r:r+1]
            self._dilation_struct = (x*x + y*y) <= r*r  # circular kernel

        self.extractor = TileExtractor(tile_grid_cells, tile_size, overlap=0.0)

        # Load metadata if provided
        self.metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)

        # Find all image-mask pairs from all sources
        self.all_samples = self._find_samples()

        foundry_count = sum(1 for s in self.all_samples if s['source'] == 'foundry')
        watabou_count = sum(1 for s in self.all_samples if s['source'] == 'watabou')
        print(f"Found {len(self.all_samples)} image-mask pairs "
              f"(Foundry: {foundry_count}, Watabou: {watabou_count})")
        if watabou_count > 0:
            print(f"  Watabou include probability: {watabou_include_prob:.2f} "
                  f"(~{int(watabou_count * watabou_include_prob)} per epoch)")

        # Active samples for current epoch (resampled each epoch)
        self.active_samples = list(self.all_samples)
        self.resample_for_epoch()

        # Build augmentation pipelines
        self.transform = self._build_transform()
        self.watabou_transform = self._build_watabou_transform()

        # Build global image transform (normalize only, no augmentation)
        if self.global_image_size > 0 and HAS_ALBUMENTATIONS:
            if self.use_imagenet_norm:
                gnorm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                gnorm = A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
            self.global_transform = A.Compose([gnorm, ToTensorV2()])
        else:
            self.global_transform = None

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Expand non-background classes outward, writing only into background pixels.

        Each class is dilated independently so class boundaries are preserved —
        door pixels are never overwritten by wall dilation and vice versa.
        """
        result = mask.copy()
        background = mask == 0
        num_classes = int(mask.max()) + 1
        for cls in range(1, num_classes):
            expanded = binary_dilation(mask == cls, structure=self._dilation_struct)
            # Only write into background — never overwrite other classes
            result[expanded & background] = cls
            background = result == 0  # update remaining background
        return result

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for fuzzy matching."""
        # Convert to lowercase
        name = name.lower()
        # Replace separators with nothing
        name = name.replace('_', '').replace('-', '').replace(' ', '')
        # Remove common articles that might be missing
        name = name.replace('the', '').replace('of', '')
        return name

    def _find_samples(self) -> List[Dict]:
        """Find matching image-mask pairs from all sources."""
        samples = []
        samples.extend(self._find_foundry_samples())
        if self.watabou_dir:
            samples.extend(self._find_watabou_samples())
        return samples

    def _find_foundry_samples(self) -> List[Dict]:
        """Find Foundry image-mask pairs using name mapping and fuzzy matching."""
        samples = []
        extensions = ['.jpg', '.jpeg', '.png', '.webp']

        # Try to load explicit name mapping first
        name_mapping = {}
        mapping_path = self.image_dir / 'name_mapping.json'
        if mapping_path.exists():
            with open(mapping_path) as f:
                name_mapping = json.load(f)
            print(f"Loaded name mapping with {len(name_mapping)} entries")

        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(self.image_dir.glob(f'*{ext}'))
            image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))

        # Filter out mask/viz files
        image_files = [f for f in image_files if '_mask' not in f.stem and '_viz' not in f.stem]

        # Build lookup by filename
        image_by_name = {f.name: f for f in image_files}

        # Build lookup by normalized name (fallback)
        image_lookup = {}
        for img_path in image_files:
            norm_name = self._normalize_name(img_path.stem)
            image_lookup[norm_name] = img_path

        # Get all mask files
        mask_files = list(self.mask_dir.glob('*_mask_lines.png'))

        for mask_path in mask_files:
            # Extract base name (remove _mask_lines.png suffix)
            base_name = mask_path.stem.replace('_mask_lines', '')

            # Try explicit mapping first
            image_path = None
            if base_name in name_mapping:
                mapped_name = name_mapping[base_name]
                image_path = image_by_name.get(mapped_name)

            # Fall back to normalized name matching
            if image_path is None:
                norm_name = self._normalize_name(base_name)
                image_path = image_lookup.get(norm_name)

            if image_path is None:
                # Try partial matching
                norm_name = self._normalize_name(base_name)
                for img_norm, img_path in image_lookup.items():
                    if norm_name in img_norm or img_norm in norm_name:
                        image_path = img_path
                        break

            if image_path is None:
                print(f"Warning: No image found for mask {mask_path.name}")
                continue

            # Get grid size from metadata or use default
            grid_size = self.metadata.get(base_name, {}).get('grid_size', self.default_grid_size)

            samples.append({
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'mask_path_alt': None,
                'name': base_name,
                'grid_size': grid_size,
                'source': 'foundry',
            })

        return samples

    def _find_watabou_samples(self) -> List[Dict]:
        """Find Watabou image-mask pairs (direct filename match, edge masks)."""
        samples = []
        image_dir = self.watabou_dir / 'watabou_images'
        mask_dir = self.watabou_dir / 'watabou_edge_mask'

        for image_path in sorted(image_dir.glob('*.png')):
            name = image_path.stem
            mask = mask_dir / f"{name}.png"

            if not mask.exists():
                print(f"Warning: No edge mask found for watabou map {name}")
                continue

            samples.append({
                'image_path': str(image_path),
                'mask_path': str(mask),
                'name': name,
                'grid_size': 70,  # Watabou fixed grid size
                'source': 'watabou',
            })

        return samples

    def resample_for_epoch(self):
        """Resample active samples for a new epoch.

        Foundry maps are always included. Watabou maps are subsampled
        with watabou_include_prob.
        """
        self.active_samples = []

        for sample in self.all_samples:
            if sample['source'] == 'watabou':
                if random.random() > self.watabou_include_prob:
                    continue
            self.active_samples.append(sample)

    def _build_watabou_transform(self):
        """Heavier augmentation pipeline for Watabou (vector art) maps.

        Watabou maps have a very different visual style (flat/vector art) compared
        to Foundry photorealistic maps. Heavy augmentation prevents overfitting to
        the vector art style and improves generalisation.
        """
        if not HAS_ALBUMENTATIONS:
            return self._build_transform()

        if self.use_imagenet_norm:
            norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            norm = A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

        if self.augment:
            return A.Compose([
                # Geometric
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                # Heavy colour: wide brightness/contrast/saturation/hue swings
                A.ColorJitter(
                    brightness=0.6,
                    contrast=0.6,
                    saturation=0.5,
                    hue=0.2,
                    p=0.8,
                ),
                A.RandomGamma(gamma_limit=(40, 200), p=0.5),

                # Occasionally convert to greyscale (forces texture focus)
                A.ToGray(p=0.2),

                # Quality / texture noise
                A.OneOf([
                    A.GaussNoise(std_range=(0.03, 0.08), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.ImageCompression(quality_range=(30, 80), p=1.0),
                    A.Downscale(scale_range=(0.5, 0.8), p=1.0),
                ], p=0.5),

                norm,
                ToTensorV2()
            ])
        else:
            return A.Compose([norm, ToTensorV2()])

    def _build_transform(self):
        """Build augmentation pipeline."""
        if not HAS_ALBUMENTATIONS:
            return None

        if self.use_imagenet_norm:
            norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            norm = A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

        if self.augment:
            return A.Compose([
                # Geometric augmentations (applied to both image and mask)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                # Color augmentations (image only)
                A.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3,
                    hue=0.1,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(60, 140), p=0.3),

                # Quality augmentations (image only)
                A.OneOf([
                    A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.ImageCompression(quality_range=(40, 90), p=1.0),
                ], p=0.3),

                # Normalize and convert
                norm,
                ToTensorV2()
            ])
        else:
            return A.Compose([
                norm,
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.active_samples) * self.tiles_per_image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample (tile).

        Returns:
            Dict with:
                'image': Image tile tensor (C, H, W)
                'mask': Mask tile tensor (H, W) with class indices
                'name': Sample name for debugging
        """
        # Determine which image and which tile
        sample_idx = idx // self.tiles_per_image
        sample = self.active_samples[sample_idx]

        # Load image and mask
        try:
            image = np.array(Image.open(sample['image_path']).convert('RGB'))
            mask = np.array(Image.open(sample['mask_path']))
        except Exception as e:
            print(f"Error loading {sample['name']}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Convert mask values to class indices
        # Original classes: 0=Background, 1=Wall, 2=Terrain, 3=Door
        mask = mask // self.mask_scale

        # Merge terrain into wall if requested
        # Remap: 0=Background, 1=Wall+Terrain, 2=Door (was 3)
        if self.merge_terrain:
            mask[mask == 2] = 1   # Terrain -> Wall
            mask[mask == 3] = 2   # Door -> 2 (shift down)

        grid_size = sample['grid_size']

        # Extract a random tile
        tile_positions = self.extractor.compute_tile_positions(
            image.shape[1], image.shape[0], grid_size
        )

        if not tile_positions:
            print(f"Warning: No valid tiles for {sample['name']}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Select random tile position
        tile_info = random.choice(tile_positions)

        # Extract tiles
        image_tile = self.extractor.extract_tile(image, tile_info, grid_size)
        mask_tile = self.extractor.extract_tile(mask, tile_info, grid_size)

        # Dilate mask for training (expands thin wall lines for stronger gradient signal)
        if self.mask_dilation > 0:
            mask_tile = self._dilate_mask(mask_tile)

        # Apply augmentations (heavier pipeline for Watabou vector art)
        transform = self.watabou_transform if sample['source'] == 'watabou' else self.transform
        if transform is not None:
            transformed = transform(image=image_tile, mask=mask_tile)
            image_tensor = transformed['image']
            mask_tensor = transformed['mask']
            # ToTensorV2 may return tensor or numpy depending on version
            if isinstance(mask_tensor, np.ndarray):
                mask_tensor = torch.from_numpy(mask_tensor)
            mask_tensor = mask_tensor.long()
        else:
            # Basic fallback
            image_tensor = torch.from_numpy(image_tile.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_tile).long()

        result = {
            'image': image_tensor,
            'mask': mask_tensor,
            'name': sample['name'],
        }

        # Add downscaled global image if requested
        if self.global_image_size > 0:
            gs = self.global_image_size
            global_img = np.array(Image.fromarray(image).resize((gs, gs), Image.LANCZOS))
            if self.global_transform is not None:
                global_img = self.global_transform(image=global_img)['image']
            else:
                global_img = torch.from_numpy(global_img.transpose(2, 0, 1)).float() / 255.0
            result['global_image'] = global_img

        return result


class ValidationDataset(Dataset):
    """
    Validation dataset that extracts ALL tiles from each image.

    Used for evaluating model performance on full images with tiled inference.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        metadata_file: Optional[str] = None,
        tile_grid_cells: int = 8,
        tile_size: int = 512,
        overlap: float = 0.5,
        mask_scale: int = 50,
        default_grid_size: int = 140,
    ):
        """
        Args:
            image_dir: Directory containing battlemap images
            mask_dir: Directory containing mask files
            metadata_file: Optional JSON file with per-image metadata
            tile_grid_cells: Number of grid cells per tile
            tile_size: Output tile size in pixels
            overlap: Overlap ratio for tiling
            mask_scale: Scale factor used in mask files
            default_grid_size: Default grid size if not in metadata
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.tile_grid_cells = tile_grid_cells
        self.tile_size = tile_size
        self.overlap = overlap
        self.mask_scale = mask_scale
        self.default_grid_size = default_grid_size

        self.extractor = TileExtractor(tile_grid_cells, tile_size, overlap)

        # Load metadata
        self.metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)

        # Find samples
        self.samples = self._find_samples()

        # Pre-compute all tiles
        self.tiles = []
        self._precompute_tiles()

        print(f"Validation: {len(self.samples)} images, {len(self.tiles)} total tiles")

    def _find_samples(self) -> List[Dict]:
        """Find matching image-mask pairs."""
        samples = []
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
        mask_files = list(self.mask_dir.glob('*_mask_lines.png'))

        for mask_path in mask_files:
            base_name = mask_path.stem.replace('_mask_lines', '')

            image_path = None
            for ext in extensions:
                candidate = self.image_dir / f"{base_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
                candidate = self.image_dir / f"{base_name.replace('_', ' ')}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                continue

            grid_size = self.metadata.get(base_name, {}).get('grid_size', self.default_grid_size)

            samples.append({
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'name': base_name,
                'grid_size': grid_size,
            })

        return samples

    def _precompute_tiles(self):
        """Pre-compute all tile positions for all images."""
        for sample_idx, sample in enumerate(self.samples):
            try:
                image = Image.open(sample['image_path'])
                w, h = image.size
            except Exception as e:
                print(f"Error loading {sample['name']}: {e}")
                continue

            tile_positions = self.extractor.compute_tile_positions(
                w, h, sample['grid_size']
            )

            for tile_idx, tile_info in enumerate(tile_positions):
                self.tiles.append({
                    'sample_idx': sample_idx,
                    'tile_info': tile_info,
                    'tile_idx': tile_idx,
                })

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tile."""
        tile_data = self.tiles[idx]
        sample = self.samples[tile_data['sample_idx']]
        tile_info = tile_data['tile_info']

        # Load image and mask
        image = np.array(Image.open(sample['image_path']).convert('RGB'))
        mask = np.array(Image.open(sample['mask_path']))
        mask = mask // self.mask_scale

        grid_size = sample['grid_size']

        # Extract tile
        image_tile = self.extractor.extract_tile(image, tile_info, grid_size)
        mask_tile = self.extractor.extract_tile(mask, tile_info, grid_size)

        # Convert to tensors (no augmentation for validation)
        image_tensor = torch.from_numpy(image_tile.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask_tile).long()

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'name': sample['name'],
            'sample_idx': tile_data['sample_idx'],
            'tile_idx': tile_data['tile_idx'],
        }


def create_dataloader(
    image_dir: str,
    mask_dir: str,
    batch_size: int = 8,
    tile_grid_cells: int = 8,
    tile_size: int = 512,
    tiles_per_image: int = 4,
    mask_scale: int = 50,
    num_workers: int = 4,
    augment: bool = True,
    shuffle: bool = True,
    metadata_file: Optional[str] = None,
    default_grid_size: int = 140,
) -> DataLoader:
    """
    Create a DataLoader for training.

    Args:
        image_dir: Directory containing battlemap images
        mask_dir: Directory containing mask files
        batch_size: Batch size
        tile_grid_cells: Number of grid cells per tile
        tile_size: Output tile size
        tiles_per_image: Tiles to extract per image per epoch
        mask_scale: Scale factor in mask files
        num_workers: Data loading workers
        augment: Apply augmentations
        shuffle: Shuffle data
        metadata_file: Optional metadata JSON
        default_grid_size: Default grid size

    Returns:
        DataLoader
    """
    dataset = WallSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        metadata_file=metadata_file,
        tile_grid_cells=tile_grid_cells,
        tile_size=tile_size,
        tiles_per_image=tiles_per_image,
        mask_scale=mask_scale,
        augment=augment,
        default_grid_size=default_grid_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    # Test dataset
    import sys

    # Default paths
    image_dir = "data/foundry_to_mask"
    mask_dir = "data/foundry_to_mask/line_masks"

    if len(sys.argv) > 2:
        image_dir = sys.argv[1]
        mask_dir = sys.argv[2]

    print(f"Testing dataset:")
    print(f"  Image dir: {image_dir}")
    print(f"  Mask dir: {mask_dir}")

    dataset = WallSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        tile_grid_cells=8,
        tile_size=512,
        tiles_per_image=2,
        mask_scale=50,
        augment=True,
        default_grid_size=200,  # Crypt uses 200px grid
    )

    if len(dataset) > 0:
        print(f"\nDataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Mask unique values: {torch.unique(sample['mask'])}")
        print(f"Name: {sample['name']}")

        # Test dataloader
        loader = create_dataloader(
            image_dir=image_dir,
            mask_dir=mask_dir,
            batch_size=2,
            num_workers=0,
            default_grid_size=200,
        )

        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        print(f"  Image: {batch['image'].shape}")
        print(f"  Mask: {batch['mask'].shape}")
    else:
        print("No samples found. Check paths.")
