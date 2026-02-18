"""
Inference script for wall segmentation model.

Runs tile-based prediction on battlemap images and outputs Foundry-compatible
wall segment data.

Usage:
    python -m primary_model_training.inference --checkpoint model.pt --image map.jpg --grid-size 140
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .model import WallSegmentationUNet, SegFormerWrapper, GlobalContextSegFormer
from .tiling import TilePipeline, TileExtractor, TileStitcher
from .config import Config


# Class definitions matching training
CLASS_NAMES = {
    0: 'background',
    1: 'wall',
    2: 'terrain',
    3: 'door',
    4: 'secret_door',
}

# Visualization colors (RGB)
VIZ_COLORS = {
    0: (0, 0, 0),        # Background - black
    1: (255, 0, 0),      # Wall - red
    2: (0, 255, 255),    # Terrain - cyan
    3: (0, 255, 0),      # Door - green
    4: (255, 0, 255),    # Secret door - magenta
}


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device

    Returns:
        Tuple of (model, config_dict) where config_dict has model metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'unet')

    if model_type == 'segformer_gc':
        model = GlobalContextSegFormer(
            num_classes=config.get('num_classes', 3),
            variant=config.get('segformer_variant', 'b0'),
            context_dim=config.get('global_context_dim', 128),
        )
    elif model_type == 'segformer':
        model = SegFormerWrapper(
            num_classes=config.get('num_classes', 3),
            variant=config.get('segformer_variant', 'b0'),
        )
    else:
        model = WallSegmentationUNet(
            in_channels=config.get('in_channels', 3),
            num_classes=config.get('num_classes', 5),
            features=config.get('features', [64, 128, 256, 512]),
            use_aspp=config.get('use_aspp', True),
            use_attention=config.get('use_attention', True),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def _prepare_global_image(
    image: np.ndarray,
    global_image_size: int,
    device: str,
) -> torch.Tensor:
    """Downscale and ImageNet-normalize a full image for global context."""
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(image).resize(
        (global_image_size, global_image_size), PILImage.Resampling.BILINEAR
    )
    t = torch.from_numpy(np.array(pil_img)).float().permute(2, 0, 1) / 255.0
    # ImageNet normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)  # (1, 3, H, W)


def predict_mask(
    model,
    image: np.ndarray,
    grid_size: int,
    tile_grid_cells: int = 8,
    tile_size: int = 512,
    overlap: float = 0.5,
    batch_size: int = 4,
    device: str = 'cuda',
    model_config: Optional[Dict] = None,
) -> np.ndarray:
    """
    Run tile-based prediction on an image.

    Args:
        model: Trained segmentation model
        image: Input image (H, W, C), values 0-255
        grid_size: Grid cell size in pixels
        tile_grid_cells: Grid cells per tile
        tile_size: Model input size
        overlap: Tile overlap ratio
        batch_size: Tiles per batch
        device: Torch device
        model_config: Config dict from checkpoint (for model_type detection)

    Returns:
        Predicted class mask (H, W)
    """
    if model_config is None:
        model_config = {}

    model_type = model_config.get('model_type', 'unet')
    use_imagenet_norm = model_type in ('segformer', 'segformer_gc')

    pipeline = TilePipeline(
        model=model,
        tile_grid_cells=tile_grid_cells,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
        imagenet_norm=use_imagenet_norm,
    )

    # Prepare global context image for segformer_gc
    global_image = None
    if model_type == 'segformer_gc':
        global_image_size = model_config.get('global_image_size', 256)
        global_image = _prepare_global_image(image, global_image_size, device)

    return pipeline.predict(image, grid_size, batch_size, global_image=global_image)


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA)
# ---------------------------------------------------------------------------

# Each entry: (name, forward_transform, inverse_transform)
# Forward transforms operate on (H, W, C) images, inverse on (H, W) masks.
_TTA_TRANSFORMS = [
    ('original',    lambda img: img,                                          lambda m: m),
    ('hflip',       lambda img: np.flip(img, axis=1).copy(),                  lambda m: np.flip(m, axis=1).copy()),
    ('vflip',       lambda img: np.flip(img, axis=0).copy(),                  lambda m: np.flip(m, axis=0).copy()),
    ('hvflip',      lambda img: np.flip(img, axis=(0, 1)).copy(),             lambda m: np.flip(m, axis=(0, 1)).copy()),
    ('rot90',       lambda img: np.rot90(img, k=1).copy(),                    lambda m: np.rot90(m, k=-1).copy()),
    ('rot180',      lambda img: np.rot90(img, k=2).copy(),                    lambda m: np.rot90(m, k=-2).copy()),
    ('rot270',      lambda img: np.rot90(img, k=3).copy(),                    lambda m: np.rot90(m, k=-3).copy()),
    ('rot90_hflip', lambda img: np.flip(np.rot90(img, k=1), axis=1).copy(),   lambda m: np.rot90(np.flip(m, axis=1), k=-1).copy()),
]


def predict_mask_tta(
    model,
    image: np.ndarray,
    grid_size: int,
    tta_passes: int = 8,
    **kwargs,
) -> np.ndarray:
    """
    Run TTA inference: apply geometric augmentations, predict each,
    inverse-transform, and take per-pixel majority vote.

    Args:
        model: Trained segmentation model
        image: Input image (H, W, C), values 0-255
        grid_size: Grid cell size in pixels
        tta_passes: Number of augmentation passes (1-8)
        **kwargs: Forwarded to predict_mask (device, model_config, etc.)

    Returns:
        Predicted class mask (H, W) after majority voting
    """
    transforms = _TTA_TRANSFORMS[:tta_passes]
    masks = []

    for name, fwd, inv in transforms:
        aug_image = fwd(image)
        aug_mask = predict_mask(model, aug_image, grid_size, **kwargs)
        mask = inv(aug_mask)
        masks.append(mask)

    # Majority vote: for each pixel pick the most frequent class
    stacked = np.stack(masks)  # (N, H, W)
    num_classes = int(stacked.max()) + 1
    h, w = masks[0].shape
    votes = np.zeros((num_classes, h, w), dtype=np.int32)
    for c in range(num_classes):
        votes[c] = (stacked == c).sum(axis=0)
    return votes.argmax(axis=0).astype(np.uint8)


def mask_to_visualization(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create a colored visualization of the mask.

    Args:
        mask: Class mask (H, W)
        image: Optional background image for overlay
        alpha: Overlay transparency

    Returns:
        RGB visualization (H, W, 3)
    """
    h, w = mask.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in VIZ_COLORS.items():
        viz[mask == class_id] = color

    if image is not None:
        # Overlay on image
        image = image.astype(np.float32)
        viz = viz.astype(np.float32)

        # Only overlay non-background classes
        overlay_mask = mask > 0
        result = image.copy()
        result[overlay_mask] = image[overlay_mask] * (1 - alpha) + viz[overlay_mask] * alpha

        return result.astype(np.uint8)

    return viz


def mask_to_segments(
    mask: np.ndarray,
    grid_size: int,
    min_length: int = 10,
) -> List[Dict]:
    """
    Convert mask to line segments (simplified extraction).

    This is a basic implementation that extracts horizontal and vertical
    line segments from the mask. A more sophisticated implementation would
    use contour detection and line fitting.

    Args:
        mask: Class mask (H, W)
        grid_size: Grid cell size for coordinate scaling
        min_length: Minimum segment length in pixels

    Returns:
        List of segment dictionaries compatible with Foundry wall format
    """
    segments = []
    segment_id = 0

    # For each non-background class
    for class_id in [1, 2, 3, 4]:  # wall, terrain, door, secret_door
        class_mask = (mask == class_id).astype(np.uint8)

        if class_mask.sum() == 0:
            continue

        # Find contours
        import cv2
        contours, _ = cv2.findContours(
            class_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Simplify contour
            epsilon = 2.0
            approx = cv2.approxPolyDP(contour, epsilon, closed=False)

            # Convert to line segments
            points = approx.reshape(-1, 2)
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]

                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < min_length:
                    continue

                # Map class to Foundry wall properties
                if class_id == 1:  # wall
                    door, sight, light, move = 0, 20, 20, 20
                elif class_id == 2:  # terrain
                    door, sight, light, move = 0, 10, 10, 20
                elif class_id == 3:  # door
                    door, sight, light, move = 1, 20, 20, 20
                elif class_id == 4:  # secret_door
                    door, sight, light, move = 2, 20, 20, 20
                else:
                    continue

                segments.append({
                    '_id': f'wall_{segment_id:06d}',
                    'c': [int(x1), int(y1), int(x2), int(y2)],
                    'door': door,
                    'sight': sight,
                    'light': light,
                    'move': move,
                    'ds': 0,  # door state (closed)
                    'dir': 0,  # direction
                })
                segment_id += 1

    return segments


def export_foundry_walls(
    segments: List[Dict],
    output_path: str,
    scene_name: str = "Generated Scene",
    width: int = 4800,
    height: int = 6000,
    grid_size: int = 140,
):
    """
    Export segments as Foundry VTT compatible JSON.

    Args:
        segments: List of wall segment dictionaries
        output_path: Output JSON path
        scene_name: Scene name
        width: Scene width
        height: Scene height
        grid_size: Grid cell size
    """
    foundry_data = {
        'name': scene_name,
        'width': width,
        'height': height,
        'padding': 0,
        'grid': {
            'size': grid_size,
            'type': 1,
            'distance': 5,
            'units': 'ft',
        },
        'walls': segments,
    }

    with open(output_path, 'w') as f:
        json.dump(foundry_data, f, indent=2)

    print(f"Exported {len(segments)} wall segments to {output_path}")


def process_image(
    checkpoint_path: str,
    image_path: str,
    grid_size: int,
    output_dir: Optional[str] = None,
    save_mask: bool = True,
    save_viz: bool = True,
    save_walls: bool = True,
    tile_grid_cells: int = 8,
    tile_size: int = 512,
    overlap: float = 0.5,
    device: str = 'cuda',
    tta_passes: int = 0,
):
    """
    Process a single image and save outputs.

    Args:
        checkpoint_path: Path to model checkpoint
        image_path: Path to input image
        grid_size: Grid cell size in pixels
        output_dir: Output directory (default: same as input)
        save_mask: Save class mask PNG
        save_viz: Save visualization overlay
        save_walls: Save Foundry wall JSON
        tile_grid_cells: Grid cells per tile
        tile_size: Model input size
        overlap: Tile overlap ratio
        device: Torch device
    """
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, model_config = load_model(checkpoint_path, device)
    print(f"  Model type: {model_config.get('model_type', 'unet')}, "
          f"Classes: {model_config.get('num_classes', '?')}")

    # Load image
    print(f"Processing {image_path}")
    image = np.array(Image.open(image_path).convert('RGB'))
    h, w = image.shape[:2]
    print(f"  Image size: {w}x{h}, Grid size: {grid_size}px")

    # Predict
    predict_kwargs = dict(
        tile_grid_cells=tile_grid_cells,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
        model_config=model_config,
    )

    if tta_passes > 1:
        print(f"  Running TTA prediction ({tta_passes} passes)...")
        mask = predict_mask_tta(
            model, image, grid_size,
            tta_passes=tta_passes,
            **predict_kwargs,
        )
    else:
        print("  Running tile-based prediction...")
        mask = predict_mask(model, image, grid_size, **predict_kwargs)

    # Output directory
    image_path = Path(image_path)
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = image_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = image_path.stem

    # Save outputs
    if save_mask:
        mask_path = out_dir / f"{base_name}_predicted_mask.png"
        # Scale mask values for visibility
        mask_scaled = (mask * 50).astype(np.uint8)
        Image.fromarray(mask_scaled).save(mask_path)
        print(f"  Saved mask: {mask_path}")

    if save_viz:
        viz_path = out_dir / f"{base_name}_predicted_viz.jpg"
        viz = mask_to_visualization(mask, image, alpha=0.5)
        Image.fromarray(viz).save(viz_path, quality=90)
        print(f"  Saved visualization: {viz_path}")

    if save_walls:
        # Extract segments
        segments = mask_to_segments(mask, grid_size)

        # Export
        walls_path = out_dir / f"{base_name}_walls.json"
        export_foundry_walls(
            segments, str(walls_path),
            scene_name=base_name,
            width=w, height=h,
            grid_size=grid_size,
        )

    # Class distribution
    unique, counts = np.unique(mask, return_counts=True)
    print("  Class distribution:")
    for class_id, count in zip(unique, counts):
        pct = count / mask.size * 100
        name = CLASS_NAMES.get(class_id, '?')
        print(f"    {name}: {pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Run wall segmentation inference')

    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', required=True,
                        help='Path to input image')
    parser.add_argument('--grid-size', type=int, required=True,
                        help='Grid cell size in pixels')
    parser.add_argument('--output-dir',
                        help='Output directory (default: same as input)')

    # Output options
    parser.add_argument('--no-mask', action='store_true',
                        help='Do not save mask PNG')
    parser.add_argument('--no-viz', action='store_true',
                        help='Do not save visualization')
    parser.add_argument('--no-walls', action='store_true',
                        help='Do not save Foundry wall JSON')

    # Tiling options
    parser.add_argument('--tile-grid-cells', type=int, default=8,
                        help='Grid cells per tile')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Model input size')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Tile overlap ratio')

    # TTA
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test-time augmentation passes (0=off, 8=full)')

    # Device
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')

    args = parser.parse_args()

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')

    process_image(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        grid_size=args.grid_size,
        output_dir=args.output_dir,
        save_mask=not args.no_mask,
        save_viz=not args.no_viz,
        save_walls=not args.no_walls,
        tile_grid_cells=args.tile_grid_cells,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=device,
        tta_passes=args.tta,
    )


if __name__ == '__main__':
    main()
