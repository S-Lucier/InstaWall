"""
Visualize what the texture-prototype model receives during training.

Produces a single figure per sample showing:
  - Row 1: image tile | raw mask | dilated mask (if dilation > 0)
  - Row 2: wall crop 0 | wall crop 1 | ... | door crop 0 | door crop 1 | ...

Usage:
    python -m primary_model_training.visualize_training_input
    python -m primary_model_training.visualize_training_input \
        --map Brazenthrone_-_The_Old_Palace_1F \
        --n-samples 4 \
        --output outputs/training_input_viz.png
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import torch

from .config import Config
from .dataset import WallSegmentationDataset


# ── visual constants ────────────────────────────────────────────────────────

# Class colours in 0-1 RGB for matplotlib
CLASS_COLORS = {
    0: (0.08, 0.08, 0.08),   # background  — near-black
    1: (0.95, 0.20, 0.20),   # wall        — red
    2: (0.10, 0.85, 0.85),   # terrain     — cyan
    3: (0.15, 0.90, 0.15),   # door        — green
}
CLASS_NAMES = {0: 'background', 1: 'wall', 2: 'terrain', 3: 'door'}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denorm(t: torch.Tensor) -> np.ndarray:
    """ImageNet-denormalise a (3, H, W) tensor → (H, W, 3) uint8."""
    arr = t.permute(1, 2, 0).cpu().numpy()
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray, alpha_bg: float = 0.35) -> np.ndarray:
    """Convert a class-index mask (H, W) → RGBA (H, W, 4) image."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        where = mask == cls
        rgba[where, :3] = color
        rgba[where,  3] = alpha_bg if cls == 0 else 1.0
    return rgba


def overlay_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray,
                           alpha: float = 0.45) -> np.ndarray:
    """Blend coloured mask over image (both uint8 H×W×3)."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        where = mask == cls
        overlay[where] = [c * 255 for c in color]
    blended = image_rgb.astype(np.float32).copy()
    fg = mask > 0
    blended[fg] = (1 - alpha) * image_rgb[fg] + alpha * overlay[fg]
    return blended.astype(np.uint8)


def mark_crop_center(ax, cy: int, cx: int, crop_size: int, color: str):
    """Draw a small crosshair + square on ax at the crop location."""
    half = crop_size // 2
    rect = plt.Rectangle(
        (cx - half, cy - half), crop_size, crop_size,
        linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--',
    )
    ax.add_patch(rect)
    ax.plot(cx, cy, '+', color=color, markersize=8, markeredgewidth=1.5)


# ── dataset helper ──────────────────────────────────────────────────────────

def _build_dataset(map_name: str | None, merge_terrain: bool,
                   mask_dilation: int, texture_crop_size: int,
                   texture_crops_per_class: int) -> WallSegmentationDataset:
    ds = WallSegmentationDataset(
        image_dir='data/foundry_to_mask/Map_Images',
        mask_dir='data/foundry_to_mask/line_masks',
        metadata_file='data/foundry_to_mask/grid_metadata.json',
        tile_grid_cells=8,
        tile_size=512,
        tiles_per_image=4,
        mask_scale=50,
        augment=False,          # no colour jitter — we want the raw appearance
        merge_terrain=merge_terrain,
        use_imagenet_norm=True,
        mask_dilation=mask_dilation,
        tile_context_cells=1,
        sample_texture_crops=False,  # we'll call _sample_texture_crops directly
        texture_crop_size=texture_crop_size,
        texture_crops_per_class=texture_crops_per_class,
    )
    if map_name:
        ds.active_samples = [s for s in ds.active_samples if map_name in s['name']]
        if not ds.active_samples:
            raise ValueError(f"Map '{map_name}' not found. "
                             f"Available: {[s['name'] for s in ds.all_samples[:10]]}")
    return ds


# ── per-sample figure ────────────────────────────────────────────────────────

def visualize_sample(sample: dict, texture_crop_size: int,
                     texture_crops_per_class: int,
                     merge_terrain: bool) -> plt.Figure:
    image_t     = sample['image']          # (3, H, W) normalised
    mask_t      = sample['mask']           # (H, W) long
    wall_crops  = sample['wall_crops']     # (N, 3, K, K)
    door_crops  = sample['door_crops']     # (N, 3, K, K)
    name        = sample['name']

    image_rgb = denorm(image_t)            # (H, W, 3) uint8
    mask_np   = mask_t.numpy()            # (H, W) int

    N = texture_crops_per_class
    K = texture_crop_size
    wall_absent = wall_crops.abs().sum() == 0
    door_absent = door_crops.abs().sum() == 0

    # ── layout ──────────────────────────────────────────────────────────────
    # Top row : image | mask overlay | class legend
    # Bottom row: wall crop 0 … N-1 | door crop 0 … N-1
    n_crop_cols = N * 2          # wall crops + door crops
    n_top_cols  = max(n_crop_cols, 2)

    fig = plt.figure(figsize=(max(n_top_cols * 2.8, 8), 7.5), dpi=120)
    fig.patch.set_facecolor('#1a1a2e')

    gs = GridSpec(
        2, max(n_top_cols, 2),
        figure=fig,
        hspace=0.35, wspace=0.08,
        top=0.91, bottom=0.04, left=0.02, right=0.98,
    )

    # ── top-left: image tile ─────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, :n_top_cols // 2])
    ax_img.imshow(image_rgb)
    ax_img.set_title('Image tile (512 px, ImageNet-norm reversed)',
                     color='white', fontsize=8, pad=4)
    ax_img.axis('off')

    # ── top-right: mask overlay ───────────────────────────────────────────────
    ax_mask = fig.add_subplot(gs[0, n_top_cols // 2:])
    ax_mask.imshow(overlay_mask_on_image(image_rgb, mask_np, alpha=0.55))

    # Draw legend patches
    present = sorted(set(mask_np.flatten().tolist()))
    patches = [
        mpatches.Patch(color=CLASS_COLORS[c],
                       label=CLASS_NAMES.get(c, str(c)),
                       alpha=0.9)
        for c in present
    ]
    ax_mask.legend(handles=patches, loc='upper right', fontsize=6,
                   facecolor='#22223b', labelcolor='white',
                   framealpha=0.85, edgecolor='none')
    ax_mask.set_title('Mask overlay (training target)', color='white', fontsize=8, pad=4)
    ax_mask.axis('off')

    # ── bottom row: texture crops ─────────────────────────────────────────────
    crop_axes = [fig.add_subplot(gs[1, i]) for i in range(n_crop_cols)]

    def _show_crop(ax, crop_t, label, border_color, absent):
        crop_rgb = denorm(crop_t)
        if absent:
            # Grey out and annotate
            grey = np.full_like(crop_rgb, 40)
            ax.imshow(grey)
            ax.text(0.5, 0.5, 'absent\n(zeroed)', transform=ax.transAxes,
                    ha='center', va='center', color='#888888', fontsize=7,
                    style='italic')
        else:
            ax.imshow(crop_rgb)

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, color='white', fontsize=7, pad=3)
        # Draw K×K marker in corner
        ax.text(0.02, 0.97, f'{K}×{K} px', transform=ax.transAxes,
                ha='left', va='top', color='white', fontsize=5.5,
                bbox=dict(facecolor='black', alpha=0.5, pad=1.5, edgecolor='none'))

    wall_color = '#ff4040'
    door_color = '#30e030'

    for n in range(N):
        _show_crop(crop_axes[n],
                   wall_crops[n],
                   f'Wall crop {n + 1}',
                   wall_color, bool(wall_absent))

    for n in range(N):
        _show_crop(crop_axes[N + n],
                   door_crops[n],
                   f'Door crop {n + 1}',
                   door_color, bool(door_absent))

    # Divider label
    mid = N - 0.5
    fig.text(
        0.5, 0.025,
        f'Texture crops — {N} wall (red border) + {N} door (green border)   '
        f'│   Crop size: {K}×{K} px in 512-px tile space   '
        f'│   Sampled from undilated mask, normalized identically to tile',
        ha='center', va='bottom', color='#aaaacc', fontsize=6.5,
    )

    # ── figure title ──────────────────────────────────────────────────────────
    merge_note = '  [merge_terrain=True → 3 classes]' if merge_terrain else ''
    fig.suptitle(
        f'Training input — segformer_texture   │   {name}{merge_note}',
        color='white', fontsize=9, fontweight='bold', y=0.97,
    )

    return fig


# ── entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize training inputs for segformer_texture'
    )
    parser.add_argument('--map', default=None,
                        help='Substring of map name to filter (default: random)')
    parser.add_argument('--n-samples', type=int, default=3,
                        help='Number of tiles to visualize (default: 3)')
    parser.add_argument('--texture-crop-size', type=int, default=64,
                        help='Texture crop side length (default: 64)')
    parser.add_argument('--texture-crops-per-class', type=int, default=2,
                        help='Crops per class (default: 2)')
    parser.add_argument('--mask-dilation', type=int, default=0,
                        help='Mask dilation radius for comparison (default: 0)')
    parser.add_argument('--merge-terrain', action='store_true',
                        help='Use merge_terrain=True (3 classes)')
    parser.add_argument('--output', default='outputs/training_input_viz.png',
                        help='Output PNG path (default: outputs/training_input_viz.png)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f'Building dataset...')
    ds = _build_dataset(
        map_name=args.map,
        merge_terrain=args.merge_terrain,
        mask_dilation=args.mask_dilation,
        texture_crop_size=args.texture_crop_size,
        texture_crops_per_class=args.texture_crops_per_class,
    )
    print(f'  Active samples: {len(ds.active_samples)}')

    from .tiling import TileExtractor
    from PIL import Image as PILImage

    extractor = TileExtractor(8, 512, overlap=0.0, context_cells=1)
    IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # For each sample, pick a map that has at least one interesting tile
    # (wall or door pixels present). Retry up to 20 times per map.
    pool = list(ds.active_samples)
    random.shuffle(pool)

    figs = []
    attempts = 0
    while len(figs) < args.n_samples and pool and attempts < 200:
        attempts += 1
        sample_meta = random.choice(pool)

        image_raw = np.array(PILImage.open(sample_meta['image_path']).convert('RGB'))
        mask_raw  = np.array(PILImage.open(sample_meta['mask_path'])) // ds.mask_scale
        if args.merge_terrain:
            mask_raw[mask_raw == 2] = 1
            mask_raw[mask_raw == 3] = 2
        grid_size = sample_meta['grid_size']

        tile_positions = extractor.compute_tile_positions(
            image_raw.shape[1], image_raw.shape[0], grid_size
        )
        if not tile_positions:
            continue
        tile_info = random.choice(tile_positions)

        image_tile = extractor.extract_tile(image_raw, tile_info, grid_size)  # (512,512,3) uint8
        mask_tile  = extractor.extract_tile(mask_raw,  tile_info, grid_size)  # (512,512) int

        # Skip background-only tiles so every frame is informative
        if mask_tile.max() == 0:
            continue

        wall_class = 1
        door_class = 2 if args.merge_terrain else 3

        wall_crops = ds._sample_texture_crops(
            image_tile, mask_tile, wall_class,
            args.texture_crops_per_class, args.texture_crop_size,
        )
        door_crops = ds._sample_texture_crops(
            image_tile, mask_tile, door_class,
            args.texture_crops_per_class, args.texture_crop_size,
        )

        # Dilate mask (if requested) for the mask overlay display
        if ds.mask_dilation > 0:
            mask_display = ds._dilate_mask(mask_tile)
        else:
            mask_display = mask_tile

        # ImageNet-normalise image tile → tensor
        image_t = torch.from_numpy(image_tile.astype(np.float32) / 255.0).permute(2, 0, 1)
        image_t = (image_t - IMAGENET_MEAN_T) / IMAGENET_STD_T

        wall_abs = wall_crops.abs().sum() == 0
        door_abs = door_crops.abs().sum() == 0

        print(f'  Sample {len(figs)+1}/{args.n_samples}: {sample_meta["name"]}  '
              f'mask_classes={sorted(set(mask_tile.flatten().tolist()))}  '
              f'wall_absent={bool(wall_abs)}  door_absent={bool(door_abs)}')

        sample_dict = {
            'image':      image_t,
            'mask':       torch.from_numpy(mask_display).long(),
            'wall_crops': wall_crops,
            'door_crops': door_crops,
            'name':       sample_meta['name'],
        }

        fig = visualize_sample(
            sample_dict,
            texture_crop_size=args.texture_crop_size,
            texture_crops_per_class=args.texture_crops_per_class,
            merge_terrain=args.merge_terrain,
        )
        figs.append(fig)

    if not figs:
        print('No suitable tiles found — try a different map or seed.')
        return

    # Stack figures vertically into one output file
    if len(figs) == 1:
        figs[0].savefig(args.output, dpi=120, bbox_inches='tight',
                        facecolor=figs[0].get_facecolor())
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import PIL.Image as PILImage

        strips = []
        for fig in figs:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            # buffer_rgba returns RGBA; drop alpha channel to get RGB
            buf_rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            strips.append(buf_rgba[:, :, :3])
            plt.close(fig)

        combined = np.vstack(strips)
        PILImage.fromarray(combined).save(args.output, quality=92)

    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
