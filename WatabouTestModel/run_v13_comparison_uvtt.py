"""
Run v13 comparison tests with UVTT output.

V13 fixes: L-shaped door merge bug in snap_doors_to_wall_endpoints()

Generates:
- Comparison images (3-panel: GT walls, model raw, model post-processed)
- UVTT files from the model inference output
"""

import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "mask_to_walls (old)"))

# Import inference functions
from inference_3class_enhanced import (
    load_model, preprocess_image, predict_3class, post_process_3class
)

# Import mask_to_walls (old) v13
from mask_to_walls_v13 import (
    calculate_params_from_scale,
    extract_walls_from_wall_mask,
    extract_doors,
    add_perimeter_walls,
    straighten_walls,
    smart_corner_extension,
    extend_doors_to_walls,
    filter_doors_crossing_walls,
    split_walls_at_doors,
    snap_doors_to_wall_endpoints,
    filter_zero_length_segments,
    filter_short_segments,
    merge_dangling_endpoints
)

# Import UVTT converter
from walls_to_uvtt import create_uvtt


def run_mask_to_walls_v13(mask, grid_size=70, epsilon=None):
    """Run v13 pipeline on a mask image."""
    params = calculate_params_from_scale(grid_size, epsilon_override=epsilon)

    walls = extract_walls_from_wall_mask(mask, params)
    doors = extract_doors(mask, params)

    # V12: Add perimeter walls
    walls = add_perimeter_walls(walls, mask.shape)

    all_segments = walls + doors

    all_segments = straighten_walls(all_segments, params)
    all_segments = smart_corner_extension(all_segments, params)
    all_segments = extend_doors_to_walls(all_segments, params)
    all_segments = filter_doors_crossing_walls(all_segments, max_distance=8)
    all_segments = split_walls_at_doors(all_segments, params)
    # V13: snap_doors_to_wall_endpoints now includes validation
    all_segments = snap_doors_to_wall_endpoints(all_segments, snap_distance=8)
    all_segments = filter_zero_length_segments(all_segments)
    all_segments = filter_short_segments(all_segments, params['min_wall_length'])

    # V12: Merge dangling wall endpoints only
    all_segments = merge_dangling_endpoints(all_segments, merge_distance=25)

    return all_segments


def create_comparison_image(map_name, original_image, gt_segments, model_segments, output_path):
    """Create 3-panel comparison image with walls overlaid on original."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    h, w = original_image.shape[:2]

    # Panel 1: Ground Truth walls on original
    axes[0].imshow(original_image)
    for seg in gt_segments:
        color = 'red' if seg['type'] == 'door' else 'blue'
        axes[0].plot([seg['x1'], seg['x2']], [seg['y1'], seg['y2']],
                     color=color, linewidth=1.5, alpha=0.8)
    axes[0].set_title(f"Ground Truth\n{len([s for s in gt_segments if s['type']=='wall'])} walls, "
                      f"{len([s for s in gt_segments if s['type']=='door'])} doors", fontsize=12)
    axes[0].axis('off')

    # Panel 2: Model walls on original
    axes[1].imshow(original_image)
    for seg in model_segments:
        color = 'red' if seg['type'] == 'door' else 'blue'
        axes[1].plot([seg['x1'], seg['x2']], [seg['y1'], seg['y2']],
                     color=color, linewidth=1.5, alpha=0.8)
    axes[1].set_title(f"Model (v13)\n{len([s for s in model_segments if s['type']=='wall'])} walls, "
                      f"{len([s for s in model_segments if s['type']=='door'])} doors", fontsize=12)
    axes[1].axis('off')

    # Panel 3: Side-by-side wall visualization (black background)
    wall_vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw GT in blue
    for seg in gt_segments:
        draw_line_on_image(wall_vis, seg, (100, 100, 255) if seg['type'] == 'wall' else (255, 100, 100))

    # Draw model in green/yellow
    for seg in model_segments:
        draw_line_on_image(wall_vis, seg, (100, 255, 100) if seg['type'] == 'wall' else (255, 255, 100))

    axes[2].imshow(wall_vis)
    axes[2].set_title(f"Overlay (Blue=GT, Green=Model)\nWalls & Doors", fontsize=12)
    axes[2].axis('off')

    plt.suptitle(f"{map_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def draw_line_on_image(img, seg, color, thickness=2):
    """Draw a line segment on an image array."""
    cv2.line(img,
             (int(seg['x1']), int(seg['y1'])),
             (int(seg['x2']), int(seg['y2'])),
             color, thickness)


def main():
    import torch

    # Configuration
    data_dir = Path("data/watabou_3class_filtered")
    output_dir = Path("../results/comparison_v13")
    checkpoint = "saved_models_3class/3class_aten_multi_1450.pth.tar"
    grid_size = 70
    num_samples = 15

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all available maps
    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    all_maps = [f.stem for f in image_dir.glob("*.png")]
    print(f"Found {len(all_maps)} maps total")

    # Select random maps
    random.seed(42)  # For reproducibility
    selected_maps = random.sample(all_maps, min(num_samples, len(all_maps)))
    print(f"Selected {len(selected_maps)} maps for testing")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = load_model(checkpoint, device)

    # Process each map
    for i, map_name in enumerate(selected_maps):
        print(f"\n[{i+1}/{len(selected_maps)}] Processing: {map_name}")

        image_path = image_dir / f"{map_name}.png"
        gt_mask_path = mask_dir / f"{map_name}.png"

        # Load original image
        original_image = np.array(Image.open(image_path).convert('RGB'))

        # Load ground truth mask
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))

        # Run inference
        print("  Running inference...")
        image_tensor, _, original_size = preprocess_image(str(image_path))
        prediction_raw = predict_3class(model, image_tensor, device, original_size)
        prediction_processed = post_process_3class(prediction_raw)

        # Convert prediction to mask format (0=walls, 127=doors, 255=floor)
        model_mask = np.zeros_like(prediction_processed, dtype=np.uint8)
        model_mask[prediction_processed == 0] = 0
        model_mask[prediction_processed == 1] = 127
        model_mask[prediction_processed == 2] = 255

        # Run mask_to_walls (old) on both
        print("  Extracting walls (GT)...")
        gt_segments = run_mask_to_walls_v13(gt_mask, grid_size)

        print("  Extracting walls (model)...")
        model_segments = run_mask_to_walls_v13(model_mask, grid_size)

        # Create comparison image
        print("  Creating comparison image...")
        comparison_path = output_dir / f"{map_name}_comparison.png"
        create_comparison_image(map_name, original_image, gt_segments, model_segments, comparison_path)

        # Create UVTT file from model output
        print("  Creating UVTT file...")
        uvtt_path = output_dir / f"{map_name}.uvtt"

        segments_data = {
            'segments': model_segments,
            'image_width': model_mask.shape[1],
            'image_height': model_mask.shape[0]
        }

        create_uvtt(segments_data, str(image_path), grid_size, str(uvtt_path))

        print(f"  Saved: {comparison_path.name}, {uvtt_path.name}")

    print(f"\n{'='*60}")
    print(f"Completed! Output saved to: {output_dir}")
    print(f"  - {len(selected_maps)} comparison images")
    print(f"  - {len(selected_maps)} UVTT files")


if __name__ == "__main__":
    main()
