"""
Run stock CubiCasa5k floor plan model on battlemaps.
Outputs side-by-side: original | room prediction | wall/door heatmaps.

CubiCasa5k output channels (44 total):
  Channels 0-20: heatmaps (sigmoid), for icon/boundary detection
  Channels 21-43: room type logits

Room classes (12): Background, Outdoor, Wall, Kitchen, Living Room,
                   Bedroom, Bath, Hallway, Railing, Storage, Garage, Other
Icon classes (11): Empty, Window, Door, Closet, Electr.Appl., Toilet,
                   Sink, Sauna bench, Fire Place, Bathtub, Chimney

The first 21 channels are split as:
  0-10: icon heatmaps (Empty, Window, Door, Closet, Electr.Appl., Toilet,
                        Sink, Sauna bench, Fire Place, Bathtub, Chimney)
  11-20: boundary heatmaps (some for walls, openings, etc.)
  21-32: room type logits (12 classes)
  33-43: icon type logits (11 classes)
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

# Add this directory so we can import the model
sys.path.insert(0, str(Path(__file__).parent))

# Patch out the floortrans.models import that hg_furukawa_original.py needs
# (only used in init_weights, which we skip since we load pretrained)
import types
floortrans = types.ModuleType('floortrans')
floortrans.models = types.ModuleType('floortrans.models')
floortrans.models.model_1427 = types.ModuleType('floortrans.models.model_1427')
sys.modules['floortrans'] = floortrans
sys.modules['floortrans.models'] = floortrans.models
sys.modules['floortrans.models.model_1427'] = floortrans.models.model_1427

from hg_furukawa_original import hg_furukawa_original

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
FOUNDRY_DIR = PROJECT_ROOT / "data" / "foundry_to_mask" / "Map_Images"
WATABOU_DIR = PROJECT_ROOT / "data" / "watabou_to_mask" / "watabou_images"
OUTPUT_DIR = SCRIPT_DIR / "test_output"
WEIGHTS_PATH = SCRIPT_DIR / "model_best_val_loss_var.pkl"

# Class labels
ROOM_CLASSES = [
    "Background", "Outdoor", "Wall", "Kitchen", "Living Room",
    "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other"
]
ICON_CLASSES = [
    "Empty", "Window", "Door", "Closet", "Electr.Appl.", "Toilet",
    "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"
]

ROOM_COLORS = [
    (0, 0, 0),        # Background - black
    (128, 128, 0),    # Outdoor - olive
    (255, 0, 0),      # Wall - red
    (255, 165, 0),    # Kitchen - orange
    (0, 128, 255),    # Living Room - blue
    (128, 0, 255),    # Bedroom - purple
    (0, 200, 200),    # Bath - teal
    (200, 200, 200),  # Hallway - light gray
    (255, 255, 0),    # Railing - yellow
    (128, 64, 0),     # Storage - brown
    (64, 64, 64),     # Garage - dark gray
    (255, 128, 128),  # Other - pink
]

ICON_COLORS = [
    (0, 0, 0),        # Empty
    (0, 255, 255),    # Window - cyan
    (0, 255, 0),      # Door - green
    (255, 0, 255),    # Closet - magenta
    (255, 255, 0),    # Electr.Appl. - yellow
    (128, 128, 255),  # Toilet - light blue
    (0, 128, 128),    # Sink - dark teal
    (200, 100, 0),    # Sauna bench - dark orange
    (255, 64, 0),     # Fire Place - red-orange
    (64, 128, 255),   # Bathtub - medium blue
    (128, 0, 0),      # Chimney - dark red
]


def load_model(device):
    """Load CubiCasa5k pretrained model."""
    model = hg_furukawa_original(n_classes=44)
    ckpt = torch.load(str(WEIGHTS_PATH), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model


def preprocess(image_path, max_dim=1024):
    """Load and preprocess image. Returns (image_np_resized, tensor)."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # Resize to fit in max_dim
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    image_np = np.array(image)

    # Normalize to [0, 1] and convert to tensor
    tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0
    return image_np, tensor.unsqueeze(0)


def run_inference(model, tensor, device):
    """Run model and return parsed outputs."""
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)  # (1, 44, H, W)

    output = output.squeeze(0).cpu()

    # Split outputs
    icon_heatmaps = output[:11].numpy()     # (11, H, W) - sigmoid activated
    boundary_heatmaps = output[11:21].numpy()  # (10, H, W) - sigmoid activated
    room_logits = output[21:33]              # (12, H, W)
    icon_logits = output[33:44]              # (11, H, W)

    room_pred = room_logits.argmax(dim=0).numpy()  # (H, W)
    icon_pred = icon_logits.argmax(dim=0).numpy()   # (H, W)

    return icon_heatmaps, boundary_heatmaps, room_pred, icon_pred


def make_visualization(image_np, icon_heatmaps, boundary_heatmaps, room_pred, icon_pred):
    """Create visualization panels."""
    h, w = room_pred.shape

    # Resize image to match prediction size
    if image_np.shape[:2] != (h, w):
        image_np = np.array(Image.fromarray(image_np).resize((w, h), Image.LANCZOS))

    # Panel 1: Room type prediction
    room_viz = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(ROOM_COLORS):
        room_viz[room_pred == cls_id] = color

    # Panel 2: Wall heatmap (boundary channel most likely to be walls)
    # Try all boundary heatmaps, pick the one with most activation
    best_boundary = None
    best_sum = -1
    for i in range(boundary_heatmaps.shape[0]):
        s = boundary_heatmaps[i].sum()
        if s > best_sum:
            best_sum = s
            best_boundary = i

    # Combine wall-related signals:
    # - Room class 2 = Wall
    # - Door heatmap (icon channel 2)
    # - Window heatmap (icon channel 1)
    wall_from_rooms = (room_pred == 2).astype(np.float32)
    door_heatmap = icon_heatmaps[2]  # Door
    window_heatmap = icon_heatmaps[1]  # Window

    # Wall+door+window overlay on image
    overlay = image_np.copy().astype(np.float32)
    # Wall areas in red
    wall_mask = wall_from_rooms > 0.5
    overlay[wall_mask] = overlay[wall_mask] * 0.4 + np.array([255, 0, 0], dtype=np.float32) * 0.6
    # Door heatmap in green
    door_mask = door_heatmap > 0.3
    overlay[door_mask] = overlay[door_mask] * 0.4 + np.array([0, 255, 0], dtype=np.float32) * 0.6
    # Window heatmap in cyan
    window_mask = window_heatmap > 0.3
    overlay[window_mask] = overlay[window_mask] * 0.4 + np.array([0, 255, 255], dtype=np.float32) * 0.6
    overlay = overlay.clip(0, 255).astype(np.uint8)

    # Panel 3: Best boundary heatmap as grayscale
    boundary_viz = (boundary_heatmaps[best_boundary] * 255).clip(0, 255).astype(np.uint8)
    boundary_viz = np.stack([boundary_viz] * 3, axis=-1)

    # Stack: original | overlay | room types | boundary heatmap
    panel = np.concatenate([image_np, overlay, room_viz, boundary_viz], axis=1)
    return panel


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading CubiCasa5k model (44 classes)...")
    model = load_model(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Pick same 5 Foundry + 5 Watabou as the SegFormer test (seed=42)
    random.seed(42)
    foundry_images = sorted(FOUNDRY_DIR.glob("*"))
    foundry_images = [f for f in foundry_images if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    watabou_images = sorted(WATABOU_DIR.glob("*.png"))

    selected_foundry = random.sample(foundry_images, min(5, len(foundry_images)))
    selected_watabou = random.sample(watabou_images, min(5, len(watabou_images)))

    all_images = [(p, "foundry") for p in selected_foundry] + [(p, "watabou") for p in selected_watabou]

    for img_path, source in all_images:
        print(f"\n{'='*60}")
        print(f"[{source.upper()}] {img_path.name}")

        image_np, tensor = preprocess(img_path)
        icon_hm, boundary_hm, room_pred, icon_pred = run_inference(model, tensor, device)

        # Print detected classes
        print("  Room classes detected:")
        for cls_id in range(len(ROOM_CLASSES)):
            count = np.sum(room_pred == cls_id)
            if count > 0:
                pct = count / room_pred.size * 100
                print(f"    {ROOM_CLASSES[cls_id]}: {pct:.1f}%")

        print("  Icon heatmap activations (>0.3):")
        for i, name in enumerate(ICON_CLASSES):
            activated = np.sum(icon_hm[i] > 0.3) / icon_hm[i].size * 100
            if activated > 0.1:
                print(f"    {name}: {activated:.1f}% pixels")

        print("  Boundary heatmap activations (>0.3):")
        for i in range(boundary_hm.shape[0]):
            activated = np.sum(boundary_hm[i] > 0.3) / boundary_hm[i].size * 100
            if activated > 0.1:
                print(f"    Channel {i}: {activated:.1f}% pixels")

        panel = make_visualization(image_np, icon_hm, boundary_hm, room_pred, icon_pred)

        # Save
        panel_img = Image.fromarray(panel)
        max_width = 5120
        if panel_img.width > max_width:
            ratio = max_width / panel_img.width
            panel_img = panel_img.resize((max_width, int(panel_img.height * ratio)), Image.LANCZOS)

        out_name = f"{source}_{img_path.stem}.jpg"
        panel_img.save(OUTPUT_DIR / out_name, quality=85)
        print(f"  Saved: {OUTPUT_DIR / out_name}")

    print(f"\nDone! Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
