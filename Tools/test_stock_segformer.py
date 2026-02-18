"""
Run stock ADE20K SegFormer-B0 on battlemaps to see what it picks up.
Outputs side-by-side visualizations: original | ADE20K prediction | wall-relevant classes only.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Paths
FOUNDRY_DIR = Path("data/foundry_to_mask/Map_Images")
WATABOU_DIR = Path("data/watabou_to_mask/watabou_images")
OUTPUT_DIR = Path("outputs/stock_segformer_test")
MODEL_PATH = Path("primary_model_training/pretrained/segformer-b0")

# ADE20K classes relevant to our wall segmentation task
RELEVANT_CLASSES = {
    0: ("wall", (255, 0, 0)),         # red
    14: ("door", (0, 255, 0)),        # green
    32: ("fence", (0, 255, 255)),     # cyan (terrain-like)
    38: ("railing", (255, 165, 0)),   # orange (terrain-like)
    8: ("windowpane", (255, 0, 255)), # magenta
    3: ("floor", (64, 64, 64)),       # dark gray
    95: ("bannister", (255, 255, 0)), # yellow
}

# Full ADE20K color palette (random but deterministic)
rng = np.random.RandomState(42)
ADE20K_COLORS = rng.randint(0, 256, (150, 3), dtype=np.uint8)
# Override relevant ones for consistency
for cls_id, (name, color) in RELEVANT_CLASSES.items():
    ADE20K_COLORS[cls_id] = color


def run_inference(model, processor, image_path, device, max_dim=1024):
    """Run stock SegFormer on a single image, return predicted class mask."""
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Resize large images to avoid OOM (battlemaps can be 5000+ px)
    scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        image_resized = image.resize((new_w, new_h), Image.LANCZOS)
    else:
        image_resized = image

    res_w, res_h = image_resized.size

    inputs = processor(images=image_resized, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, 150, H/4, W/4)
    logits = F.interpolate(logits, size=(res_h, res_w),
                           mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1).squeeze().cpu().numpy()

    # Return image at same resolution as pred for visualization
    image_np = np.array(image_resized)
    return image_np, pred


def make_visualization(image, pred, name):
    """Create side-by-side: original | full ADE20K | relevant classes only."""
    h, w = pred.shape

    # Full ADE20K colorized
    full_viz = ADE20K_COLORS[pred]  # (H, W, 3)

    # Relevant classes only
    relevant_viz = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, (cls_name, color) in RELEVANT_CLASSES.items():
        mask = pred == cls_id
        relevant_viz[mask] = color

    # Overlay relevant classes on original image
    overlay = image.copy().astype(np.float32)
    relevant_mask = np.zeros((h, w), dtype=bool)
    for cls_id in RELEVANT_CLASSES:
        if cls_id == 3:  # skip floor for overlay
            continue
        relevant_mask |= (pred == cls_id)

    overlay[relevant_mask] = overlay[relevant_mask] * 0.4 + relevant_viz[relevant_mask].astype(np.float32) * 0.6
    overlay = overlay.astype(np.uint8)

    # Build class legend
    detected = {}
    for cls_id, (cls_name, color) in RELEVANT_CLASSES.items():
        count = np.sum(pred == cls_id)
        if count > 0:
            pct = count / pred.size * 100
            detected[cls_name] = (color, pct)

    # Stack horizontally: original | overlay | relevant-only
    panel = np.concatenate([image, overlay, relevant_viz], axis=1)

    return panel, detected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load stock model (all 150 ADE20K classes)
    print("Loading stock SegFormer-B0 (ADE20K 150 classes)...")
    model = SegformerForSemanticSegmentation.from_pretrained(str(MODEL_PATH))
    model.to(device).eval()
    processor = SegformerImageProcessor.from_pretrained(str(MODEL_PATH))

    # Pick 5 Foundry + 5 Watabou
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

        image, pred = run_inference(model, processor, img_path, device)
        panel, detected = make_visualization(image, pred, img_path.stem)

        # Print detected relevant classes
        if detected:
            print("  Relevant ADE20K classes detected:")
            for cls_name, (color, pct) in sorted(detected.items(), key=lambda x: -x[1][1]):
                print(f"    {cls_name}: {pct:.1f}%")
        else:
            print("  No relevant classes detected")

        # Save — resize panel if too wide
        max_width = 4096
        panel_img = Image.fromarray(panel)
        if panel_img.width > max_width:
            ratio = max_width / panel_img.width
            panel_img = panel_img.resize((max_width, int(panel_img.height * ratio)), Image.LANCZOS)

        out_name = f"{source}_{img_path.stem}.jpg"
        panel_img.save(OUTPUT_DIR / out_name, quality=85)
        print(f"  Saved: {OUTPUT_DIR / out_name}")

    print(f"\nDone! Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
