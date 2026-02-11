"""
Batch Foundry Mask Generator

Reads Foundry VTT scene JSON files, locates their background map images from
the Foundry data directory, copies them locally, and generates masks.

Usage:
    python batch_foundry_masks.py
    python batch_foundry_masks.py --hybrid --viz
    python batch_foundry_masks.py --json-dir path/to/jsons --foundry-data path/to/Data
"""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import unquote


# Defaults
DEFAULT_JSON_DIR = Path(__file__).parent.parent / "data" / "foundry_to_mask" / "Map_Jsons"
DEFAULT_IMAGE_DIR = Path(__file__).parent.parent / "data" / "foundry_to_mask" / "Map_Images"
DEFAULT_FOUNDRY_DATA = Path(r"C:\Users\Public\Desktop\FoundryVTT-WindowsPortable-13.348\Data")


def resolve_image_path(json_path: Path, foundry_data: Path) -> tuple[str, Path] | None:
    """Extract background.src from a scene JSON and resolve it to a file path.

    Returns:
        (src_string, resolved_path) or None if not found.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    src = data.get("background", {}).get("src", "")
    if not src:
        return None

    # URL-decode the path and convert forward slashes
    decoded = unquote(src)
    resolved = foundry_data / Path(decoded)

    if resolved.exists():
        return src, resolved

    return None


def main():
    parser = argparse.ArgumentParser(description="Batch process Foundry scenes into masks")
    parser.add_argument(
        "--json-dir",
        default=str(DEFAULT_JSON_DIR),
        help=f"Directory containing scene JSON files (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
        help=f"Directory to copy map images to (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--foundry-data",
        default=str(DEFAULT_FOUNDRY_DATA),
        help=f"Foundry VTT Data directory (default: {DEFAULT_FOUNDRY_DATA})",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Also generate hybrid masks (floor detection) in addition to line masks",
    )
    parser.add_argument(
        "--hybrid-mode",
        choices=["floor", "wall"],
        default="floor",
        help="Hybrid mask mode: floor or wall (default: floor)",
    )
    parser.add_argument("--viz", action="store_true", help="Generate visualization images")
    parser.add_argument(
        "--remove-edge-walls",
        action="store_true",
        default=True,
        help="Remove walls along image edges (default: enabled)",
    )
    parser.add_argument(
        "--no-remove-edge-walls",
        action="store_true",
        help="Disable edge wall removal",
    )

    args = parser.parse_args()

    remove_edge_walls = not args.no_remove_edge_walls

    json_dir = Path(args.json_dir)
    image_dir = Path(args.image_dir)
    foundry_data = Path(args.foundry_data)

    if not json_dir.exists():
        print(f"Error: JSON directory not found: {json_dir}")
        return 1

    if not foundry_data.exists():
        print(f"Error: Foundry Data directory not found: {foundry_data}")
        return 1

    image_dir.mkdir(parents=True, exist_ok=True)

    # Collect all JSON files
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return 1

    print(f"Found {len(json_files)} JSON file(s) in {json_dir}")
    print(f"Foundry Data: {foundry_data}")
    print(f"Image output: {image_dir}")
    print()

    # Phase 1: Resolve and copy images
    scenes = []  # List of (json_path, image_path) tuples
    for jf in json_files:
        result = resolve_image_path(jf, foundry_data)
        if result is None:
            print(f"  SKIP {jf.name}: no background.src or image not found")
            continue

        src, source_path = result
        dest_path = image_dir / source_path.name

        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            print(f"  Copied {source_path.name}")
        else:
            print(f"  Already exists: {source_path.name}")

        scenes.append((jf, dest_path))

    print(f"\n{len(scenes)} scene(s) ready for mask generation\n")
    if not scenes:
        return 0

    # Phase 2: Generate line masks and build name mapping
    # Import here so the script can show errors about missing images before
    # pulling in heavy dependencies (numpy, cv2, PIL)
    from foundry_to_mask_v3 import process_single_file as process_lines, FoundryScene

    # Build a mapping from mask base name -> image filename
    name_mapping = {}

    print("=" * 60)
    print("Generating line masks")
    print("=" * 60)
    for json_path, image_path in scenes:
        try:
            # Derive the safe name the same way process_single_file does
            scene = FoundryScene.from_json(str(json_path))
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in scene.name)
            name_mapping[safe_name] = image_path.name

            process_lines(
                str(json_path),
                mode="lines",
                create_viz=args.viz,
                image_path=str(image_path),
                remove_edge_walls=remove_edge_walls,
            )
        except Exception as e:
            print(f"  ERROR processing {json_path.name}: {e}")
        print()

    # Save name mapping for training dataset
    mapping_path = image_dir / "name_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(name_mapping, f, indent=2)
    print(f"Saved name mapping ({len(name_mapping)} entries) to {mapping_path}")

    # Phase 3: Generate hybrid masks (optional)
    if args.hybrid:
        from foundry_to_mask_hybrid_v3 import process_single_file as process_hybrid

        print("=" * 60)
        print(f"Generating hybrid masks (mode: {args.hybrid_mode})")
        print("=" * 60)
        for json_path, image_path in scenes:
            try:
                process_hybrid(
                    str(json_path),
                    create_viz=args.viz,
                    image_path=str(image_path),
                    mask_mode=args.hybrid_mode,
                    remove_edge_walls=remove_edge_walls,
                )
            except Exception as e:
                print(f"  ERROR processing {json_path.name}: {e}")
            print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
