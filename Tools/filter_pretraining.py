#!/usr/bin/env python3
"""
Filter pre-training images based on tags.

Moves images to appropriate folders:
- discards/: Images tagged with 'discard'
- outdoors/: Pure outdoor images (outdoor tag without structure tags)

Each folder gets its own tags.json tracking the moved images.
"""

import json
import shutil
import argparse
from pathlib import Path

# Structure tags that indicate an outdoor map has walls
STRUCTURE_TAGS = {'dungeon', 'cave', 'ship', 'building', 'urban'}


def load_tags(tags_file):
    """Load tags from JSON file."""
    if tags_file.exists():
        with open(tags_file) as f:
            return json.load(f)
    return {}


def save_tags(tags_file, tags):
    """Save tags to JSON file."""
    tags_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tags_file, 'w') as f:
        json.dump(tags, f, indent=2)


def is_pure_outdoor(image_tags):
    """Check if image is pure outdoor (no structure tags)."""
    tag_set = set(image_tags)
    if 'outdoor' not in tag_set:
        return False
    # Check if any structure tags present
    return not tag_set.intersection(STRUCTURE_TAGS)


def filter_images(data_dir, dry_run=False):
    """
    Filter images based on tags.

    Args:
        data_dir: Path to pretraining_data directory
        dry_run: If True, only print what would be done
    """
    data_dir = Path(data_dir)
    tags_file = data_dir / 'tags.json'

    if not tags_file.exists():
        print(f"Error: Tags file not found: {tags_file}")
        return

    # Load existing tags
    all_tags = load_tags(tags_file)
    print(f"Loaded {len(all_tags)} tagged images")

    # Setup output directories
    discards_dir = data_dir / 'discards'
    outdoors_dir = data_dir / 'outdoors'

    # Track moved images
    discard_tags = {}
    outdoor_tags = {}
    kept_tags = {}

    # Categorize images
    for filename, tags in all_tags.items():
        image_path = data_dir / filename

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        if 'discard' in tags:
            # Move to discards
            dest = discards_dir / filename
            discard_tags[filename] = tags

            if dry_run:
                print(f"[DRY RUN] Would move to discards: {filename}")
            else:
                discards_dir.mkdir(exist_ok=True)
                shutil.move(str(image_path), str(dest))
                print(f"Moved to discards: {filename}")

        elif is_pure_outdoor(tags):
            # Move to outdoors
            dest = outdoors_dir / filename
            outdoor_tags[filename] = tags

            if dry_run:
                print(f"[DRY RUN] Would move to outdoors: {filename}")
            else:
                outdoors_dir.mkdir(exist_ok=True)
                shutil.move(str(image_path), str(dest))
                print(f"Moved to outdoors: {filename}")

        else:
            # Keep in main directory
            kept_tags[filename] = tags

    # Save separate tag files
    if not dry_run:
        if discard_tags:
            save_tags(discards_dir / 'tags.json', discard_tags)
            print(f"\nSaved {len(discard_tags)} entries to discards/tags.json")

        if outdoor_tags:
            save_tags(outdoors_dir / 'tags.json', outdoor_tags)
            print(f"Saved {len(outdoor_tags)} entries to outdoors/tags.json")

        # Update main tags file to only include kept images
        save_tags(tags_file, kept_tags)
        print(f"Updated main tags.json with {len(kept_tags)} entries")

    # Summary
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Discards: {len(discard_tags)}")
    print(f"  Pure outdoors: {len(outdoor_tags)}")
    print(f"  Kept: {len(kept_tags)}")
    print(f"  Total: {len(discard_tags) + len(outdoor_tags) + len(kept_tags)}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter pre-training images based on tags'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='pretraining_data',
        help='Path to pretraining_data directory (default: pretraining_data)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without moving files'
    )

    args = parser.parse_args()
    filter_images(args.data_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
