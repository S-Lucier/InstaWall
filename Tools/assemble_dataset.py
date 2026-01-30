"""
Dataset Assembler

Reads tagged images and assembles a training dataset based on specified
tag ratios and filters.

Usage:
    # Copy all dungeon images that are high quality
    python assemble_dataset.py --tags ./tags.json --source ./images --output ./training_data \
        --require high_quality --require dungeon

    # Assemble with ratios: 70% dungeon, 20% cave, 10% outdoor
    python assemble_dataset.py --tags ./tags.json --source ./images --output ./training_data \
        --ratio dungeon:0.7 --ratio cave:0.2 --ratio outdoor:0.1 --total 1000

    # Exclude low quality and discarded
    python assemble_dataset.py --tags ./tags.json --source ./images --output ./training_data \
        --exclude low_quality --exclude discard

Examples:
    # Pretraining dataset: mostly indoor, some outdoor, no low quality
    python assemble_dataset.py -t tags.json -s ./images -o ./pretrain_data \
        --exclude low_quality --exclude discard \
        --ratio dungeon:0.7 --ratio cave:0.1 --ratio urban:0.1 --ratio outdoor:0.1 \
        --total 5000
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def load_tags(tags_file):
    """Load tags from JSON file."""
    with open(tags_file) as f:
        return json.load(f)


def filter_by_tags(all_tags, require=None, exclude=None, any_of=None):
    """Filter images based on tag requirements.

    Args:
        all_tags: Dict of filename -> list of tags
        require: Tags that MUST be present (AND logic)
        exclude: Tags that MUST NOT be present
        any_of: At least one of these tags must be present (OR logic)

    Returns:
        Set of filenames that match criteria
    """
    require = set(require or [])
    exclude = set(exclude or [])
    any_of = set(any_of or [])

    matching = set()

    for filename, tags in all_tags.items():
        tags_set = set(tags)

        # Check exclusions first
        if exclude and tags_set & exclude:
            continue

        # Check requirements
        if require and not require.issubset(tags_set):
            continue

        # Check any_of
        if any_of and not (tags_set & any_of):
            continue

        matching.add(filename)

    return matching


def group_by_tag(all_tags, filenames, tag_list):
    """Group filenames by which tag they have.

    Args:
        all_tags: Dict of filename -> list of tags
        filenames: Set of filenames to consider
        tag_list: List of tags to group by

    Returns:
        Dict of tag -> list of filenames
    """
    groups = defaultdict(list)

    for filename in filenames:
        tags = set(all_tags.get(filename, []))
        for tag in tag_list:
            if tag in tags:
                groups[tag].append(filename)
                break  # Only add to first matching group

    return groups


def select_with_ratios(groups, ratios, total):
    """Select images according to specified ratios.

    Args:
        groups: Dict of tag -> list of filenames
        ratios: Dict of tag -> ratio (should sum to ~1.0)
        total: Total number of images to select

    Returns:
        List of selected filenames
    """
    selected = []

    # Normalize ratios
    ratio_sum = sum(ratios.values())
    if ratio_sum > 0:
        ratios = {k: v / ratio_sum for k, v in ratios.items()}

    # Calculate target counts
    targets = {tag: int(total * ratio) for tag, ratio in ratios.items()}

    # Select from each group
    for tag, target in targets.items():
        available = groups.get(tag, [])
        if not available:
            print(f"Warning: No images with tag '{tag}'")
            continue

        # Sample with replacement if not enough
        if len(available) >= target:
            selected.extend(random.sample(available, target))
        else:
            print(f"Warning: Only {len(available)} images with tag '{tag}', "
                  f"need {target}. Using all + duplicates.")
            selected.extend(available)
            # Fill remainder with duplicates
            remaining = target - len(available)
            selected.extend(random.choices(available, k=remaining))

    return selected


def copy_images(filenames, source_dir, output_dir, dry_run=False):
    """Copy selected images to output directory.

    Args:
        filenames: List of filenames to copy
        source_dir: Source directory
        output_dir: Output directory
        dry_run: If True, just print what would be copied

    Returns:
        Number of files copied
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    seen_names = {}  # Handle duplicates

    for filename in filenames:
        src = source_dir / filename

        if not src.exists():
            # Try to find in subdirectories
            matches = list(source_dir.rglob(filename))
            if matches:
                src = matches[0]
            else:
                print(f"Warning: File not found: {filename}")
                continue

        # Handle duplicate filenames (from sampling with replacement)
        base_name = Path(filename).stem
        ext = Path(filename).suffix

        if filename in seen_names:
            seen_names[filename] += 1
            dest_name = f"{base_name}_{seen_names[filename]}{ext}"
        else:
            seen_names[filename] = 0
            dest_name = filename

        dest = output_dir / dest_name

        if dry_run:
            print(f"Would copy: {src} -> {dest}")
        else:
            shutil.copy2(src, dest)

        copied += 1

    return copied


def parse_ratio(ratio_str):
    """Parse ratio string like 'dungeon:0.7' into (tag, ratio)."""
    parts = ratio_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid ratio format: {ratio_str}. Use 'tag:ratio'")
    return parts[0], float(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description='Assemble training dataset from tagged images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/output
    parser.add_argument('--tags', '-t', type=str, required=True,
                       help='JSON file with image tags')
    parser.add_argument('--source', '-s', type=str, required=True,
                       help='Source directory containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for assembled dataset')

    # Filtering
    parser.add_argument('--require', '-r', action='append', default=[],
                       help='Required tag (can specify multiple, AND logic)')
    parser.add_argument('--exclude', '-x', action='append', default=[],
                       help='Excluded tag (can specify multiple)')
    parser.add_argument('--any-of', '-a', action='append', default=[],
                       help='At least one of these tags required (OR logic)')

    # Ratios
    parser.add_argument('--ratio', action='append', default=[],
                       help='Tag ratio like "dungeon:0.7" (can specify multiple)')
    parser.add_argument('--total', '-n', type=int, default=None,
                       help='Total number of images to select')

    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be copied without copying')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--clear', action='store_true',
                       help='Clear output directory before copying')

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Load tags
    print(f"Loading tags from {args.tags}")
    all_tags = load_tags(args.tags)
    print(f"Found {len(all_tags)} tagged images")

    # Apply filters
    matching = filter_by_tags(
        all_tags,
        require=args.require if args.require else None,
        exclude=args.exclude if args.exclude else None,
        any_of=args.any_of if args.any_of else None
    )
    print(f"After filtering: {len(matching)} images")

    if not matching:
        print("No images match the criteria!")
        return

    # Handle ratios
    if args.ratio:
        ratios = dict(parse_ratio(r) for r in args.ratio)
        total = args.total or len(matching)

        print(f"\nRatios requested:")
        for tag, ratio in ratios.items():
            print(f"  {tag}: {ratio:.1%}")
        print(f"Total target: {total}")

        # Group by ratio tags
        groups = group_by_tag(all_tags, matching, list(ratios.keys()))

        print(f"\nAvailable per tag:")
        for tag in ratios.keys():
            count = len(groups.get(tag, []))
            print(f"  {tag}: {count}")

        # Select with ratios
        selected = select_with_ratios(groups, ratios, total)

    else:
        # No ratios - take all matching (or up to --total)
        selected = list(matching)
        if args.total and args.total < len(selected):
            selected = random.sample(selected, args.total)

    print(f"\nSelected {len(selected)} images")

    # Count tags in selection
    tag_counts = defaultdict(int)
    for filename in selected:
        for tag in all_tags.get(filename, []):
            tag_counts[tag] += 1

    print("\nTag distribution in selection:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        pct = count / len(selected) * 100
        print(f"  {tag}: {count} ({pct:.1f}%)")

    # Clear output if requested
    output_dir = Path(args.output)
    if args.clear and output_dir.exists() and not args.dry_run:
        print(f"\nClearing {output_dir}")
        shutil.rmtree(output_dir)

    # Copy images
    print(f"\n{'Would copy' if args.dry_run else 'Copying'} to {args.output}")
    copied = copy_images(selected, args.source, args.output, dry_run=args.dry_run)

    print(f"\n{'Would copy' if args.dry_run else 'Copied'} {copied} images")

    # Save manifest
    if not args.dry_run:
        manifest_path = output_dir / 'manifest.json'
        manifest = {
            'source': str(args.source),
            'filters': {
                'require': args.require,
                'exclude': args.exclude,
                'any_of': args.any_of
            },
            'ratios': dict(parse_ratio(r) for r in args.ratio) if args.ratio else None,
            'total': len(selected),
            'tag_distribution': dict(tag_counts),
            'files': selected
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {manifest_path}")


if __name__ == '__main__':
    main()
