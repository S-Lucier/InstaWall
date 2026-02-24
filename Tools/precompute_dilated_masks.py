"""
Pre-compute dilated versions of all masks and save to disk.

Instead of dilating on-the-fly per tile (slow, breaks multiprocessing workers),
this runs dilation once on the full mask image and saves it. Training then loads
pre-dilated masks with num_workers=4 like any normal file.

Usage:
    python Tools/precompute_dilated_masks.py --radius 5

Output:
    data/foundry_to_mask/line_masks_d5/   (same filenames as line_masks/)
    data/watabou_to_mask_d5/watabou_edge_mask/  (same filenames)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def make_kernel(radius: int) -> np.ndarray:
    r = radius
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= r * r).astype(np.uint8)


def dilate_mask(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same logic as dataset.py _dilate_mask — dilate each class outward into background.
    Uses cv2.dilate which is ~20-50x faster than scipy on large images."""
    result = mask.copy()
    background = mask == 0
    unique_classes = [c for c in np.unique(mask) if c != 0]
    for cls in unique_classes:
        binary = (mask == cls).astype(np.uint8)
        expanded = cv2.dilate(binary, kernel, iterations=1).astype(bool)
        result[expanded & background] = cls
        background = result == 0
    return result


def process_dir(src: Path, dst: Path, struct: np.ndarray, glob: str = '*.png'):
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob(glob))
    print(f'  {src} -> {dst}  ({len(files)} files)')

    for i, fpath in enumerate(files):
        out_path = dst / fpath.name
        if out_path.exists():
            print(f'  [{i+1}/{len(files)}] skip (exists): {fpath.name}')
            continue

        img = Image.open(fpath)
        mask = np.array(img)

        dilated = dilate_mask(mask, struct)

        Image.fromarray(dilated).save(out_path)
        print(f'  [{i+1}/{len(files)}] {fpath.name}')
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--foundry-mask-dir', default='data/foundry_to_mask/line_masks')
    parser.add_argument('--watabou-mask-dir', default='data/watabou_to_mask/watabou_edge_mask')
    args = parser.parse_args()

    r = args.radius
    struct = make_kernel(r)
    print(f'Dilation radius: {r}  (kernel {2*r+1}x{2*r+1})')

    foundry_src = Path(args.foundry_mask_dir)
    foundry_dst = foundry_src.parent / f'{foundry_src.name}_d{r}'

    watabou_src = Path(args.watabou_mask_dir)
    watabou_dst = watabou_src.parent.parent / f'{watabou_src.parent.name}_d{r}' / watabou_src.name

    print('\nFoundry masks:')
    process_dir(foundry_src, foundry_dst, struct, glob='*_mask_lines.png')

    print('\nWatabou masks:')
    process_dir(watabou_src, watabou_dst, struct, glob='*.png')

    print('\nDone.')
    print(f'  Foundry dilated:  {foundry_dst}')
    print(f'  Watabou dilated:  {watabou_dst}')
    print()
    print('Training command (remove --mask-dilation, update dirs):')
    print(f'  --mask-dir {foundry_dst}')
    print(f'  --watabou-dir {watabou_dst.parent}')


if __name__ == '__main__':
    main()
