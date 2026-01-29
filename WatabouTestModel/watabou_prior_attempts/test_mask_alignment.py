"""
Diagnostic tool to test mask alignment issues.
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def create_checkerboard_overlay(png_path, mask_path, output_path):
    """
    Create a checkerboard pattern overlay to easily spot misalignment.
    """
    # Load image and mask
    img = cv2.imread(str(png_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Ensure same size
    if img.shape[:2] != mask.shape[:2]:
        print(f"Size mismatch! Image: {img.shape[:2]}, Mask: {mask.shape[:2]}")
        return

    # Create checkerboard pattern on the mask regions
    result = img.copy()

    # Where mask is white (playable areas), apply checkerboard
    checker_size = 20
    for y in range(0, mask.shape[0], checker_size * 2):
        for x in range(0, mask.shape[1], checker_size * 2):
            # First checker
            y1, y2 = y, min(y + checker_size, mask.shape[0])
            x1, x2 = x, min(x + checker_size, mask.shape[1])
            if np.any(mask[y1:y2, x1:x2] == 255):
                result[y1:y2, x1:x2] = [255, 0, 0]  # Red

            # Second checker (diagonal)
            y1, y2 = y + checker_size, min(y + checker_size * 2, mask.shape[0])
            x1, x2 = x + checker_size, min(x + checker_size * 2, mask.shape[1])
            if np.any(mask[y1:y2, x1:x2] == 255):
                result[y1:y2, x1:x2] = [0, 0, 255]  # Blue

    # Blend with original
    blended = cv2.addWeighted(img, 0.5, result, 0.5, 0)

    # Draw grid lines on room boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(str(output_path), blended)
    print(f"Saved checkerboard overlay to {output_path}")


def analyze_mask_content(mask_path):
    """
    Analyze where the actual mask content is located.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Find bounding box of all white pixels
    white_pixels = np.where(mask == 255)

    if len(white_pixels[0]) == 0:
        print("No white pixels found in mask!")
        return

    min_y, max_y = white_pixels[0].min(), white_pixels[0].max()
    min_x, max_x = white_pixels[1].min(), white_pixels[1].max()

    content_width = max_x - min_x
    content_height = max_y - min_y

    print(f"Mask content bounds:")
    print(f"  Full mask size: {mask.shape[1]} x {mask.shape[0]}")
    print(f"  Content region: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    print(f"  Content size: {content_width} x {content_height}")
    print(f"  Content position: {(min_x, min_y)}")
    print(f"  Margins: left={min_x}, top={min_y}, right={mask.shape[1]-max_x}, bottom={mask.shape[0]-max_y}")


if __name__ == "__main__":
    # Test with chapel_of_supremus
    png = Path(r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.png")
    mask = Path(r"data/watabou_corrected/masks/chapel_of_supremus.png")
    output = Path(r"C:\Users\shini\Downloads\watabou_exports\debug2\chapel_checkerboard.png")

    print("="*60)
    analyze_mask_content(mask)
    print("="*60)
    create_checkerboard_overlay(png, mask, output)
    print("\nCheck the checkerboard overlay - if alignment is perfect,")
    print("the checkerboard should stay within room boundaries.")
