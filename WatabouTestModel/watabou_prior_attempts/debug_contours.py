"""Debug script to visualize contours before and after simplification."""

import cv2
import numpy as np
from PIL import Image

# Load mask
mask = np.array(Image.open('../data/val_masks/abbey_of_the_moon_king.png').convert('L'))

# Extract floor and door masks
floor_mask = (mask == 255).astype(np.uint8) * 255
door_mask = (mask == 127).astype(np.uint8) * 255

# Dilate floor
wall_offset = 5
kernel_size = wall_offset * 2 + 1
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
floor_dilated = cv2.dilate(floor_mask, kernel)

# Subtract doors
floor_dilated_no_doors = cv2.subtract(floor_dilated, door_mask)

# Find contours
contours, _ = cv2.findContours(floor_dilated_no_doors, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Create visualization - focus on upper-left area
vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Draw first contour in detail (should be the main outer boundary)
if contours:
    contour = contours[0]

    # Draw all points in the contour
    for point in contour:
        pt = tuple(point[0])
        cv2.circle(vis, pt, 1, (255, 255, 0), -1)  # Yellow dots for every contour point

    # Simplify with epsilon=3
    simplified = cv2.approxPolyDP(contour, 3, closed=True)

    # Draw simplified points in different color
    for point in simplified:
        pt = tuple(point[0])
        cv2.circle(vis, pt, 3, (0, 255, 0), -1)  # Green dots for simplified points

    # Draw simplified segments
    for i in range(len(simplified)):
        pt1 = tuple(simplified[i][0])
        pt2 = tuple(simplified[(i+1) % len(simplified)][0])
        cv2.line(vis, pt1, pt2, (0, 0, 255), 1)  # Red lines for simplified segments

# Crop to upper-left area (first door region)
vis_crop = vis[0:400, 0:400]

cv2.imwrite('../../results/debug_contours_upperleft.png', vis_crop)
print(f"Total contour points (unsimplified): {len(contours[0]) if contours else 0}")
print(f"Total contour points (simplified, epsilon=3): {len(simplified) if contours else 0}")
print("Visualization saved to results/debug_contours_upperleft.png")
print("Yellow = original contour points")
print("Green = points kept after Douglas-Peucker")
print("Red = simplified segments")
