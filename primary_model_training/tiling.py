"""
Tiling utilities for processing large battlemaps.

Implements overlapping tile extraction and center-crop stitching to handle
maps of any size while maintaining consistent scale for the model.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TileInfo:
    """Information about a single tile."""
    # Position in the original image (top-left corner)
    x: int
    y: int
    # Size of the tile in original image pixels
    width: int
    height: int
    # Grid cell coordinates (for debugging)
    grid_x: int
    grid_y: int


class TileExtractor:
    """
    Extracts overlapping tiles from a battlemap image.

    Tiles are extracted at a consistent scale (fixed number of grid cells per tile)
    to ensure the model sees walls at consistent relative thickness.
    """

    def __init__(
        self,
        tile_grid_cells: int = 8,
        tile_size: int = 512,
        overlap: float = 0.5
    ):
        """
        Args:
            tile_grid_cells: Number of grid cells per tile (tiles are square)
            tile_size: Output tile size in pixels (model input resolution)
            overlap: Overlap ratio between adjacent tiles (0.5 = 50% overlap)
        """
        self.tile_grid_cells = tile_grid_cells
        self.tile_size = tile_size
        self.overlap = overlap

    def compute_tile_positions(
        self,
        image_width: int,
        image_height: int,
        grid_size: int
    ) -> List[TileInfo]:
        """
        Compute tile positions for extracting from an image.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            grid_size: Grid cell size in pixels

        Returns:
            List of TileInfo objects describing each tile
        """
        # Tile size in original image pixels
        tile_pixels = self.tile_grid_cells * grid_size

        # Stride between tiles (accounting for overlap)
        stride_cells = int(self.tile_grid_cells * (1 - self.overlap))
        stride_pixels = stride_cells * grid_size

        # Compute tile positions using a simple grid
        # We'll add edge-aligned tiles at the end if needed
        tiles = []

        # Generate regular grid positions
        y_positions = list(range(0, image_height - tile_pixels + 1, stride_pixels))
        x_positions = list(range(0, image_width - tile_pixels + 1, stride_pixels))

        # Add edge-aligned position if last tile doesn't reach the edge
        if y_positions and y_positions[-1] + tile_pixels < image_height:
            y_positions.append(image_height - tile_pixels)
        if x_positions and x_positions[-1] + tile_pixels < image_width:
            x_positions.append(image_width - tile_pixels)

        # Handle small images where no full tile fits
        if not y_positions:
            y_positions = [0]
        if not x_positions:
            x_positions = [0]

        for grid_y, y in enumerate(y_positions):
            for grid_x, x in enumerate(x_positions):
                tile_w = min(tile_pixels, image_width - x)
                tile_h = min(tile_pixels, image_height - y)

                tiles.append(TileInfo(
                    x=x, y=y,
                    width=tile_w, height=tile_h,
                    grid_x=grid_x, grid_y=grid_y
                ))

        return tiles

    def extract_tile(
        self,
        image: np.ndarray,
        tile_info: TileInfo,
        grid_size: int
    ) -> np.ndarray:
        """
        Extract and resize a single tile from the image.

        Args:
            image: Source image (H, W, C) or (H, W)
            tile_info: Tile position and size info
            grid_size: Grid cell size in pixels

        Returns:
            Tile resized to (tile_size, tile_size, C) or (tile_size, tile_size)
        """
        from PIL import Image as PILImage

        # Expected tile size
        expected_size = self.tile_grid_cells * grid_size

        # Extract tile region
        x, y = tile_info.x, tile_info.y
        w, h = tile_info.width, tile_info.height

        tile = image[y:y+h, x:x+w]

        # Pad if tile is smaller than expected (edge tiles)
        if w < expected_size or h < expected_size:
            if len(tile.shape) == 3:
                padded = np.zeros((expected_size, expected_size, tile.shape[2]), dtype=tile.dtype)
            else:
                padded = np.zeros((expected_size, expected_size), dtype=tile.dtype)
            padded[:h, :w] = tile
            tile = padded

        # Resize to model input size
        pil_tile = PILImage.fromarray(tile)
        resample = PILImage.Resampling.BILINEAR if len(tile.shape) == 3 else PILImage.Resampling.NEAREST
        pil_tile = pil_tile.resize((self.tile_size, self.tile_size), resample)

        return np.array(pil_tile)

    def extract_all_tiles(
        self,
        image: np.ndarray,
        grid_size: int
    ) -> Tuple[List[np.ndarray], List[TileInfo]]:
        """
        Extract all tiles from an image.

        Args:
            image: Source image (H, W, C) or (H, W)
            grid_size: Grid cell size in pixels

        Returns:
            Tuple of (list of tile arrays, list of TileInfo)
        """
        h, w = image.shape[:2]
        tile_infos = self.compute_tile_positions(w, h, grid_size)

        tiles = []
        for info in tile_infos:
            tile = self.extract_tile(image, info, grid_size)
            tiles.append(tile)

        return tiles, tile_infos


class TileStitcher:
    """
    Stitches tile predictions back into a full-size mask.

    Uses center-crop strategy: only the center portion of each tile prediction
    is used, where the model has maximum context.
    """

    def __init__(
        self,
        tile_grid_cells: int = 8,
        tile_size: int = 512,
        overlap: float = 0.5
    ):
        """
        Args:
            tile_grid_cells: Number of grid cells per tile
            tile_size: Model output tile size in pixels
            overlap: Overlap ratio used during extraction
        """
        self.tile_grid_cells = tile_grid_cells
        self.tile_size = tile_size
        self.overlap = overlap

        # Center crop ratio (portion of tile to keep)
        # With 50% overlap, we keep the center 50%
        self.keep_ratio = 1 - overlap

    def stitch(
        self,
        predictions: List[np.ndarray],
        tile_infos: List[TileInfo],
        output_width: int,
        output_height: int,
        grid_size: int
    ) -> np.ndarray:
        """
        Stitch tile predictions into a full-size output.

        Args:
            predictions: List of tile predictions (tile_size, tile_size) or (tile_size, tile_size, C)
            tile_infos: List of TileInfo from extraction
            output_width: Target output width
            output_height: Target output height
            grid_size: Grid cell size in pixels

        Returns:
            Stitched output array
        """
        from PIL import Image as PILImage

        # Determine output shape
        sample = predictions[0]
        if len(sample.shape) == 3:
            output = np.zeros((output_height, output_width, sample.shape[2]), dtype=sample.dtype)
        else:
            output = np.zeros((output_height, output_width), dtype=sample.dtype)

        # Track which pixels have been filled (for averaging overlapping regions)
        # For simplicity, we use a last-write-wins strategy for the center crop
        # This works because center crops shouldn't overlap with proper stride

        # Calculate crop boundaries in tile pixel space
        crop_margin = int(self.tile_size * self.overlap / 2)
        crop_size = self.tile_size - 2 * crop_margin

        # Scale factor from tile space to original image space
        tile_pixels = self.tile_grid_cells * grid_size
        scale = tile_pixels / self.tile_size

        # Determine which tiles are at edges (need wider or full crop)
        max_x = max(info.x for info in tile_infos) if tile_infos else 0
        max_y = max(info.y for info in tile_infos) if tile_infos else 0

        for pred, info in zip(predictions, tile_infos):
            # For edge tiles, extend the crop to cover the image boundary
            top_margin = crop_margin if info.y > 0 else 0
            left_margin = crop_margin if info.x > 0 else 0
            bottom_margin = crop_margin if info.y < max_y else 0
            right_margin = crop_margin if info.x < max_x else 0

            # Crop prediction with edge-aware margins
            y1 = top_margin
            y2 = self.tile_size - bottom_margin
            x1 = left_margin
            x2 = self.tile_size - right_margin
            cropped = pred[y1:y2, x1:x2]

            # Calculate position in output
            out_x = info.x + int(left_margin * scale)
            out_y = info.y + int(top_margin * scale)

            # Resize cropped prediction to original scale
            out_w = int((x2 - x1) * scale)
            out_h = int((y2 - y1) * scale)

            pil_cropped = PILImage.fromarray(cropped)
            resample = PILImage.Resampling.NEAREST  # Use nearest for class labels
            pil_cropped = pil_cropped.resize((out_w, out_h), resample)
            resized = np.array(pil_cropped)

            # Clamp to output bounds
            src_x1, src_y1 = 0, 0
            dst_x1, dst_y1 = out_x, out_y
            dst_x2, dst_y2 = min(out_x + out_w, output_width), min(out_y + out_h, output_height)

            if dst_x1 < 0:
                src_x1 = -dst_x1
                dst_x1 = 0
            if dst_y1 < 0:
                src_y1 = -dst_y1
                dst_y1 = 0

            copy_w = dst_x2 - dst_x1
            copy_h = dst_y2 - dst_y1

            if copy_w > 0 and copy_h > 0:
                output[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]

        return output


class TilePipeline:
    """
    Combined pipeline for tile-based inference.

    Handles extraction, model inference, and stitching.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tile_grid_cells: int = 8,
        tile_size: int = 512,
        overlap: float = 0.5,
        device: str = 'cuda',
        imagenet_norm: bool = False,
    ):
        """
        Args:
            model: Trained segmentation model
            tile_grid_cells: Number of grid cells per tile
            tile_size: Model input/output size
            overlap: Overlap ratio
            device: Torch device
            imagenet_norm: Use ImageNet normalization instead of simple 0-1
        """
        self.model = model
        self.device = device
        self.imagenet_norm = imagenet_norm
        self.extractor = TileExtractor(tile_grid_cells, tile_size, overlap)
        self.stitcher = TileStitcher(tile_grid_cells, tile_size, overlap)

    def predict(
        self,
        image: np.ndarray,
        grid_size: int,
        batch_size: int = 4,
        global_image: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Run tile-based prediction on a full image.

        Args:
            image: Input image (H, W, C), values 0-255
            grid_size: Grid cell size in pixels
            batch_size: Number of tiles to process at once
            global_image: Optional pre-processed global context tensor (1, 3, H, W)

        Returns:
            Predicted class mask (H, W)
        """
        h, w = image.shape[:2]

        # Extract tiles
        tiles, tile_infos = self.extractor.extract_all_tiles(image, grid_size)

        # ImageNet normalization constants
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Convert tiles to tensors
        tile_tensors = []
        for tile in tiles:
            # Normalize to 0-1 and convert to tensor
            t = torch.from_numpy(tile).float() / 255.0
            if len(t.shape) == 2:
                t = t.unsqueeze(0)  # Add channel dim
            else:
                t = t.permute(2, 0, 1)  # HWC -> CHW

            if self.imagenet_norm:
                t = (t - IMAGENET_MEAN) / IMAGENET_STD

            tile_tensors.append(t)

        # Batch inference
        predictions = []
        self.model.eval()

        with torch.no_grad():
            for i in range(0, len(tile_tensors), batch_size):
                batch = torch.stack(tile_tensors[i:i+batch_size]).to(self.device)

                kwargs = {}
                if global_image is not None:
                    # Expand global_image to match batch size
                    bs = batch.shape[0]
                    kwargs['global_image'] = global_image.expand(bs, -1, -1, -1)

                logits = self.model(batch, **kwargs)
                preds = logits.argmax(dim=1).cpu().numpy()

                for pred in preds:
                    predictions.append(pred.astype(np.uint8))

        # Stitch predictions
        output = self.stitcher.stitch(predictions, tile_infos, w, h, grid_size)

        return output


if __name__ == "__main__":
    # Test tiling
    print("Testing TileExtractor...")

    extractor = TileExtractor(tile_grid_cells=8, tile_size=512, overlap=0.5)

    # Simulate a 4800x6000 image with 200px grid
    image_w, image_h = 4800, 6000
    grid_size = 200

    tiles = extractor.compute_tile_positions(image_w, image_h, grid_size)
    print(f"Image: {image_w}x{image_h}, Grid: {grid_size}px")
    print(f"Tiles: {len(tiles)}")

    # Print first few tiles
    for i, t in enumerate(tiles[:5]):
        print(f"  Tile {i}: pos=({t.x}, {t.y}), size=({t.width}x{t.height}), grid=({t.grid_x}, {t.grid_y})")

    # Test with actual image
    print("\nTesting tile extraction...")
    dummy_image = np.random.randint(0, 256, (image_h, image_w, 3), dtype=np.uint8)

    extracted, infos = extractor.extract_all_tiles(dummy_image, grid_size)
    print(f"Extracted {len(extracted)} tiles")
    print(f"Tile shape: {extracted[0].shape}")

    # Test stitching
    print("\nTesting TileStitcher...")
    stitcher = TileStitcher(tile_grid_cells=8, tile_size=512, overlap=0.5)

    # Simulate predictions (random class labels)
    dummy_preds = [np.random.randint(0, 5, (512, 512), dtype=np.uint8) for _ in extracted]

    stitched = stitcher.stitch(dummy_preds, infos, image_w, image_h, grid_size)
    print(f"Stitched shape: {stitched.shape}")
    print(f"Unique values: {np.unique(stitched)}")
