"""
Configuration for wall segmentation model training.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    """Training configuration."""

    # ==========================================================================
    # Paths
    # ==========================================================================

    # Directory containing battlemap images
    image_dir: str = "data/foundry_to_mask"

    # Directory containing mask files (line segment masks)
    mask_dir: str = "data/foundry_to_mask/line_masks"

    # Output directory for checkpoints and logs
    output_dir: str = "outputs/wall_segmentation"

    # ==========================================================================
    # Model Architecture
    # ==========================================================================

    # Number of input channels (RGB)
    in_channels: int = 3

    # Number of output classes
    # 0: Background, 1: Wall (includes secret doors, windows), 2: Terrain, 3: Door
    num_classes: int = 4

    # Feature channels at each encoder level
    features: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # Use ASPP (Atrous Spatial Pyramid Pooling) in bottleneck
    use_aspp: bool = True

    # Use attention gates in decoder
    use_attention: bool = True

    # ==========================================================================
    # Tiling
    # ==========================================================================

    # Number of grid cells per tile (tiles are square)
    # Model sees this many grid cells at a time
    tile_grid_cells: int = 8

    # Model input size (tiles resized to this)
    tile_size: int = 512

    # Overlap ratio for tiling (0.5 = 50% overlap)
    tile_overlap: float = 0.5

    # ==========================================================================
    # Training
    # ==========================================================================

    # Batch size
    batch_size: int = 8

    # Number of epochs
    epochs: int = 100

    # Learning rate
    learning_rate: float = 1e-4

    # Weight decay (L2 regularization)
    weight_decay: float = 1e-5

    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"

    # Number of data loading workers
    num_workers: int = 4

    # Random seed for reproducibility
    seed: int = 42

    # ==========================================================================
    # Class Weights
    # ==========================================================================

    # Class weights for imbalanced data
    # Higher weight = more importance during training
    # Background is most common, walls/doors are rare
    # [Background, Wall, Terrain, Door]
    class_weights: List[float] = field(default_factory=lambda: [0.1, 1.0, 1.0, 2.0])

    # ==========================================================================
    # Augmentation
    # ==========================================================================

    # Whether to apply augmentations during training
    augment: bool = True

    # Horizontal flip probability
    hflip_prob: float = 0.5

    # Vertical flip probability
    vflip_prob: float = 0.5

    # Rotation (90 degree increments) probability
    rotate_prob: float = 0.5

    # Color jitter probability
    color_jitter_prob: float = 0.3

    # ==========================================================================
    # Logging & Checkpointing
    # ==========================================================================

    # Log every N batches
    log_interval: int = 10

    # Validate every N epochs
    val_interval: int = 1

    # Save checkpoint every N epochs
    save_interval: int = 5

    # Keep only the N best checkpoints
    keep_best_n: int = 3

    # ==========================================================================
    # Mask Processing
    # ==========================================================================

    # Scale factor used in mask files (class_value = class_id * scale)
    mask_scale: int = 50

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert 0 < self.tile_overlap < 1, "tile_overlap must be between 0 and 1"
        assert len(self.class_weights) == self.num_classes, \
            f"class_weights length ({len(self.class_weights)}) must match num_classes ({self.num_classes})"

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = Config()
