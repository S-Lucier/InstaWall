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

    # Optional JSON file with per-map metadata (grid_size, etc.)
    metadata_file: Optional[str] = None

    # Output directory for checkpoints and logs
    output_dir: str = "outputs/wall_segmentation"

    # Optional Watabou data directory (contains watabou_images/, watabou_edge_mask/, watabou_recessed_mask/)
    watabou_dir: Optional[str] = None

    # Per-epoch inclusion probability for Watabou maps (0.36 ≈ match Foundry count)
    watabou_include_prob: float = 0.36

    # ==========================================================================
    # Model Architecture
    # ==========================================================================

    # Model type: "unet", "segformer", "segformer_gc" (with global context),
    # or "segformer_texture" (few-shot texture conditioning)
    model_type: str = "unet"

    # SegFormer variant: "b0" through "b5" (only used when model_type="segformer"/"segformer_gc")
    segformer_variant: str = "b0"

    # Use ImageNet normalization (auto-set to True for segformer in __post_init__)
    use_imagenet_norm: bool = False

    # Global context settings (only used when model_type="segformer_gc")
    use_global_context: bool = False
    global_image_size: int = 256
    global_context_dim: int = 128

    # Texture prototype settings (only used when model_type="segformer_texture")
    texture_crop_size: int = 64         # crop side length in 512px tile space
    texture_crops_per_class: int = 2    # crops to sample per class per tile

    # Number of input channels (RGB)
    in_channels: int = 3

    # Number of output classes
    # Default (4): 0: Background, 1: Wall (includes secret doors, windows), 2: Terrain, 3: Door
    # With merge_terrain (3): 0: Background, 1: Wall (includes terrain), 2: Door
    num_classes: int = 4

    # Whether to merge terrain class into wall class
    merge_terrain: bool = False

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

    # Random tiles extracted per image per epoch
    tiles_per_image: int = 4

    # Number of epochs
    epochs: int = 1000

    # Early stopping: stop if no new best IoU for this many epochs (0 = disabled)
    early_stopping_patience: int = 50

    # Learning rate
    learning_rate: float = 1e-4

    # Weight decay (L2 regularization)
    weight_decay: float = 1e-5

    # EMA decay rate (0 = disabled, 0.999 typical for segmentation)
    ema_decay: float = 0.999

    # Focal loss (replaces cross-entropy). gamma=0 is equivalent to cross-entropy.
    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    # Focal loss phase schedule: CE → Focal → CE curriculum (default off).
    # focal_schedule_start: epoch to switch from CE to Focal.
    # focal_schedule_end:   epoch to switch back to CE (0 = never switch back).
    focal_loss_schedule: bool = False
    focal_schedule_start: int = 15
    focal_schedule_end: int = 30

    # Mask dilation radius in pixels (training only). 0 = disabled.
    # Expands thin wall lines so the model receives stronger gradient signal.
    mask_dilation: int = 0

    # Context border tiling: number of extra grid cells included on each side of
    # each tile as context. 0 = disabled (old overlap-based stitching).
    # At inference, overlap auto-set to 0.0 when context_cells > 0.
    tile_context_cells: int = 1

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
    # Default (4 classes): [Background, Wall, Terrain, Door]
    # With merge_terrain (3 classes): [Background, Wall, Door]
    class_weights: Optional[List[float]] = None

    # ==========================================================================
    # Augmentation
    # ==========================================================================

    # Randomly convert tiles to greyscale during training (probability 0.3).
    # Reduces colour bias across different map art styles.
    grayscale_aug: bool = True

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
    save_interval: int = 50

    # Keep only the N best checkpoints
    keep_best_n: int = 3

    # ==========================================================================
    # Mask Processing
    # ==========================================================================

    # Scale factor used in mask files (class_value = class_id * scale)
    mask_scale: int = 50

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.model_type in ("unet", "segformer", "segformer_gc", "segformer_texture"), \
            f"model_type must be 'unet', 'segformer', 'segformer_gc', or 'segformer_texture', " \
            f"got '{self.model_type}'"
        assert 0 <= self.tile_overlap < 1, "tile_overlap must be between 0 (inclusive) and 1 (exclusive)"

        # Auto-enable ImageNet normalization for SegFormer variants
        if self.model_type in ("segformer", "segformer_gc", "segformer_texture"):
            self.use_imagenet_norm = True

        # Auto-enable global context for segformer_gc
        if self.model_type == "segformer_gc":
            self.use_global_context = True

        # Apply merge_terrain settings
        if self.merge_terrain:
            self.num_classes = 3
            if self.class_weights is None:
                # [Background, Wall+Terrain, Door]
                self.class_weights = [0.1, 1.0, 2.0]
        else:
            if self.class_weights is None:
                # [Background, Wall, Terrain, Door]
                self.class_weights = [0.1, 1.0, 1.0, 2.0]

        assert len(self.class_weights) == self.num_classes, \
            f"class_weights length ({len(self.class_weights)}) must match num_classes ({self.num_classes})"

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = Config()
