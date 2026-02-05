"""
Primary model training for wall segmentation.

This module trains a UNet model to predict Foundry VTT wall segments
from battlemap images.
"""

from .config import Config
from .model import WallSegmentationUNet
from .dataset import WallSegmentationDataset, create_dataloader
from .tiling import TileExtractor, TileStitcher
