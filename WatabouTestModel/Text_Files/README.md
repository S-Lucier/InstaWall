# U-Net Image Segmentation for TTRPG Maps

This project implements a U-Net model for detecting walls and doors in top-down TTRPG maps. This is a learning implementation using PyTorch.

## Project Structure

```
UnetTinkering/
├── unet_model.py          # U-Net architecture implementation
├── dataset.py             # Dataset loading and augmentation
├── train.py               # Training script
├── utils.py               # Helper functions (checkpoints, metrics)
├── download_dataset.py    # Helper script to download practice datasets
├── requirements.txt       # Python dependencies
└── data/                  # Dataset directory (created after download)
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download a practice dataset:**

   For the Carvana dataset (recommended for learning):
   - Go to https://www.kaggle.com/c/carvana-image-masking-challenge/data
   - Download `train.zip` and `train_masks.zip`
   - Extract them to create this structure:
     ```
     data/
     ├── train_images/
     └── train_masks/
     ```
   - Split into train/val folders (80/20 split recommended)

   Alternatively, use a smaller dataset to start:
   - Oxford-IIIT Pet Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
   - Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/

3. **Test the model:**
   ```bash
   python unet_model.py
   ```
   This will run a test to verify the architecture is working.

## Training

Once you have the dataset set up:

```bash
python train.py
```

Training parameters can be adjusted at the top of `train.py`:
- `LEARNING_RATE`: Learning rate for optimizer (default: 1e-4)
- `BATCH_SIZE`: Batch size (default: 16)
- `NUM_EPOCHS`: Number of training epochs (default: 20)
- `IMAGE_HEIGHT`, `IMAGE_WIDTH`: Input image dimensions (default: 256x256)

## Understanding U-Net

U-Net has two main parts:

1. **Encoder (Contracting Path):** Downsamples the image to extract features
   - Each level: 2 conv layers + pooling
   - Captures "what" is in the image

2. **Decoder (Expanding Path):** Upsamples back to original resolution
   - Uses skip connections from encoder
   - Captures "where" things are located

3. **Skip Connections:** Copy features from encoder to decoder
   - Preserves spatial information lost during downsampling
   - Critical for accurate localization

## Key Files Explained

### unet_model.py
- `DoubleConv`: Basic building block (2 conv layers with BatchNorm and ReLU)
- `UNet`: Main model architecture with encoder, decoder, and skip connections

### dataset.py
- `SegmentationDataset`: Generic dataset loader for images and masks
- `get_train_transforms()`: Data augmentation for training
- `get_val_transforms()`: Preprocessing for validation (no augmentation)

### train.py
- Main training loop with mixed precision training
- Saves checkpoints after each epoch
- Calculates accuracy and Dice score for validation

### utils.py
- `check_accuracy()`: Calculate pixel accuracy and Dice score
- `save_predictions()`: Save sample predictions for visualization
- `dice_coefficient()`, `iou_score()`: Additional metrics

## Next Steps for TTRPG Maps

After learning with this practice implementation:

1. **Create your own dataset:**
   - Collect TTRPG map images
   - Label walls and doors (tools: GIMP, Photoshop, or labelme)
   - Use different colors for walls vs doors (multi-class segmentation)

2. **Modify for multi-class:**
   - Change `out_channels` in U-Net to number of classes (e.g., 3: background, walls, doors)
   - Use `CrossEntropyLoss` instead of `BCEWithLogitsLoss`

3. **Post-processing:**
   - Apply morphological operations (closing, opening) to clean up predictions
   - Convert to Foundry VTT wall format (line segments or polygons)
   - Use traditional CV techniques (contour detection, line fitting)

## Monitoring Training

Training will output:
- Loss per batch (in progress bar)
- Validation accuracy and Dice score after each epoch
- Sample predictions saved to `saved_predictions/`

Good Dice scores for segmentation:
- >0.90: Excellent
- 0.80-0.90: Good
- <0.80: Needs improvement

## GPU Recommendations

- This code uses CUDA if available
- Training on CPU will be slow but functional for small datasets
- Mixed precision training (automatic) speeds up training on modern GPUs
