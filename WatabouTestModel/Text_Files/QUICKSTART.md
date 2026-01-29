# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Test the Model Architecture

Verify everything is working:

```bash
python unet_model.py
```

Expected output:
```
Input shape: torch.Size([1, 3, 256, 256])
Output shape: torch.Size([1, 1, 256, 256])
Model parameters: 31,037,633
Model test passed!
```

## 3. Prepare Dataset

### Option A: Create Directory Structure Only

```bash
python prepare_dataset.py
```

This will show instructions for downloading datasets.

### Option B: Download and Split Dataset

1. Download Carvana dataset from Kaggle:
   - https://www.kaggle.com/c/carvana-image-masking-challenge/data
   - Download `train.zip` and `train_masks.zip`

2. Extract to `data/raw/`:
   ```
   data/raw/images/      (extract train.zip here)
   data/raw/masks/       (extract train_masks.zip here)
   ```

3. Split into train/val:
   ```bash
   python prepare_dataset.py --split
   ```

## 4. Train the Model

```bash
python train.py
```

Training will:
- Save checkpoints to `saved_models/`
- Save sample predictions to `saved_predictions/`
- Display progress with loss, accuracy, and Dice score

## 5. Run Inference on New Images

After training, test your model:

```bash
python inference_binary.py --image path/to/your/image.jpg --checkpoint saved_models/checkpoint_epoch_20.pth.tar
```

This will display the original image and predicted mask side by side.

## Understanding the Output

### During Training
- **Loss**: Should decrease over time (lower is better)
- **Accuracy**: Percentage of correctly classified pixels
- **Dice Score**: Overlap between prediction and ground truth
  - 0.0 = no overlap
  - 1.0 = perfect overlap
  - >0.90 is excellent

### Sample Predictions
Check `saved_predictions/` after each epoch to see:
- `pred_*.png`: Model predictions
- `mask_*.png`: Ground truth masks

Compare these visually to see if the model is learning.

## Common Issues

### CUDA Out of Memory
Reduce `BATCH_SIZE` in `train.py`:
```python
BATCH_SIZE = 8  # or 4
```

### Training is Slow
- Make sure you're using GPU (check "Training on device: cuda")
- Reduce `IMAGE_HEIGHT` and `IMAGE_WIDTH` to 128x128
- Reduce `NUM_WORKERS` if on Windows

### Poor Results
- Train for more epochs (increase `NUM_EPOCHS`)
- Check that masks match images (same filenames)
- Verify masks are binary (0 and 255 values only)
- Increase dataset size if possible

## Next Steps

Once you're comfortable with the practice dataset:

1. **Create your TTRPG dataset:**
   - Use tools like labelme, GIMP, or Photoshop
   - Label walls as white (255), background as black (0)
   - For doors, consider multi-class segmentation

2. **Modify for multiple classes:**
   ```python
   # In train.py, change:
   model = UNet(in_channels=3, out_channels=3)  # 3 classes: bg, walls, doors
   loss_fn = nn.CrossEntropyLoss()
   ```

3. **Add post-processing:**
   - Morphological operations (erosion, dilation)
   - Contour detection with OpenCV
   - Convert to Foundry VTT wall format

## File Overview

| File | Purpose |
|------|---------|
| `unet_model.py` | U-Net architecture |
| `dataset.py` | Data loading and augmentation |
| `train.py` | Training loop |
| `inference.py` | Run predictions on new images |
| `utils.py` | Helper functions (metrics, checkpoints) |
| `prepare_dataset.py` | Dataset preparation script |

## Key Hyperparameters

Located in `train.py`:
- `LEARNING_RATE = 1e-4`: How fast the model learns
- `BATCH_SIZE = 16`: Number of images per training step
- `NUM_EPOCHS = 20`: How many times to iterate over dataset
- `IMAGE_HEIGHT/WIDTH = 256`: Input image size

Start with defaults, adjust if needed!
