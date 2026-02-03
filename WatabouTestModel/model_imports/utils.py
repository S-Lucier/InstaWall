import torch
import torchvision
from tqdm import tqdm


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save model checkpoint."""
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """Load model checkpoint."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda"):
    """Calculate accuracy and Dice score on validation set."""
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            # Forward pass
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Calculate accuracy
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # Calculate Dice score
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    accuracy = (num_correct / num_pixels) * 100
    dice_score = dice_score / len(loader)

    model.train()
    return accuracy, dice_score


def save_predictions(loader, model, folder="saved_predictions", device="cuda"):
    """Save a few prediction examples as images."""
    model.eval()

    for idx, (x, y) in enumerate(loader):
        if idx >= 5:  # Save only first 5 batches
            break

        x = x.to(device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Save images
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f"{folder}/mask_{idx}.png"
        )

    model.train()


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient (F1 score for binary segmentation)."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice


def iou_score(pred, target, smooth=1e-6):
    """Calculate Intersection over Union (IoU) score."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou
