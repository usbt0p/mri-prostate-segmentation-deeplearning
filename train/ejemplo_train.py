#!/usr/bin/env python3

import os
import sys
import json
import time
import psutil
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, NormalizeIntensityd, RandSpatialCropd,
    RandFlipd, RandRotate90d, ToTensord, EnsureTyped
)
from monai.data import Dataset, DataLoader, create_test_image_3d
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    return cpu_mem, gpu_mem

def load_data_splits(data_dir):
    # TODO modify according to data structure

    """Load prepared data splits"""
    data_dir = Path(data_dir)
    
    NFOLDS = 5
    # Load file lists
    train_files = []
    folds = []

    with open(data_dir / "train_data.json", 'r') as f:
        train_files = json.load(f)
        tr = len(train_files['images'])
        print(f"Loaded {tr} training samples from JSON")
    
    for i in range(NFOLDS):
        fold_files = [
            {"image": img, "label": lbl} for img, lbl in zip(
                train_files['images'][i:50:5], 
                train_files['labels'][i:50:5])
        ]
        folds.append(fold_files)
    print(f"Created {NFOLDS} folds with {len(folds[0])} samples each")

    return train_files, folds

def create_transforms(is_train=True):
    """Create data transforms with proper label handling"""
    from monai.transforms import (
        RandGaussianNoised, RandAdjustContrastd, RandGaussianSmoothd,
        RandScaleIntensityd, RandShiftIntensityd, AsDiscreted, ResizeWithPadOrCropd
    )
    
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # TODO investigar y cambiar
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), 
                  mode=("bilinear", "nearest")),

        # Ensure divisible dimensions for UNet
        # TODO esto es para estar seguros de que no hay problemas con las dimensiones
        # TODO poner y quitar esto para saber si es totalmente necesario
        ResizeWithPadOrCropd(keys=["image", "label"], 
                             # TODO investigar y cambiar
                             spatial_size=(160, 256, 256)),
        # TODO investigar y cambiar
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        
        # Ensure labels are integers
        AsDiscreted(keys=["label"], rounding="torchrounding"),
        EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.long])
    ]
    
    if is_train:
        # More aggressive augmentation for overfitting
        train_transforms = base_transforms + [
            # TODO investigar y cambiar
            # TODO buscar function que mantenga al menos un voxel de mascara
            # esto es muy importante: si no hay máscara, hay explosion / desvanecimiento de gradientes
            # estimar el tamaño de la roi de forma que contanga al 
            # menos un cuarto de prostata (o asi, (100,100,100))
            RandSpatialCropd(keys=["image", "label"], roi_size=(80, 80, 80), random_size=False),
            RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.7),
            RandRotate90d(keys=["image", "label"], prob=0.7, spatial_axes=[0, 1]),
            # More intensive augmentation for overfitting
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.3),
            RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),
            ToTensord(keys=["image", "label"])
        ]
        return Compose(train_transforms)
    else:
        # Validation transforms (no augmentation but ensure proper label format)
        val_transforms = base_transforms + [
            ToTensord(keys=["image", "label"])
        ]
        return Compose(val_transforms)

def create_model(num_classes, device):
    """Create UNet model"""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=0.1,
        bias=True,
        act="PRELU"
    )
    
    return model.to(device)

def train_epoch(model, train_loader, optimizer, loss_function, device, epoch, writer):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        inputs = batch_data["image"].to(device)
        targets = batch_data["label"].to(device).long()
        
        # Debug shapes for first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            print(f"Debug - Input shape: {inputs.shape}")
            print(f"Debug - Target shape: {targets.shape}")
            print(f"Debug - Target unique values: {torch.unique(targets)}")
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Debug output shape for first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            print(f"Debug - Output shape: {outputs.shape}")
        
        loss = loss_function(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Log memory usage
        cpu_mem, gpu_mem = get_memory_usage()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                  f"CPU Mem: {cpu_mem:.1f}MB, GPU Mem: {gpu_mem:.1f}MB")
            
            # Log to tensorboard
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/Train", loss.item(), step)
            writer.add_scalar("Memory/CPU_MB", cpu_mem, step)
            writer.add_scalar("Memory/GPU_MB", gpu_mem, step)
    
    return epoch_loss / len(train_loader)

def validate_epoch(model, val_loader, loss_function, dice_metric, device, epoch, writer):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0
    dice_metric.reset()
    
    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data["image"].to(device)
            targets = batch_data["label"].to(device).long()
            
            # Use sliding window inference for validation with smaller ROI
            roi_size = (80, 80, 80)  # Consistent with training ROI
            sw_batch_size = 1
            
            outputs = sliding_window_inference(
                inputs, roi_size, sw_batch_size, model, overlap=0.5
            )
            
            loss = loss_function(outputs, targets)
            epoch_loss += loss.item()
            
            # Calculate dice score
            outputs_softmax = torch.softmax(outputs, dim=1)
            dice_metric(y_pred=outputs_softmax, y=targets)
            
            # Debug: Print some stats for first batch of first epoch
            if epoch == 0 and len(dice_metric._buffers) == 1:
                pred_classes = torch.argmax(outputs_softmax, dim=1)
                print(f"  Debug - Pred classes: {torch.unique(pred_classes)}")
                print(f"  Debug - Target classes: {torch.unique(targets)}")
                print(f"  Debug - Output shape: {outputs.shape}")
    
    # Get mean dice score
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    # Log to tensorboard
    writer.add_scalar("Loss/Validation", epoch_loss / len(val_loader), epoch)
    writer.add_scalar("Dice/Validation", mean_dice, epoch) # FIXME
    
    return epoch_loss / len(val_loader), mean_dice

def main():
    """Main training function"""
    # Set random seeds for reproducibility
    set_determinism(42)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"MONAI version: {monai.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir / "logs")
    
    #try:
    # Load data
    print("Loading data...")
    train_files, folds = load_data_splits("/home/guest/code/data_jsons")
    
    # Cross-validation loop
    best_dice_scores = []
    for fold_idx, fold_files in enumerate(folds):
        print(f"Starting fold {fold_idx + 1}/{len(folds)}")
        
        # Split data into training and validation sets
        val_files = fold_files
        train_files_fold = [
            f for idx, f in enumerate(folds) if idx != fold_idx 
        ]
        
        # Load first sample to get number of classes
        sample_data = nib.load(val_files[0]["label"])
        sample_mask = sample_data.get_fdata()
        sample_mask = np.rint(sample_mask).astype(int)
        num_classes = int(np.max(sample_mask)) + 1
        print(f"Number of classes: {num_classes}")
        
        # Create transforms
        train_transforms = create_transforms(is_train=True)
        val_transforms = create_transforms(is_train=False)
        
        # Create datasets
        train_dataset = Dataset(data=train_files_fold, transform=train_transforms)
        val_dataset = Dataset(data=val_files, transform=val_transforms)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        # Create model
        print("Creating model...")
        model = create_model(num_classes, device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup loss function and optimizer
        loss_function = DiceCELoss(to_onehot_y=num_classes, softmax=True, include_background=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
        
        # Setup metrics
        dice_metric = DiceMetric(include_background=False, reduction="mean")
    
        # Training loop
        print("Starting training...")
        num_epochs = 10 # 100  # Higher number for overfitting
        best_dice = 0
        epochs_without_improvement = 0
        patience = 10  # Patience for early stopping
        n_epochs_to_save = 10
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, epoch, writer)
            
            # Validate
            val_loss, val_dice = validate_epoch(model, val_loader, loss_function, dice_metric, device, epoch, writer)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Additional debugging info
            if epoch < 5:  # First 5 epochs
                print(f"  Train Loss: {train_loss:.4f} (Threshold: < 1.0 for good start)")
                print(f"  Val Dice: {val_dice:.4f} (Threshold: > 0.2 for improvement)")
                print(f"  Loss decreasing: {'Yes' if train_loss < 1.0 else 'No'}")
                print(f"  Dice improving: {'Yes' if val_dice > 0.2 else 'No'}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_loss': val_loss,
                }, output_dir / "best_model.pth")
                print(f"  New best model saved! Dice: {best_dice:.4f}")
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint every x epochs
            if (epoch + 1) % n_epochs_to_save == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_loss': val_loss,
                }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            
            # Early stopping if overfitting achieved
            # if val_dice > 0.90:  # Lower threshold for overfitting
            #     print(f"Overfitting achieved! Dice: {val_dice:.4f}")
            #     break
            
            # Early stopping if no improvement for patience epochs
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs. Early stopping.")
                break
        
        print(f"Training completed! Best Dice: {best_dice:.4f}")
        
    # except Exception as e:
    #     print(f"Error during training: {e}")
    #     sys.exit(1)
    
    # finally:
    writer.close()

if __name__ == "__main__":
    main()
    #load_data_splits("/home/guest/code/data_jsons")