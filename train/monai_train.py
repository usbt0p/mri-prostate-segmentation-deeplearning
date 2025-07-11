import json
from glob import glob
import numpy as np
from monai.transforms import *
from monai.data import CacheDataset, DataLoader

# paths = {
#     "images": "/media/guest/PORT-DISK/Datasets/nnUNet_raw/Dataset003_decathlon/imagesTr",
#     "labels": "/media/guest/PORT-DISK/Datasets/nnUNet_raw/Dataset003_decathlon/labelsTr",
# }

# train_images = glob(paths["images"] + "/*")
# train_labels = glob(paths["labels"] + "/*")

# # 2. Build your dataset information list
# data_dicts = [
#     {"image": img_in, "label": lbl_in}
#     for img_in, lbl_in 
#     in zip(train_images, train_labels)
# ]

# 1. Load dataset information from JSON
def load_dataset_from_json(json_file):
    """Load dataset information from a JSON file"""
    with open(json_file, 'r') as f:
        data_dicts = json.load(f)
    return data_dicts

# 2. Define transforms
def create_transforms(is_train=True):
    """Create data transforms"""
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        #Orientationd(keys=["image", "label"], axcodes="RAS"),
        #Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"])
    ]
    
    if is_train:
        # Add data augmentation for training
        train_transforms = base_transforms + [
            # RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
            # RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
            # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
            ToTensord(keys=["image", "label"])
        ]
        return Compose(train_transforms)
    else:
        # Validation transforms (no augmentation)
        val_transforms = base_transforms + [
            ToTensord(keys=["image", "label"])
        ]
        return Compose(val_transforms)


# 3. Create dataset and loader
dataset = CacheDataset(data=data_dicts, transform=preprocess, cache_rate=1.0)
loader = DataLoader(dataset, batch_size=1, num_workers=0)

# 5. Iterate to verify correct loading and optionally save preprocessed arrays
for batch in loader:
    img, lbl = batch["image"], batch["label"]

    print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Loaded label shape: {lbl.shape}, dtype: {lbl.dtype}")

    # Basic sanity checks
    assert img.ndim == 5, f"Expected shape (B, C, D, H, W), got {img.ndim}"
    assert lbl.ndim == 5, f"Expected shape (B, C, D, H, W), got {lbl.ndim}"
    assert img.shape == lbl.shape, "Image and label shapes must match"
