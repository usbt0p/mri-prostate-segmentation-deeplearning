# Data Loading and Preprocessing Pipeline

This document explains how the data loading process works, how it processes and structures the data, and how parallelization is implemented to optimize performance.

## Overview

The data loading pipeline consists of several key components:
1. Reading raw medical imaging data from various sources
2. Preprocessing the data through a configurable pipeline
3. Storing the processed data in a standardized format
4. Creating metadata JSON files to track the data and preprocessing steps

The entire process is designed to be efficient, scalable, and to handle large medical imaging datasets while preserving the relationship between images and their corresponding labels/masks.

## Directory Structure

### Input Structure
The input data comes from various datasets with different structures. Each dataset script (`build_data_*.py`) handles the specific directory structure of its source. Common datasets used include:

- **PICAI** (Prostate Imaging: Cancer AI) dataset
- **Prostate158** dataset
- **Decathlon** dataset

### Output Structure
The processed data follows the nnUNet directory structure convention:

```
/path/to/nnUNet_raw/
├── Dataset001_picai/
│   ├── imagesTr/
│   │   ├── picai_0000_0000.nii.gz
│   │   ├── picai_0001_0000.nii.gz
│   │   └── ...
│   ├── labelsTr/
│   │   ├── picai_0000.nii.gz
│   │   ├── picai_0001.nii.gz
│   │   └── ...
│   ├── preprocessing.json
│   └── dataset.json
├── Dataset002_prostate158/
│   ├── ...same structure...
└── Dataset003_decathlon/
    └── ...same structure...
```

## JSON Files

Two important JSON files are created for each dataset:

### 1. preprocessing.json

This file maps input paths to output paths and stores metadata about the preprocessing:

```json
{
  "imagesTr": {
    "/path/to/input/image1.mha": "/path/to/output/image1_0000.nii.gz",
    "/path/to/input/image2.mha": "/path/to/output/image2_0000.nii.gz",
    ...
  },
  "labelsTr": {
    "/path/to/input/label1.mha": "/path/to/output/label1.nii.gz",
    "/path/to/input/label2.mha": "/path/to/output/label2.nii.gz",
    ...
  },
  "metadata": {
    "pipeline_images": "Description of image pipeline steps",
    "pipeline_labels": "Description of label pipeline steps",
    "preprocessing_time": 123.45
  }
}
```

### 2. dataset.json

This file follows the nnUNet convention for dataset metadata:

```json
{
  "channel_names": {
    "0": "T2"
  },
  "labels": {
    "background": 0,
    "TZ": 1,
    "PZ": 2
  },
  "numTraining": 123,
  "file_ending": ".nii.gz"
}
```

## Preprocessing Steps

The preprocessing is implemented as a configurable pipeline of operations that can be applied to both images and masks. Each dataset has its own optimized pipeline based on the specific characteristics of the data.

### Pipeline Class

The `Pipeline` class allows chaining multiple preprocessing operations in a specific order. It supports parallel processing of multiple images for efficiency.

```python
pipeline = Pipeline()
pipeline.add(load_image) \
        .add(resample_image, interpolator=sitk.sitkLinear, out_spacing=spacing) \
        .add(get_region_of_interest, crop=crop_factor) \
        .add(n4_bias_field_correction)
```

### Preprocessing Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `load_image` | Loads a medical image from disk | Path to image file |
| `resample_image` | Resamples image to specified voxel spacing | `out_spacing`, `interpolator` |
| `get_region_of_interest` | Crops image around center | `crop` (percentage to keep) |
| `n4_bias_field_correction` | Corrects intensity inhomogeneity | `shrink_factor`, `num_iterations` |
| `normalize_image` | Normalizes image intensities | `method` ('zscore' or 'minmax') |
| `reorient_image` | Standardizes image orientation | `target_direction` |
| `swap_zonal_mask_values` | Swaps mask values if needed | None |

## Dataset-Specific Preprocessing Parameters

| Dataset | Spacing (mm) | Crop Factor | Swap Mask Values | Orientation | Special Notes |
|---------|--------------|-------------|------------------|-------------|---------------|
| PICAI | (0.5, 0.5, 3.0) | 0.65 | False | Default | Processes in batches of 150 due to memory constraints |
| Prostate158 | (0.5, 0.5, 3.0) | 0.75 | True | Default | Includes anatomy reader filtering |
| Decathlon | (0.5, 0.5, 3.0) | 0.55 | False | RPS | Uses reorient_image step |

## Parallelization Strategy

The data loading pipeline leverages parallelism in multiple ways:

1. **Parallel Image-Label Processing**: 
   - Uses `preprocess_pairs_parallel()` to process image-label pairs simultaneously
   - Maintains the relationship between images and their labels

2. **Parallel Saving**:
   - Uses `save_pairs_parallel()` to write processed images and labels to disk in parallel
   - Optimizes I/O operations

3. **Thread Pool Execution**:
   - Uses Python's `ThreadPoolExecutor` to manage parallel tasks
   - Configurable number of workers (typically set to `os.cpu_count()`)

4. **Batch Processing** (PICAI dataset):
   - Processes data in batches to manage memory usage
   - Important for very large datasets that might not fit in memory

Example parallelization code:
```python
# Process image-label pairs in parallel
paired_results = preprocess_pairs_parallel(
    img_lbl_pairs, pipeline_images, pipeline_labels, workers=workers
)

# Save processed pairs to disk in parallel
out_i, out_l = save_pairs_parallel(
    paired_results, out_images, out_labels, workers=workers
)
```

## Memory Management

For large datasets (e.g., PICAI), the processing is divided into batches to prevent memory overflow:

```python
# Process in batches of 150 pairs
batch = 150
for i in range(0, len(img_lbl_pairs), batch):
    pairs_slice = img_lbl_pairs[i:i+batch]
    out_images_slice = out_images[i:i+batch]
    out_labels_slice = out_labels[i:i+batch]
    
    # Process this batch
    paired_results = preprocess_pairs_parallel(...)
    out_i, out_l = save_pairs_parallel(...)
    
    # Free memory
    del paired_results, out_l, out_i, pairs_slice, out_images_slice, out_labels_slice
```

## Validation and Quality Control

The pipeline includes several validation steps:

1. **Input-Output Path Mapping**: Ensures each input has a corresponding output path
2. **Image-Label Correspondence**: Validates that images match their labels in dimensions and spacing
3. **Visual Inspection**: Random samples are visualized after processing for quality control
4. **Logging**: Processing steps and results are logged for tracking and debugging

## Conclusion

This data loading and preprocessing pipeline provides a robust framework for handling medical imaging data from diverse sources. Its modular design, parallel processing capabilities, and standardized output make it suitable for large-scale deep learning projects in medical image segmentation.
