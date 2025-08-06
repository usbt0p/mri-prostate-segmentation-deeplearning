# Preprocessing Module

This module provides a comprehensive preprocessing pipeline for MRI prostate anatomy segmentation data. It contains modular tools for image loading, processing, and batch operations optimized for medical imaging workflows.

## Overview

The preprocessing module consists of four main components:

- **PreProcessor.py**: Core image processing functions for medical images
- **Pipeline.py**: Chainable pipeline system for sequential processing operations  
- **Utils.py**: Utility functions for visualization and debugging
- **TestPreprocessing.py**: Visual testing suite for individual preprocessing functions

## Core Modules

### PreProcessor.py

Contains individual preprocessing functions for medical image processing:

#### Image Loading and Conversion
- `load_image(image_path)`: Loads 3D medical images from file paths
- `ensure_3d(image)`: Converts 4D images to 3D by extracting single volumes

#### Spatial Processing
- `resample_image(image, out_spacing=(0.5, 0.5, 3.0))`: Resamples images to consistent voxel spacing
- `get_region_of_interest(image, crop=0.6)`: Extracts ROI by cropping around image center
- `reorient_image(image, target_direction)`: Reorients images to target anatomical direction

#### Intensity Processing
- `normalize_image(image, method="zscore")`: Normalizes image intensities using Z-score or Min-Max
- `n4_bias_field_correction(image)`: Applies N4 bias field correction to reduce MRI field inhomogeneities

#### Mask Operations
- `combine_zonal_masks(zonal_mask, pz_value, tz_value)`: Combines peripheral and transition zone masks
- `swap_zonal_mask_values(zonal_mask)`: Swaps zone label values for multi-dataset compatibility

#### Utility Functions
- `describe_image(image)`: Prints image metadata (size, spacing, origin, direction)
- `to_array(image)`: Converts SimpleITK images to NumPy arrays
- `save_image(image, out_path)`: Saves SimpleITK images to disk

### Pipeline.py

Provides a chainable pipeline system for sequential image processing:

#### Key Features
- Method chaining for readable preprocessing workflows
- Parallel processing support with progress tracking
- Thread-safe execution for batch operations
- Configurable worker count for performance optimization

#### Usage Example
```python
from preprocessing.Pipeline import Pipeline
from preprocessing.PreProcessor import *

# Create and configure pipeline
pipeline = Pipeline()
pipeline.add(load_image) \
        .add(get_region_of_interest, crop=0.6) \
        .add(resample_image, out_spacing=(0.5, 0.5, 3.0)) \
        .add(n4_bias_field_correction) \
        .add(normalize_image, method="zscore")

# Process single image
processed_image = pipeline(image_path)

# Process multiple images in parallel
processed_images = pipeline(image_paths, parallel=True, max_workers=8)
```

### Utils.py

Contains utility functions for visualization and debugging:

#### Visualization Tools
- `visualize_dicom_slider(image_or_path)`: Interactive slider visualization for 3D images
- `plot_slices_of_images_with_slider(*images)`: Multi-image comparison with unified slider
- `show_random_from_json(json_file)`: Visualizes random samples from nnU-Net format datasets

#### Debugging and Validation
- `test_label_image_correspondence(zipped)`: Validates image-label pair compatibility
- `modify_preprocessing_json(preprocessing)`: Modifies nnU-Net JSON file paths

### TestPreprocessing.py

Visual testing suite for individual preprocessing functions:

#### Available Tests
- `roi_test()`: Tests region of interest extraction
- `n4_test()`: Tests N4 bias field correction with intensity histograms
- `normalization_test()`: Tests image normalization methods
- `test_resample_images()`: Tests image resampling with quality evaluation
- `test_resample_mask()`: Tests mask resampling with nearest neighbor interpolation
- `test_combine_zonal_masks()`: Tests zonal mask combination
- `test_swap_zonal_mask_values()`: Tests zone label value swapping

## File Output Format

Processed images follow nnU-Net naming convention:
- Images: `{prefix}_{case_id}_0000.nii.gz` (channel 0)
- Labels: `{prefix}_{case_id}.nii.gz`
