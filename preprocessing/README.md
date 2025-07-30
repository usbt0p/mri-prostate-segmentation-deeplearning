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

#### Methods
- `add(func, *args, **kwargs)`: Adds processing function to pipeline
- `run(images, parallel=False, max_workers=4)`: Executes pipeline on image(s)
- `__call__(images, parallel=True)`: Allows pipeline to be called as function

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

#### Usage
```bash
python preprocessing/TestPreprocessing.py
```

Visual outputs are saved to `./imgs/` directory for inspection.

## Common Preprocessing Workflow

### Standard Image Preprocessing Pipeline
```python
# 1. Load and ensure 3D format
image = load_image(image_path)

# 2. Extract region of interest (60-65% crop around center)  
roi_image = get_region_of_interest(image, crop=0.6)

# 3. Resample to consistent spacing
resampled_image = resample_image(roi_image, out_spacing=(0.5, 0.5, 3.0))

# 4. Apply bias field correction
corrected_image = n4_bias_field_correction(resampled_image)

# 5. Optional: Normalize intensities
normalized_image = normalize_image(corrected_image, method="zscore")
```

### Batch Processing with Pipeline
```python
from preprocessing.Pipeline import Pipeline
from preprocessing.PreProcessor import *

# Create pipelines for images and labels
image_pipeline = Pipeline().add(load_image) \
                          .add(get_region_of_interest, crop=0.6) \
                          .add(resample_image) \
                          .add(n4_bias_field_correction)

label_pipeline = Pipeline().add(load_image) \
                          .add(get_region_of_interest, crop=0.6) \
                          .add(resample_image, interpolator=sitk.sitkNearestNeighbor)

# Process image-label pairs in parallel
from preprocessing.PreProcessor import preprocess_pairs_parallel

results = preprocess_pairs_parallel(
    img_lbl_pairs=list(zip(image_paths, label_paths)),
    pipeline_images=image_pipeline,
    pipeline_labels=label_pipeline,
    workers=8
)
```

## Configuration Parameters

### Default Processing Settings
- **Resampling spacing**: `(0.5, 0.5, 3.0)` mm for x, y, z dimensions
- **ROI crop factor**: `0.6` (60% retention around image center)
- **Interpolation**: Linear for images, Nearest Neighbor for masks
- **N4 correction**: 50 iterations, 4 fitting levels, shrink factor 4
- **Normalization**: Z-score (excludes background pixels)

### Label Mapping Convention
Following PI-CAI dataset convention:
- Background: 0
- Transition Zone (TZ): 1  
- Peripheral Zone (PZ): 2

## Dependencies

Core dependencies for medical image processing:
- `SimpleITK`: Medical image I/O and processing
- `numpy`: Array operations
- `concurrent.futures`: Parallel processing
- `tqdm`: Progress tracking
- `matplotlib`: Visualization utilities

## File Output Format

Processed images follow nnU-Net naming convention:
- Images: `{prefix}_{case_id}_0000.nii.gz` (channel 0)
- Labels: `{prefix}_{case_id}.nii.gz`

## Memory Management

- Batch processing: Recommended 150 image pairs per batch
- Parallel workers: Default uses all CPU cores
- Progress tracking: Built into pipeline execution
- Thread-safe: All parallel operations use proper synchronization

## Testing and Quality Assurance

Run visual tests to validate preprocessing functions:
```bash
python preprocessing/TestPreprocessing.py
```

This creates visual outputs in `./imgs/` directory showing:
- Before/after comparisons for each preprocessing step
- Intensity histograms for normalization validation
- Slice-by-slice visualizations for spatial operations
- Error metrics for resampling quality assessment

## Performance Optimization

- Use parallel processing for batch operations
- Configure worker count based on available CPU cores
- Process images in batches to manage memory usage
- Cache intermediate results when processing large datasets
- Use appropriate interpolation methods (Linear for images, Nearest Neighbor for masks)