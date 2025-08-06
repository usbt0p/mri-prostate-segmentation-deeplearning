# ExploratoryAnalysis Module

This module provides comprehensive tools for exploring and analyzing medical imaging datasets, specifically focused on prostate MRI data. It includes the core DataAnalyzer class and Jupyter notebooks for dataset-specific analysis of PICAI, Decathlon, and Prostate158 datasets.

## Overview

The exploratoryAnalysis module consists of:

- **DataAnalyzer.py**: Core class for medical imaging data exploration and metadata extraction
- **Dataset-specific Jupyter notebooks**: Interactive analysis notebooks for each supported dataset
- **Statistical analysis tools**: Methods for understanding data distributions, quality, and preprocessing requirements

## Core Components

### DataAnalyzer.py

A comprehensive class for analyzing medical imaging directories and extracting metadata from DICOM/NIfTI files.

#### Key Features
- **File Discovery**: Regex-based file and directory navigation
- **Metadata Extraction**: DICOM/NIfTI header parsing with SimpleITK
- **Image Visualization**: Multi-image display with slice selection
- **Statistical Analysis**: Intensity histograms and data distribution analysis
- **Quality Assessment**: Empty mask detection and data validation
- **ROI Analysis**: Bounding box calculation for region-of-interest determination

#### Core Methods

##### File System Navigation
```python
# Initialize analyzer with data root
analyzer = DataAnalyzer("/path/to/data/root")

# Set regex filter for specific file types
analyzer.regex = ".*_t2w.mha$"

# Get directories and files
dirs = list(analyzer.get_dirs("dataset/path"))
files = list(analyzer.get_files("dataset/path", regex="pattern"))

# Generate file paths from subdirectories
file_paths = list(analyzer.file_paths_gen("parent_directory"))
```

##### Image Visualization
```python
# Display single or multiple images
analyzer.show_image("path/to/image.nii.gz", save="output.png")
analyzer.show_image(image1, mask1, image2, slice=15)

# Interactive visualization with specific slice
analyzer.show_image("image.nii.gz", "mask.nii.gz", slice=10, title="Analysis")
```

##### Metadata Extraction
```python
# Extract metadata from single file
metadata = analyzer.parse_metadata_file("image.nii.gz")

# Collect metadata from directory
df = analyzer.collect_metadata_to_dataframe("dataset/folder")

# Parallel metadata collection from subdirectories
df = analyzer.collect_metadata_from_subdirs("parent/folder", max_workers=8)
```

##### Statistical Analysis
```python
# Generate intensity histogram
hist, bins = analyzer.image_intensity_histogram("image.nii.gz", plot=True)

# Check for empty masks
is_empty = analyzer.is_empty_mask("mask.nii.gz")

# Count non-empty masks in directory
count, empty_list = analyzer.count_and_find_non_empty_masks("masks/folder")
```

##### ROI and Bounding Box Analysis
```python
# Calculate cube bounds for mask
bounds = analyzer.calculate_cube_bounds("mask.nii.gz")
# Returns: (start_z, end_z, start_y, end_y, start_x, end_x, 
#          mask_path, bounding_box_size, proportion_of_image_size)

# Visualize bounding box overlay
analyzer.overlay_bounding_box(mask_array, slice_idx, start_y, end_y, start_x, end_x)
```

##### Utility Functions
```python
# Pick random files or directories
random_files = analyzer.pick_random("folder/path", num=5, type="file")
random_dirs = analyzer.pick_random("folder/path", num=3, type="dir")

# Convert relative to absolute paths
abs_path = analyzer.abspath("relative/path")
```

#### Extracted Metadata Fields

The DataAnalyzer extracts the following metadata from medical images:

- **filename**: Base filename
- **dim_size**: Image dimensions (x, y, z) or (x, y, z, channels)
- **spacing**: Voxel spacing in mm (x, y, z)
- **orientation**: Anatomical orientation (RAI, ASL, RSA, etc.)
- **prostate_volume**: Prostate volume from DICOM metadata (if available)
- **vendor**: MRI scanner vendor (SIEMENS, Philips, etc.)
- **mri_name**: Scanner model name
- **psa_report**: PSA value from metadata (if available)

## Dataset Summary Table

Some info about key characteristics and analysis findings for each dataset:

| Dataset | Sample Count | Scanner Variability | Resolution Variations | Label Convention | Optimal Crop Factor | Spacing (mm) | Key Preprocessing Notes |
|---------|-------------|-------------------|---------------------|------------------|-------------------|--------------|------------------------|
| **PICAI** | ~1,500 cases<br>(5 folds) | **Multiple resolutions**<br>• T2W: 384×384, 640×640<br>• Sagittal: 320×320<br>• Coronal: 320×320 | **Standard**<br>• TZ = 1<br>• PZ = 2<br>• Background = 0 | **0.65**<br>(65% retention) | **(0.5, 0.5, 3.0)** | • Batch processing (150 pairs)<br>• Multi-fold cross-validation<br>• Memory management required<br>• 5 imaging sequences per case |
| **Decathlon** | 32+16 cases<br>(training) | **Mostly uniform**<br>• Primary: 320×320<br>• Some: 256×256, 384×384<br>• 4D format (T2W + ADC) | **Inverted**<br>• TZ = 2<br>• PZ = 1<br>• Background = 0 | **0.55**<br>(55% retention) | **(0.5, 0.5, 3.0)** | • Requires reorientation to "RPS"<br>• Label value swapping needed<br>• 4D to 3D conversion<br>• Single-batch processing |
| **Prostate158** | 139+19 cases<br>(train + test) | **Two main groups**<br>• Group 1: 270×270<br>• Group 2: 442×442<br>• Consistent slice thickness | **Inverted**<br>• TZ = 2<br>• PZ = 1<br>• Background = 0 | **0.75**<br>(75% retention) | **(0.5, 0.5, 3.0)** | • Label value swapping needed<br>• Larger anatomy extent<br>• Reader1 annotations preferred<br>• High inter-reader agreement |
