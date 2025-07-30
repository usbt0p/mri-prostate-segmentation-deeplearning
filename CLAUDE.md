# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical imaging deep learning project for anatomical segmentation of the prostate gland from MRI images. The project uses multiple datasets (PICAI, Prostate158, Decathlon) and provides a comprehensive preprocessing pipeline for preparing medical imaging data for deep learning training.

## Installation and Dependencies

Install dependencies using:
```bash
pip install -r requirements.txt
```

Core dependencies:
- `monai` - Medical imaging AI framework
- `nibabel` - NIfTI file reading 
- `simpleitk` - Medical image processing
- `matplotlib`, `numpy`, `pandas` - Data visualization and analysis
- `ipywidgets` - Jupyter notebook widgets

## Common Development Commands

### Dataset Download
```bash
./download.sh
```
Downloads PICAI dataset folds from Zenodo.

### Preprocessing Tests
Run individual preprocessing function tests:
```bash
python preprocessing/TestPreprocessing.py
```
This runs visual tests for preprocessing functions like ROI extraction, N4 bias correction, normalization, resampling, and mask operations.

### Data Preprocessing Pipeline
Execute data building scripts for different datasets:
```bash
python loadingData/build_data_picai.py
python loadingData/build_data_decathlon.py  
python loadingData/build_data_158.py
```

### Exploratory Analysis
Use Jupyter notebooks in `exploratoryAnalysis/` folder:
- `explore_picai.ipynb` - PICAI dataset analysis
- `explore_decathlon.ipynb` - Decathlon dataset analysis  
- `explore_158.ipynb` - Prostate158 dataset analysis

## Code Architecture

### Core Modules

**DataAnalyzer** (`exploratoryAnalysis/DataAnalyzer.py`)
- Medical imaging data exploration and analysis
- Metadata extraction from DICOM/NIfTI files
- Image visualization and intensity histogram generation
- File/directory navigation with regex filtering
- Methods: `show_image()`, `collect_metadata_to_dataframe()`, `image_intensity_histogram()`

**Pipeline** (`preprocessing/Pipeline.py`)
- Chainable preprocessing pipeline system
- Supports parallel processing with progress tracking
- Thread-safe execution for batch processing
- Usage: `Pipeline().add(func, *args, **kwargs).run(images, parallel=True)`

**PreProcessor** (`preprocessing/PreProcessor.py`)
- Individual preprocessing functions for medical images
- Functions: `load_image()`, `resample_image()`, `get_region_of_interest()`, `n4_bias_field_correction()`, `normalize_image()`
- Mask operations: `combine_zonal_masks()`, `swap_zonal_mask_values()`

**Utils** (`preprocessing/Utils.py`)
- Utility functions for file operations and parallel processing
- Functions: `create_filename()`, `preprocess_pairs_parallel()`, `save_pairs_parallel()`
- Image description and correspondence testing utilities

### Data Processing Workflow

1. **Data Loading**: Use `DataAnalyzer` to explore and load medical imaging datasets
2. **Preprocessing Pipeline**: Chain operations using `Pipeline` class:
   - Image loading and 4D→3D conversion
   - Resampling to consistent voxel spacing
   - ROI extraction (cropping around prostate region)
   - N4 bias field correction
   - Optional normalization (z-score or min-max)
3. **Batch Processing**: Process image-label pairs in parallel batches
4. **Output**: Save in nnU-Net format with JSON metadata

### Dataset Conventions

**Label Mapping** (following PI-CAI convention):
- Background: 0
- Transition Zone (TZ): 1  
- Peripheral Zone (PZ): 2

**File Naming**:
- Images: `{prefix}_{case_id}_0000.nii.gz` (channel 0)
- Labels: `{prefix}_{case_id}.nii.gz`

### Key Configuration Parameters

- **Resampling spacing**: `(0.5, 0.5, 3.0)` mm for x, y, z dimensions
- **ROI crop factor**: `0.6-0.65` (60-65% crop around image center)
- **Interpolation**: Linear for images, Nearest Neighbor for masks
- **Batch processing**: 150 image pairs per batch to manage memory

## Development Notes

- All paths in `DataAnalyzer` are relative to a configured data root directory
- The preprocessing pipeline is designed to maintain image-label correspondence
- Visual tests save output images to `./imgs/` directory for inspection
- Parallel processing uses all available CPU cores by default
- Progress tracking is built into pipeline execution