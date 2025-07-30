# LoadingData Module

This module provides dataset building scripts for preprocessing and converting multiple prostate MRI datasets into nnU-Net compatible format. It handles three major datasets: PICAI, Decathlon Task05_Prostate, and Prostate158, each with specific preprocessing pipelines and data structures.

## Overview

The loadingData module consists of dataset-specific build scripts that:

- Load raw medical imaging data from various dataset formats
- Apply consistent preprocessing pipelines using the preprocessing module
- Convert data to nnU-Net format with proper metadata
- Generate JSON configuration files for training frameworks
- Process data in parallel batches for memory efficiency

## Core Scripts

### build_data_picai.py

Processes the PICAI (PI-CAI) dataset for prostate cancer AI challenge data.

#### Dataset Configuration
- **Raw Data**: PI-CAI dataset with T2-weighted MRI images and zonal anatomical masks
- **Output Format**: `Dataset001_picai` in nnU-Net structure
- **Label Source**: Zonal masks (Transition Zone + Peripheral Zone) from Yuan23 annotations
- **Image Source**: T2-weighted axial images from 5-fold cross-validation splits

#### Key Features
- Processes images from all 5 PICAI folds (`picai_images_fold0` to `picai_images_fold4`)
- Uses zonal anatomical delineations for training labels
- Batch processing (150 pairs per batch) for memory management
- Correspondence validation between images and labels
- Comprehensive logging and progress tracking

#### Preprocessing Pipeline
```python
# Image Pipeline
load_image → resample_image(spacing=(0.5,0.5,3.0)) → 
get_region_of_interest(crop=0.65) → n4_bias_field_correction

# Label Pipeline  
load_image → resample_image(nearest_neighbor) → 
get_region_of_interest(crop=0.65) → [optional: swap_zonal_mask_values]
```

#### Configuration Parameters
- **Resampling spacing**: `(0.5, 0.5, 3.0)` mm
- **ROI crop factor**: `0.65` (65% retention around center)
- **Batch size**: 150 image pairs per processing batch
- **File format**: `.nii.gz` with nnU-Net naming convention

### build_data_decathlon.py

Processes the Medical Segmentation Decathlon Task05_Prostate dataset.

#### Dataset Configuration
- **Raw Data**: Decathlon prostate segmentation challenge data
- **Output Format**: `Dataset003_decathlon` in nnU-Net structure
- **Label Source**: Combined prostate gland masks (whole gland segmentation)
- **Image Source**: T2-weighted MRI volumes

#### Key Features
- Handles both training and test image sets
- Applies image reorientation to "RPS" (Right-Posterior-Superior) convention
- Single-pass processing (no batching due to smaller dataset size)
- Visual validation with random sample inspection

#### Preprocessing Pipeline
```python
# Image Pipeline
load_image → reorient_image("RPS") → resample_image(spacing=(0.5,0.5,3.0)) → 
get_region_of_interest(crop=0.55) → n4_bias_field_correction

# Label Pipeline
load_image → reorient_image("RPS") → resample_image(nearest_neighbor) → 
get_region_of_interest(crop=0.55) → [optional: swap_zonal_mask_values]
```

#### Configuration Parameters
- **Resampling spacing**: `(0.5, 0.5, 3.0)` mm
- **ROI crop factor**: `0.55` (55% retention - tighter crop)
- **Reorientation**: "RPS" anatomical direction
- **Processing**: Full dataset in single batch

### build_data_158.py

Processes the Prostate158 dataset with multi-reader annotations.

#### Dataset Configuration
- **Raw Data**: Prostate158 dataset with expert radiologist annotations
- **Output Format**: `Dataset001_prostate158` in nnU-Net structure
- **Label Source**: T2 anatomy masks from reader1 annotations
- **Image Source**: T2-weighted MRI images from train and test sets

#### Key Features
- Combines training and test sets into single training dataset
- Uses reader1 anatomical annotations (multiple readers available)
- Requires zonal mask value swapping due to different label conventions
- Larger ROI crop factor optimized for this dataset's anatomy distribution

#### Preprocessing Pipeline
```python
# Image Pipeline
load_image → resample_image(spacing=(0.5,0.5,3.0)) → 
get_region_of_interest(crop=0.75) → n4_bias_field_correction

# Label Pipeline
load_image → resample_image(nearest_neighbor) → 
get_region_of_interest(crop=0.75) → swap_zonal_mask_values
```

#### Configuration Parameters
- **Resampling spacing**: `(0.5, 0.5, 3.0)` mm
- **ROI crop factor**: `0.75` (75% retention - largest crop)
- **Label swapping**: Enabled by default (`bool_swap_mask_values = True`)
- **Reader selection**: Uses reader1 annotations specifically

### data.template.json

Template configuration file for nnU-Net dataset metadata.

#### Structure
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
    "numTraining": null,
    "file_ending": null
}
```

#### Usage
- Populated automatically by build scripts with dataset-specific values
- Defines channel information (T2-weighted MRI)
- Specifies label mapping (Background=0, Transition Zone=1, Peripheral Zone=2)
- Contains training set size and file format information

## Common Workflow

### 1. Environment Setup
```python
# Configure paths
RAW_DATA_ROOT = "/path/to/raw/dataset"
OUT_ROOT = "/path/to/nnUNet_raw/"
DATASET_NAME = "Dataset00X_name"
```

### 2. Data Discovery
```python
# Initialize DataAnalyzer for file discovery
analyzer = DataAnalyzer(RAW_DATA_ROOT)
analyzer.regex = "pattern_for_target_files"

# Collect image and label paths
image_paths = list(analyzer.file_paths_gen(image_directory))
label_paths = list(analyzer.get_files(label_directory, regex_pattern))
```

### 3. Output Path Generation
```python
# Generate nnU-Net compatible filenames
out_images = [create_filename(output_dir, idx, prefix, ending, "_0000") 
              for idx in range(len(image_paths))]
out_labels = [create_filename(output_dir, idx, prefix, ending, "") 
              for idx in range(len(label_paths))]
```

### 4. JSON Metadata Creation
```python
# Create preprocessing tracking JSON
preprocessing_json = {
    "imagesTr": {input_path: output_path for input, output in zip(...)},
    "labelsTr": {input_path: output_path for input, output in zip(...)},
    "metadata": {"pipeline_info": ..., "timing": ...}
}

# Create nnU-Net dataset JSON
dataset_json = load_template_and_populate()
```

### 5. Pipeline Execution
```python
# Define separate pipelines for images and labels
image_pipeline = Pipeline().add(load_image).add(preprocessing_steps...)
label_pipeline = Pipeline().add(load_image).add(label_specific_steps...)

# Process in parallel batches
paired_results = preprocess_pairs_parallel(
    img_lbl_pairs, image_pipeline, label_pipeline, workers=cpu_count()
)
```

### 6. Output and Validation
```python
# Save processed data
save_pairs_parallel(paired_results, out_images, out_labels)

# Visual validation
show_random_from_json(preprocessing_json_file)
```

## Dataset-Specific Considerations

### PICAI Dataset
- **Multi-fold structure**: Processes 5 cross-validation folds
- **Large scale**: ~1000+ cases requiring batch processing
- **Memory management**: 150 pairs per batch to prevent OOM
- **Label correspondence**: Strict validation between images and zonal masks

### Decathlon Dataset
- **Standardized format**: Already in medical challenge format
- **Orientation issues**: Requires reorientation to standard anatomical directions
- **Smaller scale**: ~30-50 cases, single-batch processing
- **Whole gland labels**: Combined TZ+PZ segmentation masks

### Prostate158 Dataset
- **Multi-reader annotations**: Multiple expert readers available
- **Label convention differences**: Requires zone value swapping
- **Combined sets**: Merges training and test data for larger training set
- **Higher crop retention**: 75% ROI to preserve anatomy better

## Output Structure

All build scripts generate nnU-Net compatible directory structures:

```
nnUNet_raw/
├── Dataset00X_name/
│   ├── dataset.json          # nnU-Net metadata
│   ├── preprocessing.json    # Processing tracking
│   ├── imagesTr/            # Training images
│   │   ├── name_0000_0000.nii.gz
│   │   ├── name_0001_0000.nii.gz
│   │   └── ...
│   └── labelsTr/            # Training labels
│       ├── name_0000.nii.gz
│       ├── name_0001.nii.gz
│       └── ...
```

## Performance Optimization

### Memory Management
- Batch processing for large datasets (PICAI)
- Memory cleanup between batches (`del` statements)
- Garbage collection of intermediate results

### Parallel Processing
- Uses all available CPU cores by default
- Separate parallelization for preprocessing and I/O
- Progress tracking with `tqdm` for long-running operations

### Error Prevention
- Image-label correspondence validation
- Path existence checks before processing
- JSON validation and backup creation

## Usage Instructions

### Running Dataset Building Scripts

```bash
# PICAI dataset
python loadingData/build_data_picai.py

# Decathlon dataset  
python loadingData/build_data_decathlon.py

# Prostate158 dataset
python loadingData/build_data_158.py
```

### Customizing Processing Parameters

Edit the configuration section in each script:

```python
# Modify these parameters as needed
spacing = (0.5, 0.5, 3.0)  # Resampling spacing
crop_factor = 0.65         # ROI crop factor
bool_swap_mask_values = False  # Label value swapping
```

### Output Verification

Each script generates:
1. **Processing logs**: Console output with progress and statistics
2. **JSON metadata**: Tracking files for reproducibility
3. **Visual samples**: Random image-label pairs for quality check
4. **File counts**: Verification of processed data quantities

## Dependencies  

- **DataAnalyzer**: File discovery and metadata extraction
- **preprocessing module**: Image processing pipelines
- **SimpleITK**: Medical image I/O and processing
- **concurrent.futures**: Parallel processing
- **tqdm**: Progress tracking
- **json**: Metadata serialization

## Error Handling

Common issues and solutions:

- **Path not found**: Verify `RAW_DATA_ROOT` and dataset directory structure
- **Memory errors**: Reduce batch size or available workers
- **Label mismatch**: Check regex patterns and file naming conventions
- **Processing failures**: Review preprocessing pipeline compatibility with data format