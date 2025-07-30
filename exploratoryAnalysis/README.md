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

### Dataset-Specific Notebooks

#### explore_picai.ipynb

Comprehensive analysis of the PI-CAI dataset focusing on prostate cancer detection.

**Key Analyses:**
- **Data Structure Validation**: Verifies 5-file structure per patient (T2W, DWI, ADC, sagittal, coronal)
- **Label Mapping Analysis**: Confirms TZ=1, PZ=2 convention in zonal masks
- **Mask Comparison**: Analyzes differences between whole gland and zonal segmentations
- **Metadata Exploration**: Vendor distribution (SIEMENS vs Philips), scanner models, PSA values
- **Resolution Analysis**: Image dimensions and spacing distributions by orientation
- **Intensity Analysis**: T2W image histogram patterns for normalization guidance
- **ROI Optimization**: Bounding box analysis to determine optimal crop factors (0.6-0.65)

**Key Findings:**
- TZ labeled as 1, PZ labeled as 2
- Optimal ROI crop factor: 0.65 (65% retention around center)
- Most prostates fit within 50-60% of image extent
- Multiple scanner vendors with varying resolutions

#### explore_decathlon.ipynb

Analysis of the Medical Segmentation Decathlon Task05_Prostate dataset.

**Key Analyses:**
- **4D to 3D Conversion**: Handles multi-channel images (T2W + ADC)
- **Label Convention**: Analyzes zonal labeling (TZ=2, PZ=1 - inverted from PICAI)
- **Quality Assessment**: Identifies corrupted files and empty masks
- **Orientation Analysis**: Single anatomical orientation (axial only)
- **Resolution Uniformity**: Predominantly consistent 320×320 resolution
- **Preprocessing Requirements**: Need for image reorientation to standard directions

**Key Findings:**
- Smaller dataset (~32 cases) with good quality
- Requires label value swapping for consistency with PICAI
- Optimal ROI crop factor: 0.55 (tighter crop due to consistent anatomy)
- Single-batch processing suitable due to dataset size

#### explore_158.ipynb

Analysis of the Prostate158 dataset with multi-reader annotations.

**Key Analyses:**
- **Multi-Reader Setup**: Analyzes reader1 vs reader2 anatomical annotations
- **File Structure Validation**: Different file counts due to tumor annotations
- **Label Convention**: TZ=2, PZ=1 (requires swapping for consistency)
- **Resolution Diversity**: Two main resolution groups (270×270 vs 442×442)
- **Inter-Reader Variability**: DICE score analysis between readers
- **ROI Requirements**: Larger bounding boxes requiring higher crop retention

**Key Findings:**
- High-quality anatomical annotations with good inter-reader agreement
- Optimal ROI crop factor: 0.75 (75% retention due to larger anatomy extent)
- Two distinct acquisition protocols with different resolutions
- Consistent 3.0mm slice thickness across dataset

## Common Analysis Workflows

### Dataset Exploration Workflow

```python
# 1. Initialize analyzer
analyzer = DataAnalyzer("/path/to/dataset/root")

# 2. Set appropriate regex for dataset
analyzer.regex = ".*_t2w.mha$"  # For PICAI
# analyzer.regex = "^p"          # For Decathlon  
# analyzer.regex = "t2.nii.gz"   # For Prostate158

# 3. Collect comprehensive metadata
df = analyzer.collect_metadata_from_subdirs("dataset/folder")

# 4. Analyze data distributions
print("Dimension distribution:")
print(df['dim_size'].value_counts())
print("Spacing distribution:")
print(df['spacing'].value_counts())

# 5. Visualize random samples
random_files = analyzer.pick_random("image/folder", 3)
for file in random_files:
    analyzer.show_image(file)
    analyzer.image_intensity_histogram(file, plot=True)
```

### ROI Analysis Workflow

```python
# 1. Calculate bounding boxes for all masks
import concurrent.futures
import pandas as pd

with concurrent.futures.ProcessPoolExecutor() as executor:
    records = list(executor.map(
        analyzer.calculate_cube_bounds,
        analyzer.get_files("mask/folder", ".*\.nii\.gz$")
    ))

# 2. Create analysis dataframe
df = pd.DataFrame(records, columns=[
    'start_z', 'end_z', 'start_y', 'end_y', 'start_x', 'end_x',
    'mask_path', 'bounding_box_size', 'proportion_of_image_size'
])

# 3. Analyze ROI distributions
df['proportion_of_image_size'].hist(bins=32)
plt.title("ROI Size Distribution")
plt.show()

# 4. Determine optimal crop factor
percentile_95 = df['proportion_of_image_size'].quantile(0.95)
recommended_crop = percentile_95 + 0.05  # Add 5% margin
print(f"Recommended crop factor: {recommended_crop:.2f}")
```

### Quality Assessment Workflow

```python
# 1. Check for empty masks
non_empty, empty_list = analyzer.count_and_find_non_empty_masks("mask/folder")
print(f"Found {len(empty_list)} empty masks out of {non_empty + len(empty_list)}")

# 2. Validate image-label correspondence
def validate_correspondence(image_paths, label_paths):
    for img_path, lbl_path in zip(image_paths, label_paths):
        img = sitk.ReadImage(img_path)
        lbl = sitk.ReadImage(lbl_path)
        
        if img.GetSize() != lbl.GetSize():
            print(f"Size mismatch: {img_path} vs {lbl_path}")
        if img.GetSpacing() != lbl.GetSpacing():
            print(f"Spacing mismatch: {img_path} vs {lbl_path}")

# 3. Analyze intensity distributions
intensity_stats = []
for file in random_sample_files:
    hist, bins = analyzer.image_intensity_histogram(file)
    stats = {
        'file': file,
        'mean': np.mean(hist),
        'std': np.std(hist),
        'max': np.max(hist)
    }
    intensity_stats.append(stats)
```

## Dataset-Specific Configurations

### PICAI Dataset
```python
analyzer = DataAnalyzer("/path/to/picai")
analyzer.regex = ".*_t2w.mha$"

# Optimal preprocessing parameters
SPACING = (0.5, 0.5, 3.0)
CROP_FACTOR = 0.65
LABEL_MAPPING = {"TZ": 1, "PZ": 2}  # Standard
BATCH_SIZE = 150  # For memory management
```

### Decathlon Dataset
```python
analyzer = DataAnalyzer("/path/to/decathlon")
analyzer.regex = "^p"

# Optimal preprocessing parameters  
SPACING = (0.5, 0.5, 3.0)
CROP_FACTOR = 0.55
LABEL_MAPPING = {"TZ": 2, "PZ": 1}  # Requires swapping
REORIENTATION = "RPS"  # Right-Posterior-Superior
```

### Prostate158 Dataset
```python
analyzer = DataAnalyzer("/path/to/prostate158")
analyzer.regex = "t2.nii.gz"

# Optimal preprocessing parameters
SPACING = (0.5, 0.5, 3.0)
CROP_FACTOR = 0.75
LABEL_MAPPING = {"TZ": 2, "PZ": 1}  # Requires swapping
READER_SELECTION = "reader1"  # Primary annotation source
```

## Integration with Preprocessing Pipeline

The DataAnalyzer findings directly inform preprocessing decisions:

```python
# Use analysis results to configure preprocessing
from preprocessing.Pipeline import Pipeline
from preprocessing.PreProcessor import *

# Configure based on dataset analysis
pipeline = Pipeline()
pipeline.add(load_image) \
        .add(resample_image, out_spacing=RECOMMENDED_SPACING) \
        .add(get_region_of_interest, crop=OPTIMAL_CROP_FACTOR) \
        .add(n4_bias_field_correction)

# Apply label corrections based on analysis
if REQUIRES_LABEL_SWAPPING:
    label_pipeline.add(swap_zonal_mask_values)
```

## Advanced Analysis Features

### Interactive Visualization

Using Jupyter widgets for dynamic exploration:

```python
from ipywidgets import interact, IntSlider

# Interactive slice navigation
def show_slice(slice_idx):
    analyzer.show_image(image_path, slice=slice_idx)

interact(show_slice, slice_idx=IntSlider(min=0, max=num_slices-1))
```

### Statistical Reporting

Generate comprehensive dataset reports:

```python
def generate_dataset_report(analyzer, dataset_path):
    df = analyzer.collect_metadata_from_subdirs(dataset_path)
    
    report = {
        'total_files': len(df),
        'unique_dimensions': df['dim_size'].nunique(),
        'spacing_variations': df['spacing'].nunique(),
        'vendor_distribution': df['vendor'].value_counts().to_dict(),
        'mean_prostate_volume': df['prostate_volume'].mean()
    }
    
    return report
```

## Dependencies

- **SimpleITK**: Medical image I/O and processing
- **pandas**: Data analysis and manipulation
- **matplotlib**: Visualization and plotting
- **numpy**: Numerical operations
- **concurrent.futures**: Parallel processing
- **ipywidgets**: Interactive Jupyter notebook widgets
- **regex**: Pattern matching for file discovery

## Output and Reporting

The module generates various outputs for dataset understanding:

1. **Metadata DataFrames**: Comprehensive tabular data about image properties
2. **Statistical Plots**: Histograms, distributions, and quality metrics
3. **Visual Samples**: Representative images and masks for quality assessment
4. **ROI Analysis**: Bounding box statistics for preprocessing optimization
5. **Quality Reports**: Empty mask detection and data validation results

## Best Practices

### Memory Management
- Use parallel processing for large datasets
- Process files in batches when memory is limited
- Clear large objects after analysis

### Data Validation
- Always validate image-label correspondence
- Check for corrupted or unreadable files
- Verify label value conventions across datasets

### Analysis Reproducibility
- Set random seeds for consistent sampling
- Document regex patterns and analysis parameters
- Save analysis results for preprocessing pipeline configuration

## Usage Examples

### Basic Dataset Exploration
```bash
# Run Jupyter notebooks for specific datasets
jupyter notebook exploratoryAnalysis/explore_picai.ipynb
jupyter notebook exploratoryAnalysis/explore_decathlon.ipynb  
jupyter notebook exploratoryAnalysis/explore_158.ipynb
```

### Programmatic Analysis
```python
from exploratoryAnalysis.DataAnalyzer import DataAnalyzer

# Initialize and configure
analyzer = DataAnalyzer("/path/to/data")
analyzer.regex = "target_pattern"

# Run comprehensive analysis
metadata_df = analyzer.collect_metadata_from_subdirs("dataset")
roi_analysis = perform_roi_analysis(analyzer, "masks")
quality_report = generate_quality_report(analyzer, "images", "labels")
```

## Dataset Summary Table

The following table summarizes the key characteristics and analysis findings for each dataset:

| Dataset | Sample Count | Scanner Variability | Resolution Variations | Label Convention | Optimal Crop Factor | Spacing (mm) | Key Preprocessing Notes |
|---------|-------------|-------------------|---------------------|------------------|-------------------|--------------|------------------------|
| **PICAI** | ~1,500 cases<br>(5 folds) | **High variability**<br>• SIEMENS (Prisma, Skyra)<br>• Philips Medical Systems<br>• Multiple scanner models | **Multiple resolutions**<br>• T2W: 384×384, 640×640<br>• Sagittal: 320×320<br>• Coronal: 320×320 | **Standard**<br>• TZ = 1<br>• PZ = 2<br>• Background = 0 | **0.65**<br>(65% retention) | **(0.5, 0.5, 3.0)** | • Batch processing (150 pairs)<br>• Multi-fold cross-validation<br>• Memory management required<br>• 5 imaging sequences per case |
| **Decathlon** | 32 cases<br>(training) | **Low variability**<br>• Standardized format<br>• Consistent acquisition<br>• Single protocol | **Mostly uniform**<br>• Primary: 320×320<br>• Some: 256×256, 384×384<br>• 4D format (T2W + ADC) | **Inverted**<br>• TZ = 2<br>• PZ = 1<br>• Background = 0 | **0.55**<br>(55% retention) | **(0.5, 0.5, 3.0)** | • Requires reorientation to "RPS"<br>• Label value swapping needed<br>• 4D to 3D conversion<br>• Single-batch processing |
| **Prostate158** | 139 cases<br>(train + test) | **Moderate variability**<br>• Two acquisition protocols<br>• Consistent within protocol<br>• Multi-reader annotations | **Two main groups**<br>• Group 1: 270×270<br>• Group 2: 442×442<br>• Consistent slice thickness | **Inverted**<br>• TZ = 2<br>• PZ = 1<br>• Background = 0 | **0.75**<br>(75% retention) | **(0.5, 0.5, 3.0)** | • Label value swapping needed<br>• Larger anatomy extent<br>• Reader1 annotations preferred<br>• High inter-reader agreement |

### Dataset Quality Metrics

| Dataset | Empty Masks | Data Completeness | Intensity Consistency | Anatomical Coverage |
|---------|-------------|------------------|---------------------|-------------------|
| **PICAI** | 0 empty masks | ✅ Complete<br>(5 sequences per case) | 🔶 Variable<br>(multi-vendor) | 🔶 Standard<br>(some large prostates) |
| **Decathlon** | 3 corrupted files | ✅ Good<br>(standardized format) | ✅ Consistent<br>(single protocol) | ✅ Uniform<br>(tight distribution) |
| **Prostate158** | 0 empty masks | ✅ Complete<br>(multi-reader validated) | ✅ Good<br>(two protocols) | 🔶 Variable<br>(larger anatomy range) |

### Preprocessing Pipeline Recommendations

Based on the analysis findings, the recommended preprocessing configurations are:

| Dataset | Pipeline Configuration | Special Considerations |
|---------|----------------------|----------------------|
| **PICAI** | `load_image` → `resample_image(0.5,0.5,3.0)` → `get_ROI(0.65)` → `n4_correction` | • Use batch processing<br>• Monitor memory usage<br>• Validate image-label correspondence |
| **Decathlon** | `load_image` → `reorient_image("RPS")` → `resample_image(0.5,0.5,3.0)` → `get_ROI(0.55)` → `n4_correction` | • Apply label swapping<br>• Handle 4D input format<br>• Single-pass processing |
| **Prostate158** | `load_image` → `resample_image(0.5,0.5,3.0)` → `get_ROI(0.75)` → `n4_correction` | • Apply label swapping<br>• Use reader1 annotations<br>• Larger crop retention |

### Key Insights from Analysis

1. **Scanner Variability Impact**: PICAI shows highest variability requiring robust normalization, while Decathlon provides most consistent data quality.

2. **ROI Optimization**: Crop factors vary significantly (0.55-0.75) based on prostate size distributions in each dataset.

3. **Label Harmonization**: Decathlon and Prostate158 require label value swapping to match PICAI convention for multi-dataset training.

4. **Processing Scalability**: PICAI requires batch processing due to size (~1,500 cases), while others can be processed in single batches.

5. **Data Quality**: All datasets provide high-quality annotations with minimal empty masks, suitable for deep learning training.

The exploratoryAnalysis module provides the foundation for understanding medical imaging datasets and making informed preprocessing decisions for deep learning model training.