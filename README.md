# Anatomical Segmentation of the Prostate Gland from MRI Images using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a comprehensive pipeline for anatomical segmentation of the prostate gland from MRI images using deep learning techniques. The project focuses on multi-dataset preprocessing, analysis, and preparation for training segmentation models compatible with frameworks like nnU-Net and MONAI.

### Key Features

- **Multi-dataset support**: Works with PICAI, Prostate158, and Medical Segmentation Decathlon datasets
- **Comprehensive preprocessing pipeline**: Includes resampling, ROI extraction, N4 bias correction, and normalization
- **Parallel processing**: Efficient batch processing with progress tracking
- **Data exploration tools**: Interactive analysis and visualization of medical imaging datasets
- **nnU-Net compatibility**: Automatic data structuring following nnU-Net conventions
- **Zonal segmentation**: Support for peripheral zone (PZ) and transition zone (TZ) segmentation

### Supported Datasets

| Dataset | Description | Cases | Modalities | Link |
|---------|-------------|-------|------------|----- |
| **PICAI** | PI-CAI Challenge dataset with expert annotations for prostate cancer detection | 1,500 public cases | T2w, DWI, ADC | [pi-cai.grand-challenge.org](https://pi-cai.grand-challenge.org/) |
| **Prostate158** | Expert-annotated 3T MRI dataset for anatomical zones and cancer detection | 158 cases | T2w, DWI, ADC | [GitHub](https://github.com/kbressem/prostate158) |
| **Medical Decathlon Task05** | Prostate segmentation task from Medical Segmentation Decathlon | 48 cases | T2w, ADC | [medicaldecathlon.com](http://medicaldecathlon.com/) |

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for processing large datasets)

### Dependencies Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mri-prostate-segmentation-deeplearning.git
cd mri-prostate-segmentation-deeplearning
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Core Dependencies

- **monai**: Medical imaging AI framework
- **nibabel**: NIfTI file reading and writing
- **simpleitk**: Medical image processing and analysis
- **matplotlib**: Data visualization
- **numpy**: Numerical computing
- **pandas**: Data analysis and manipulation
- **ipywidgets**: Interactive Jupyter notebook widgets

## Configuration

### Dataset Setup

1. **Download datasets** using the provided script:
```bash
./download.sh
```
This downloads the PICAI dataset folds from Zenodo.

2. **Configure data paths** in the preprocessing scripts:
```python
# Update paths in loadingData/build_data_*.py
RAW_DATA_ROOT = "/path/to/your/datasets"
OUT_ROOT = "/path/to/output/nnUNet_raw/"
```

3. **Set up output directories** for nnU-Net format:
```bash
mkdir -p /path/to/output/nnUNet_raw/Dataset001_picai/{imagesTr,labelsTr}
```

### Preprocessing Parameters

Key parameters can be configured in the build scripts:

```python
# Resampling spacing (x, y, z) in mm
spacing = (0.5, 0.5, 3.0)

# ROI crop factor (0.6 = 60% crop around center)
crop_factor = 0.65

# Label value swapping for consistency across datasets
bool_swap_mask_values = False
```

## Usage

### Quick Start

1. **Data Exploration**:
```bash
# Open Jupyter notebooks for dataset analysis
jupyter notebook exploratoryAnalysis/explore_picai.ipynb
```

2. **Dataset Preprocessing**:
```bash
# Preprocess PICAI dataset
python loadingData/build_data_picai.py

# Preprocess other datasets
python loadingData/build_data_decathlon.py
python loadingData/build_data_158.py
```

3. **Preprocessing Pipeline Testing**:
```bash
# Test individual preprocessing functions
python preprocessing/TestPreprocessing.py
```

### Detailed Workflow

#### 1. Dataset Exploration

Use the `DataAnalyzer` class to explore datasets:

```python
from exploratoryAnalysis.DataAnalyzer import DataAnalyzer

# Initialize analyzer
analyzer = DataAnalyzer("/path/to/datasets")
analyzer.regex = ".*_t2w.mha$"  # Filter for T2-weighted images

# Collect metadata
df = analyzer.collect_metadata_from_subdirs("picai_folds/picai_images_fold0")

# Visualize images
analyzer.show_image("path/to/image.mha", save="output.png")

# Generate intensity histograms
analyzer.image_intensity_histogram("path/to/image.mha", plot=True)
```

#### 2. Preprocessing Pipeline

Create custom preprocessing pipelines:

```python
from preprocessing.Pipeline import Pipeline
from preprocessing.PreProcessor import *

# Initialize pipeline
pipeline = Pipeline()

# Add preprocessing steps
pipeline.add(load_image) \
        .add(resample_image, interpolator=sitk.sitkLinear, out_spacing=(0.5, 0.5, 3.0)) \
        .add(get_region_of_interest, crop=0.65) \
        .add(n4_bias_field_correction) \
        .add(normalize_image, method="minmax")

# Process images
processed_images = pipeline.run(image_paths, parallel=True, max_workers=4)
```

#### 3. Batch Processing

Process large datasets efficiently:

```python
# Process image-label pairs in parallel
from preprocessing.Utils import preprocess_pairs_parallel, save_pairs_parallel

# Define pipelines for images and labels
img_pipeline = Pipeline().add(load_image).add(resample_image)
lbl_pipeline = Pipeline().add(load_image).add(resample_image, interpolator=sitk.sitkNearestNeighbor)

# Process pairs
paired_results = preprocess_pairs_parallel(
    list(zip(image_paths, label_paths)), 
    img_pipeline, 
    lbl_pipeline, 
    workers=8
)

# Save results
out_images, out_labels = save_pairs_parallel(
    paired_results, 
    output_image_paths, 
    output_label_paths, 
    workers=8
)
```

### Available Preprocessing Functions

- **`load_image(path)`**: Load medical images with 4D→3D conversion
- **`resample_image(image, out_spacing, interpolator)`**: Resample to target spacing
- **`get_region_of_interest(image, crop)`**: Extract ROI around prostate
- **`n4_bias_field_correction(image)`**: Remove MRI bias field artifacts
- **`normalize_image(image, method)`**: Z-score or min-max normalization
- **`combine_zonal_masks(mask, pz_val, tz_val)`**: Combine PZ and TZ masks
- **`swap_zonal_mask_values(mask, val1, val2)`**: Swap mask label values
- **`reorient_image(image, orientation)`**: Reorient to standard orientation

### Testing Preprocessing Functions

Run visual tests for preprocessing functions:

```python
# Test specific functions
python -c "from preprocessing.TestPreprocessing import *; roi_test()"
python -c "from preprocessing.TestPreprocessing import *; n4_test()"
python -c "from preprocessing.TestPreprocessing import *; normalization_test()"
```

### Output Format

The pipeline generates nnU-Net compatible datasets:

```
Dataset001_picai/
├── imagesTr/
│   ├── picai_0_0000.nii.gz
│   ├── picai_1_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── picai_0.nii.gz
│   ├── picai_1.nii.gz
│   └── ...
├── dataset.json
└── preprocessing.json
```

## Project Structure

```
.
├── exploratoryAnalysis/          # Dataset exploration and analysis
│   ├── DataAnalyzer.py           # Main analysis class
│   ├── explore_picai.ipynb       # PICAI dataset exploration
│   ├── explore_decathlon.ipynb   # Decathlon dataset exploration
│   └── explore_158.ipynb         # Prostate158 dataset exploration
├── loadingData/                  # Dataset preprocessing scripts
│   ├── build_data_picai.py       # PICAI preprocessing
│   ├── build_data_decathlon.py   # Decathlon preprocessing
│   ├── build_data_158.py         # Prostate158 preprocessing
│   └── data.template.json        # nnU-Net metadata template
├── preprocessing/                # Core preprocessing modules
│   ├── Pipeline.py               # Preprocessing pipeline framework
│   ├── PreProcessor.py           # Individual preprocessing functions
│   ├── Utils.py                  # Utility functions
│   └── TestPreprocessing.py      # Visual testing functions
├── requirements.txt              # Python dependencies
├── download.sh                   # Dataset download script
└── README.md                     # This file
```

## Features and Capabilities

### Data Exploration

- **Comprehensive dataset analysis**: The `DataAnalyzer` class provides a unified interface for exploring medical imaging datasets
- **Metadata extraction**: Automatically extracts DICOM metadata including vendor, MRI type, orientation, spacing, and clinical parameters
- **Statistical analysis**: Generates distribution plots for voxel sizes, image dimensions, and intensity values
- **Interactive visualization**: Built-in image display with slice selection and multi-image comparison
- **ROI analysis**: Calculates centered bounding boxes around prostate regions for optimal cropping
- **Parallel processing**: Multi-threaded metadata collection for large datasets

### Preprocessing Pipeline

- **Modular design**: Chain preprocessing functions using the `Pipeline` class
- **Medical image optimized**: Specialized functions for MRI preprocessing including:
  - **N4 bias field correction**: Removes scanner-induced intensity variations
  - **Intelligent resampling**: Preserves image quality while standardizing voxel spacing
  - **ROI extraction**: Crops images around prostate region to reduce computational load
  - **Intensity normalization**: Z-score or min-max normalization options
  - **4D to 3D conversion**: Handles multi-temporal or multi-contrast sequences

- **Label processing**: Specialized functions for segmentation masks:
  - **Zonal mask combination**: Merges peripheral and transition zone masks
  - **Label value consistency**: Standardizes mask values across different datasets
  - **Nearest neighbor resampling**: Preserves discrete label values during resampling

- **Quality assurance**: 
  - Visual testing functions for each preprocessing step
  - Image-label correspondence verification
  - Intensity distribution analysis before/after processing

### Parallel Processing

- **Batch processing**: Handles large datasets in memory-efficient batches
- **Multi-threading**: Parallel execution with progress tracking
- **Paired processing**: Ensures image-label correspondence during parallel operations
- **Configurable workers**: Adjustable thread count based on system resources

### Dataset Compatibility

- **Multi-format support**: Handles NIfTI (.nii.gz), DICOM, and other medical formats
- **Cross-dataset standardization**: Unified preprocessing for different data sources
- **nnU-Net integration**: Automatic output formatting for nnU-Net framework
- **Flexible labeling**: Configurable label mappings for different anatomical conventions


## Label Conventions and Data Format

### Segmentation Labels

The project follows PI-CAI labeling conventions for consistency:

```python
# Standard label mapping
labels = {
    "background": 0,    # Background/non-prostate tissue
    "TZ": 1,            # Transition Zone (central gland)
    "PZ": 2,            # Peripheral Zone
}
```

### nnU-Net Dataset Format

Output datasets follow nnU-Net conventions:

```json
{
    "channel_names": {"0": "T2"},
    "labels": {
        "background": 0,
        "TZ": 1,
        "PZ": 2
    },
    "numTraining": 1500,
    "file_ending": ".nii.gz"
}
```

### File Naming Convention

- **Images**: `{prefix}_{case_id}_0000.nii.gz` (channel 0 for T2-weighted)
- **Labels**: `{prefix}_{case_id}.nii.gz`
- **Example**: `picai_042_0000.nii.gz` (image), `picai_042.nii.gz` (label)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{prostate-segmentation-pipeline,
    title={MRI Prostate Segmentation Deep Learning Pipeline},
    author={Your Name},
    year={2024},
    url={https://github.com/your-username/mri-prostate-segmentation-deeplearning}
}
```

## Acknowledgments

- [PI-CAI Challenge](https://pi-cai.grand-challenge.org/) for providing the PICAI dataset
- [Prostate158](https://github.com/kbressem/prostate158) dataset contributors
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) organizers
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework developers
- [MONAI](https://monai.io/) community for medical imaging AI tools

## Support

For questions or issues:

1. Check the [documentation](./CLAUDE.md)
2. Search existing [issues](https://github.com/your-username/mri-prostate-segmentation-deeplearning/issues)
3. Create a new issue with detailed description

---

**Note**: This project is for research purposes. Please ensure you have appropriate permissions and follow ethical guidelines when working with medical data.

## Development Status and Roadmap

### Current Status ✅

- [x] Multi-dataset preprocessing pipeline
- [x] Parallel processing implementation
- [x] Data exploration and visualization tools
- [x] nnU-Net format compatibility
- [x] Visual testing framework
- [x] Comprehensive documentation

### Planned Features 🚧

- [ ] Deep learning model training integration
- [ ] Automated hyperparameter optimization
- [ ] Cross-validation framework
- [ ] Performance evaluation metrics
- [ ] Docker containerization
- [ ] Automated testing suite
- [ ] Additional dataset support

### Research Questions 🔬

- **Preprocessing order optimization**: Determining optimal sequence for N4 correction, ROI extraction, and resampling
- **Cross-dataset generalization**: Evaluating preprocessing parameters that work across all datasets
- **Quality preservation**: Balancing processing speed with image quality retention
- **Label consistency**: Standardizing anatomical zone definitions across different annotation schemes