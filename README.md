# Anatomical Segmentation of the Prostate Gland from MRI Images using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a comprehensive pipeline for anatomical segmentation of the prostate gland from MRI images using deep learning techniques. 
It focuses on multi-dataset preprocessing, analysis, and preparation for training segmentation models compatible with frameworks like nnU-Net and MONAI.
A training script is also provided, using U-Net, altough the convergence and metrics are on par with nnUnet.
Notebooks with dataset exploration are provided, as well as modules for data analysis and manipulation.

### Key Features

- **Multi-dataset support**: Works with PICAI, Prostate158, and Medical Segmentation Decathlon datasets
- **Comprehensive preprocessing pipeline**: Includes resampling, ROI extraction, N4 bias correction, and other data preparation steps
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
- CUDA-compatible GPU (for training, preprocessing can be done in CPU in reasonable time)

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
You might want to download the rest of the datasets as well.

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

Use the `DataAnalyzer` class to explore datasets. Example:

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

Create custom preprocessing pipelines. Example:

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

Process large datasets efficiently. Example:

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

- **`load_image(path)`**: Load medical images with 4D→3D conversion if needed
- **`resample_image(image, out_spacing, interpolator)`**: Resample to target spacing
- **`get_region_of_interest(image, crop)`**: Extract ROI around prostate
- **`n4_bias_field_correction(image)`**: Remove MRI bias field artifacts
- **`normalize_image(image, method)`**: Z-score or min-max normalization
- **`combine_zonal_masks(mask, pz_val, tz_val)`**: Combine PZ and TZ masks to form a WG mask
- **`swap_zonal_mask_values(mask, val1, val2)`**: Swap mask label values
- **`reorient_image(image, orientation)`**: Reorient to standard orientation


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

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PI-CAI Challenge](https://pi-cai.grand-challenge.org/) for providing the PICAI dataset
- [Prostate158](https://github.com/kbressem/prostate158) dataset contributors
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) organizers
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework developers
- [MONAI](https://monai.io/) community for medical imaging AI tools

---

### Citations

For training, the annotations derived from Bosma et al. where used.
https://grand-challenge.org/algorithms/prostate-segmentation/
```
@article{PICAI_Study_design, author={Anindo Saha AND Jasper J. Twilt AND Joeran S. Bosma AND Bram van Ginneken AND Derya Yakar AND Mattijs Elschot AND Jeroen Veltman AND Jurgen Fütterer AND Maarten de Rooij AND Henkjan Huisman}, title={{Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)}}, year={2022}, doi={10.5281/zenodo.6667655} }
```
