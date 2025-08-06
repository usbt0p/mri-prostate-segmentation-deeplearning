# LoadingData Module

This module provides dataset building scripts for preprocessing and converting multiple prostate MRI datasets into nnU-Net compatible format. It handles three major datasets: PICAI, Decathlon Task05_Prostate, and Prostate158, each with specific preprocessing pipelines and data structures.

## Overview

The loadingData module consists of dataset-specific build scripts that:

- Load raw medical imaging data from various dataset formats
- Apply consistent preprocessing pipelines using the preprocessing module
- Convert data to nnU-Net format with proper metadata
- Generate JSON configuration files for training frameworks
- Process data in parallel batches for memory efficiency

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

## Error Handling

Common issues and solutions:

- **Path not found**: Verify `RAW_DATA_ROOT` and dataset directory structure
- **Memory errors**: Reduce batch size or available workers
- **Label mismatch**: Check regex patterns and file naming conventions
- **Processing failures**: Review preprocessing pipeline compatibility with data format
