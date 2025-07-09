# Anatomical segmentation of the prostate gland from MRI images using Deep Learning techniques and multiple datasets

## Overview
This repository contains code that aims to provide a comprehensive pipeline for the anatomical segmentation of the prostate gland from MRI images using Deep Learning techniques. The code is designed to work with multiple datasets and provides preprocessing steps, model training, and evaluation.

Several datasets have been used in this project, including:

| Dataset Name | Description | Link |
|--------------|-------------|------|
| PICAI        |             |      |
| # TODO    |             |      |

Notebooks with dataset exploration are provided, as well as modules for data analysis and manipulation, preprocessing, # TODO ...

## Installation and Usage
    TODO

## TODO's

### DOUBTS

- TODO figure out if different prepsocessing steps are order - invariant,
or if they should be applied in a specific order.
for example, if we apply n4 bias field correction, should we apply it before or after
extracting the region of interest? after = less information to do the correction, but
faster, before = more information, but slower.
should alignignement / registration be done before or after the region of interest extraction?

- TODO determine the upscaling / downscaling process:
order matters here, when to do it?

- TODO in resampling: must conserve the quality, i.e. not resample to a bigger
voxel size or details will be lost!!
do data exploration to find the voxel size counts and usee it to inform the new size

### TODO (Ordered by priority)
- debug weird behavior in preprocessing (resampling?) 
1. preprocess data and save it to structured folders (nnUnet Format)
1. save preprocessed datasets to disk
1. try loading a small dataset in monai (from disk)
1. download the remaining datasets (script)
2. make some analysis of them 
6. analyze masks for each dataset, ensure that the convetion in the masks is made the same: (pz is 1, tz is 2, and background is 0).
5. cast all images to same size after cropping. See if it can be done in the resampling. This might also be automatically performed by nnU-Net or other frameworks.
7. re-structure folders for nnUnet, remove unused files (non-mask or t2w)
8. fix small regressions at the start of the notebooks

## Advancement logs

### Data exploration
- A data exploration module and class has been created to provide a consistent interface for data exploration across different datasets.
- Notebook for data exploration on PICAI dataset has been created. Data exploration includes:
    - Basic statistics of the dataset.
    - Visualization of the metadata distribution (vendor, mri type, orientation, etc.).
    - Analysis of the voxel size and shape of the images.
    - Analysis of the number of images per patient.
    - Analysis of the distribution of image sizes (resolution).
    - Analysis of the intensity distribution of some images
    - Analysis of the approximate image center around which to set the ROI.
    - Visualization of the centered bounding box around the prostate gland.
- Notebook for data exploration on Prostate158 dataset, similar to the previous one, has been created.

### Preprocessing

- Background masks and registering have been removed from the preprocessing as they are not needed.
- Normalization (zscore or minmax) has ben implemented.
- N4 bias field correction has been implemented.
- Image loading from path, with 3D en casting of 4D images has been implemented.
- Image cropping / region of interest extraction has been implemented.
- Resampling has been implemented.
- Zonal mask combination has been implemented, which combines the two zonal masks into a single whole gland mask.
- Zonal mask value swapping has been implemented, which swaps the values of the zonal masks (e.g. pz and tz) to a single value. This is needed to keep consitency in the masks labels troughout different datasets.
- Simple image description for debugging has been implemented (e.g. image shape, voxel size, etc.).
- Visual inspection tests have been implemented for the available preprocessing steps. They show images before and after the preprocessing step, as well as some data like intensity distribution and shape. These tests are not meant to be exhaustive, but rather to provide a quick visual check of the preprocessing steps.
-  A pipeline system has been implemented to allow for easy chaining of preprocessing steps. 
- Methods have been implemented to preprocess the images and labels in pairs, in parallel, and to save them also in parallel, ensuring that the images always match their labels.


## Data wrangling

``` python
# this is nnUnet's expected metadata json
{
    "channel_names": {"0": "T2"},  # we have only one channel, so we use "0" as the key
    "labels": {  # IMPORTANT this is our label mapping
        "background": 0,
        "TZ": 1, # use convention from PI-CAI (it provides the most images)
        "PZ": 2,
    },
    "numTraining": len(image_paths),  # number of training images
    "file_ending": FILE_ENDING
    # "overwrite_image_reader_writer": "SimpleITKIO",  # optional! If not provided nnU-Net will automatically determine the ReaderWriter
}
```