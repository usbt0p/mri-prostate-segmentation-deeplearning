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

- TODO figure out if different prepsocessing steps are order - invariant,
or if they should be applied in a specific order.
for example, if we apply n4 bias field correction, should we apply it before or after
extracting the region of interest? after = less information to do the correction, but
faster, before = more information, but slower.
should alignignement / registration be done before or after the region of interest extraction?

- TODO how to check proper functioning of the methods? analyze the intensity distribution?

- TODO determine the upscaling / downscaling process:
order matters here, when to do it?

- TODO in resampling: must conserve the quality, i.e. not resample to a bigger
voxel size or details will be lost!!
do data exploration to find the voxel size counts and usee it to inform the new size

- TODO other stuff:
1. download the remaining datasets (script)
2. make some analysis of them
3. make a pipeline system / figure out if there is one that exists and lets you
4. make a mask-joiner that creates whole gland masks from the two zonal ones
use custom. 
5. make a resizing function to cast all images to same size after cropping. See if it can be done in the resampling step.
6. ensure that the convetion in the masks is the same: pz is 1, tz is 2, and background is 0.
7. abstract the path dictionary to the analyzer, or make a config file or data loader class...

## Advancement logs

### Data exploration
- Notebook for data exploration on PICAI dataset has been created. Data exploration includes:
    - Basic statistics of the dataset.
    - Visualization of the metadata distribution (vendor, mri type, orientation, etc.).
    - Analysis of the voxel size and shape of the images.
    - Analysis of the number of images per patient.
    - Analysis of the distribution of image sizes (resolution).
    - Analysis of the intensity distribution of some images
    - Analysis of the approximate image center around which to set the ROI.
- Notebook for data exploration on Prostate158 dataset, similar to the previous one, has been created.

### Preprocessing

- Background masks and registering have been removed from the preprocessing as they are not needed.
- Normalization (zscore or minmax) has ben implemented.
- N4 bias field correction has been implemented.
- Image loading from path, with 3D en casting of 4D images has been implemented.
- Image cropping / region of interest extraction has been implemented.
- Resampling has been implemented.
- Simple image description for debugging has been implemented.
- Visual inspection tests have been implemented for the available preprocessing steps. They show images before and after the preprocessing step, as well as some data like intensity distribution and shape.