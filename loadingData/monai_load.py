# pip install monai

# TODO create a list of all the files in the dataset, and then use that to load the data.

#https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/notebooks/msd_datalist_generator.ipynb

# TODO for nnUnet, we need to create a datalist with the following structure:
# - imagesTr: list of training images
# - imagesTs: list of test images
# - labelsTr: list of training labels

from monai.data import Dataset, DataLoader
from monai.transforms import LoadImaged, EnsureChannelFirstd, ToTensord, Compose
from exploratoryAnalysis.DataAnalyzer import DataAnalyzer
import glob
import os


from preprocessing.Pipeline import Pipeline  # Assuming your Pipeline class is in pipeline.py
from preprocessing.PreProcessor import *
from exploratoryAnalysis.DataAnalyzer import DataAnalyzer
from Utils import visualize_dicom_slider
import SimpleITK as sitk
import random

# Initialize  pipeline
crop_factor = 0.75 # according to data analysis for 158
spacing = (0.5, 0.5, 0.5)  # Resampling factor for x, y, z dimensions
        
# Initialize DataAnalyzer
analyzer = DataAnalyzer("/home/guest/work/Datasets")

# Define paths and regex
paths = {
    "158_train": "prostate158/prostate158_train/train",
    "158_test": "prostate158/prostate158_test/test"
}

analyzer.regex = "t2.nii.gz"
image_paths = list(analyzer.file_paths_gen(paths["158_test"]))

# BEWARE!!!! there are two anatomy readers...
analyzer.regex = "t2_anatomy_reader2.*"
label_paths = list(analyzer.file_paths_gen(paths["158_test"]))

pipeline_images = Pipeline()
pipeline_images.show_progress = True  # Enable progress bar for image processing
pipeline_images.add(load_image) \
        .add(resample_image, interpolator=sitk.sitkLinear, out_spacing=spacing) \
        .add(get_region_of_interest, crop=crop_factor) \
        .add(n4_bias_field_correction) \

pipeline_labels = Pipeline()
pipeline_labels.show_progress = True  # Enable progress bar for label processing
pipeline_labels.add(load_image) \
        .add(resample_image, interpolator=sitk.sitkNearestNeighbor, out_spacing=spacing) \
        .add(get_region_of_interest, crop=crop_factor) \


# Preprocess images and labels using the pipeline, using parallel processing
images_preprocessed = pipeline_images(image_paths)
labels_preprocessed = pipeline_labels(label_paths)
print(f"Processed {len(images_preprocessed)} images and {len(labels_preprocessed)} labels.")

out_root = "/media/guest/PORT-DISK/Datasets/"

output_paths = {
    "images": os.path.join(out_root, "nnUNet_raw/Dataset001_prostate158/imagesTr"),
    "labels": os.path.join(out_root, "nnUNet_raw/Dataset001_prostate158/labelsTr")
}

# Ensure output directories exist
os.makedirs(output_paths["images"], exist_ok=True)
os.makedirs(output_paths["labels"], exist_ok=True)

out_images = save_images(images_preprocessed, output_paths["images"], 
            prefix="pros158_", ending=".nii.gz", channel_id="_0000")

out_labels = save_images(labels_preprocessed, output_paths["labels"], 
            prefix="pros158_", ending=".nii.gz", channel_id="")

# test loading 
from exploratoryAnalysis.DataAnalyzer import DataAnalyzer

da = DataAnalyzer(out_root)

rand = random.choices(out_labels, k=10)  
for i in rand:
    #image, label = images_preprocessed[i], labels_preprocessed[i]
    image = i
    print(type(image))
    if isinstance(image, sitk.Image):
        print(image.GetSize(), image.GetSpacing())
    visualize_dicom_slider(image)
    #visualize_dicom_slider(label)






