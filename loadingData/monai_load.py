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

# # 1. scan your folders
# DATA_ROOT = "/home/guest/work/Datasets"
# paths = {
#     "158_train" : "prostate158/prostate158_train/train",
#     "158_test" : "prostate158/prostate158_test/test"
# }
# da = DataAnalyzer(DATA_ROOT)

# da.regex = "t2.nii.gz" 
# imgs = list(da.file_paths_gen(paths["158_test"]))

# da.regex = "t2_anatomy_reader.*"
# labs = list(da.file_paths_gen(paths["158_test"]))

# data = [{"image": i, "label": l} for i,l in zip(imgs, labs)]
# print("Number of training samples:", len(data))
# for i in data[:10]:
#     print(i)

# # 2. minimal transforms
# transf_for_loading = Compose([
#     LoadImaged(keys=["image","label"]),
#     EnsureChannelFirstd(keys=["image","label"]),
#     ToTensord(keys=["image","label"]),
# ])

# # 3. dataset + loader
# ds = Dataset(data, transform=transf_for_loading)
# loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=True)

# # 4. load & check once
# batch = next(iter(loader))
# print("Image tensor shape:", batch["image"].shape)
# print("Label tensor shape:", batch["label"].shape)
# #  visualize a middle slice
# import matplotlib.pyplot as plt

# def visualize_slice(image_tensor, slice_index):
#     plt.imshow(image_tensor[0, 0, :, :, slice_index], cmap='gray')
#     plt.axis('off')
#     plt.title(f"Slice {slice_index}")
#     plt.show()

# # Visualize the middle slice of the first image in the batch
# image_tensor = batch["label"]
# middle_slice_index = image_tensor.shape[-1] // 2

# visualize_slice(image_tensor, middle_slice_index)

from monai.data import Dataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, ToTensord, ToMetaTensord
from preprocessing.Pipeline import Pipeline  # Assuming your Pipeline class is in pipeline.py
from preprocessing.PreProcessor import *
from exploratoryAnalysis.DataAnalyzer import DataAnalyzer
from Utils import visualize_dicom_slider

# Initialize  pipeline
pipeline = Pipeline()
pipeline.add(load_image) \
        .add(resample_image) \
        .add(get_region_of_interest, crop=0.6) \
        .add(n4_bias_field_correction) \
        .add(to_array)

# Initialize DataAnalyzer
analyzer = DataAnalyzer("/home/guest/work/Datasets")

# Define paths and regex
paths = {
    "158_train": "prostate158/prostate158_train/train",
    "158_test": "prostate158/prostate158_test/test"
}
analyzer.regex = "t2.nii.gz"
image_paths = list(analyzer.file_paths_gen(paths["158_test"]))

analyzer.regex = "t2_anatomy_reader.*"
label_paths = list(analyzer.file_paths_gen(paths["158_test"]))

# Preprocess images and labels using the pipeline
data = []
for img_path, lbl_path in zip(image_paths, label_paths):
    processed_image = pipeline(img_path)  # Apply pipeline to image
    processed_label = pipeline(lbl_path)  # Apply pipeline to label
    # ensure the images (now ndarrays) have metadata and channel dims
    processed_image = processed_image[None, ...]  # Add channel dimension
    processed_label = processed_label[None, ...]  # Add channel dimension

    print("Processed image shape:", processed_image.shape)
    print("Processed label shape:", processed_label.shape)
    data.append({"image": processed_image, "label": processed_label})


# Define MONAI transforms
transforms = Compose([
    # ToMetaTensord(keys=["image", "label"]),  # Convert to metadata tensors
    # EnsureChannelFirstd(keys=["image", "label"]),
    ToTensord(keys=["image", "label"]),
])

# Create MONAI Dataset and DataLoader
monai_dataset = Dataset(data=data, transform=transforms)
data_loader = DataLoader(monai_dataset, batch_size=1, num_workers=2, shuffle=True)

# 4. load & check once
batch = next(iter(data_loader))
print("Image tensor shape:", batch["image"].shape)
print("Label tensor shape:", batch["label"].shape)

#  visualize a middle slice
import matplotlib.pyplot as plt

def visualize_slice(image_tensor, slice_index):
    plt.imshow(image_tensor[0, 0, slice_index, :, :], cmap='gray')
    plt.axis('off')
    plt.title(f"Slice {slice_index}")
    plt.show()

# Visualize the middle slice of the first image in the batch

for i in range(5):
    image_tensor, label_tensor = next(iter(data_loader)).values()
    # turn metatensor in to a ndarray
    image_tensor , label_tensor = image_tensor.numpy(), label_tensor.numpy()
    print(type(image_tensor))
    middle_slice_index = image_tensor.shape[2] // 2
    #visualize_slice(image_tensor, middle_slice_index)
    visualize_dicom_slider(image_tensor)
    visualize_dicom_slider(label_tensor)





