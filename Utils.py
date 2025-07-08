'''
Module to dump utilities for various tasks.
Kind of acts as a toolbox / buffer for functions that are not yet
integrated into the main workflow but are useful somewhere
'''

from exploratoryAnalysis.DataAnalyzer import DataAnalyzer
from Utils import *

import os
import SimpleITK as sitk
import random
import json

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np

# da = DataAnalyzer(root)

# dirs_files = list(da.get_dirs(".", regex=r".*T2.*", out='rel'))

# for i in dirs_files:
#     files = list(da.get_files(i))
#     print(len(files), i)

def visualize_dicom_slider(image_or_path):
    """
    Visualize a DICOM image (or any Simple) or a series of slices with a slider.

    Parameters:
    - image_or_path: str or SimpleITK.Image
        If str, it should be the path to the DICOM file or directory.
        If SimpleITK.Image, it should be the loaded image object.
    """

    # Load the image if a path is provided
    if isinstance(image_or_path, str):
        try:
            image = sitk.ReadImage(image_or_path)
        except Exception:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(image_or_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
    elif isinstance(image_or_path, sitk.Image):
        image = image_or_path
    elif isinstance(image_or_path, np.ndarray):
        # If a numpy array is provided, convert it to a SimpleITK image
        if image_or_path.ndim > 3:
            # remove extra dimensions if necessary
            image_or_path = np.squeeze(image_or_path) 
        image = sitk.GetImageFromArray(image_or_path)
    else:
        raise ValueError("Input must be a file path or a SimpleITK.Image object.")

    # Convert to numpy array for visualization
    array = sitk.GetArrayFromImage(image)  # Shape: (slices, height, width)

    # Set up figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # Leave space for the slider
    slice_idx = 0
    im = ax.imshow(array[slice_idx], cmap='gray')
    ax.set_title(f"Slice {slice_idx+1}/{array.shape[0]}")
    ax.axis('off')

    # Add slider
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, array.shape[0]-1, valinit=0, valstep=1)

    # Update function
    def update(val):
        idx = int(slider.val)
        im.set_data(array[idx])
        ax.set_title(f"Slice {idx+1}/{array.shape[0]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def plot_slices_of_images_with_slider(*images):
    """
    Creates a fugure with as many subplots as images provided,
    with a single unified slider to navigate through the slices of all images.
    """
    # use this to visually troubleshoot the images
    # This function assumes that all images have the same number of slices in the z dimension, 
    # which should be the case 

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    images = [sitk.ReadImage(path) for path in images] if \
        isinstance(images[0], str) else images
    


    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # Ensure axes is iterable even for a single image
    if num_images == 1:
        axes = [axes]

    # Initialize the slider
    ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, images[0].GetSize()[2] - 1, valinit=0, valstep=1)

    def update(val):
        slice_index = int(slider.val)
        for ax, image in zip(axes, images):
            #print(f"Processing image with size: {image.GetSize()} and spacing: {image.GetSpacing()}")
            
            image_array = sitk.GetArrayViewFromImage(image)
            ax.clear()
            ax.imshow(image_array[slice_index], cmap='gray')
            ax.set_title(f'Slice {slice_index}')
            ax.axis('off')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    print()

    # Initial plot
    update(0)

    plt.show()


def show_random_from_json(json_file):
    """ open the preprocessing.json and load the images and labels in their pairs"""

    with open(json_file, "r") as f:
        dataset_json = json.load(f)

    rand = random.choices(list(dataset_json["imagesTr"].values()), k=20)
    # this id extraction is specific to the nnUNet format
    ids = [int(os.path.basename(path).split("_")[1]) for path in rand]

    for image, label_id in zip(rand, ids): 
        print(f"Visualizing image: {image}")
        plot_slices_of_images_with_slider(
            image, list(dataset_json["labelsTr"].values())[label_id])
        print()

def test_label_image_correspondence(zipped):
    '''Zipped is a zip of (label_paths, image_paths), in that order'''
    # pass some slice of the paths to the function
    # load them with SimpleITK and check that their dimsensions and spacing match
    # we do this to ensure that the labels and images are compatible and the preprocessing will not fail

    for label, image in zipped:

        if isinstance(label, str):
            label = sitk.ReadImage(label)
            image = sitk.ReadImage(image)

        # Check if the dimensions match
        if label.GetSize() != image.GetSize():
            raise ValueError(f"Size mismatch: {label.GetSize()} and {image.GetSize()}")

        # Check if the spacing matches
        if label.GetSpacing() != image.GetSpacing():
            raise ValueError(f"Spacing mismatch: {label.GetSpacing()} and {image.GetSpacing()}")




if __name__ == "__main__":

    # Step 1: Read the DICOM file
    p = r"/media/guest/PORT-DISK/Datasets/Prostate-Datasets/ProstateCancer/QIN-PROSTATE-Repeatability/PCAMPMRI-00001/06-20-1994-NA-PROBD Pelvis w-01133/1006.000000-T2 Weighted Axial Segmentations-7.172/1-1.dcm"
    p2 = r"/media/guest/PORT-DISK/Datasets/Prostate-Datasets/ProstateCancer/QIN-PROSTATE-Repeatability/PCAMPMRI-00001/06-20-1994-NA-PROBD Pelvis w-01133/2006.000000-T2 Weighted Axial Measurements-18.24/1-1.dcm"
    p3 = r"/media/guest/PORT-DISK/Datasets/Prostate-Datasets/ProstateCancer/QIN-PROSTATE-Repeatability/PCAMPMRI-00001/06-20-1994-NA-PROBD Pelvis w-01133/6.000000-T2 Weighted Axial-23000"

    a = p
    try:
        image = sitk.ReadImage(a)
    except Exception as e:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(a)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

    visualize_dicom_slider(image)