'''
Module to dump utilities for various tasks.
'''

from exploratoryAnalysis.DataAnalyzer import DataAnalyzer
import os
import SimpleITK as sitk
import sys

import matplotlib.pyplot as plt
from time import sleep

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