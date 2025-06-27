"""
A module for testing various preprocessing functions on medical images individually.
"""

import SimpleITK as sitk
from preprocessing.PreProcessor import *


def roi_test():
    # use custom funtion to pick random directories
    rfolders = data_analyzer.pick_random(
        "picai_folds/picai_images_fold0/", 4, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        img_path = data_analyzer.pick_random(folder, 1, type="file")
        print(f"Processing image: {img_path}")

        img = load_image(img_path)
        describe_image(img)

        roi = get_region_of_interest(img, crop=0.6)
        describe_image(roi)

        da = DataAnalyzer(".")
        da.show_image(img_path, roi, save=f"./imgs/roi_test_{i}.png")


def n4_test():

    # use custom funtion to pick random directories
    rfolders = data_analyzer.pick_random(
        "picai_folds/picai_images_fold0/", 3, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        img_path = data_analyzer.pick_random(folder, 1, type="file")

        print(f"Processing image: {img_path}")
        img = load_image(img_path)

        # example usage of n4_correction:
        img_out, log_bias = n4_bias_field_correction(img, return_log_bias=True)

        da = DataAnalyzer(".")
        da.show_image(img, log_bias, img_out, save=f"./imgs/n4_test_{i}.png")

        # histograms help understand how the processing affects intensity distribution
        da.image_intensity_histogram(
            img_path, plot=True, save=f"./imgs/n4_histogram_original_{i}.png"
        )
        da.image_intensity_histogram(
            img_out, plot=True, save=f"./imgs/n4_histogram_processed_{i}.png"
        )


def normalization_test():

    # use custom funtion to pick random directories
    rfolders = data_analyzer.pick_random(
        "picai_folds/picai_images_fold0/", 3, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        img_path = data_analyzer.pick_random(folder, 1, type="file")
        print(f"Processing image: {img_path}")

        img = load_image(img_path)

        # Example usage of norm:
        img_out = normalize_image(img, method="minmax")

        da = DataAnalyzer(".")
        da.show_image(img_path, img_out, save=f"./imgs/norm_test_{i}.png")

        # histograms help understand how the processing affects intensity distribution
        da.image_intensity_histogram(
            img_path, plot=True, save=f"./imgs/norm_histogram_original_{i}.png"
        )
        da.image_intensity_histogram(
            img_out, plot=True, save=f"./imgs/norm_histogram_processed_{i}.png"
        )


def test_resample_images(verbose=True):
    # use custom funtion to pick random directories
    rfolders = data_analyzer.pick_random(
        "picai_folds/picai_images_fold0/", 3, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        img_path = data_analyzer.pick_random(folder, 1, type="file")
        print(f"Processing image: {img_path}")

        img = load_image(img_path)
        img_resampled = resample_image(img)

        da = DataAnalyzer(".")
        da.show_image(img_path, img_resampled, save=f"./imgs/resample_test_{i}.png")

        if verbose:
            resample_verbose_evaluation(img, img_resampled, i)


def test_resample_mask(verbose=True):
    # use custom funtion to pick random directories
    rfolders = data_analyzer.pick_random(
        "picai_folds/picai_images_fold0/", 3, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        img_path = data_analyzer.pick_random(folder, 1, type="file")
        print(f"Processing image: {img_path}")

        img = load_image(img_path)

        # getting the corresponding mask from the image path is cumbersome
        mask_path = next(
            data_analyzer.get_files(
                paths["picai_labels_zonal"], regex=os.path.basename(img_path)[0:13]
            )
        )

        print(f"Using mask: {mask_path}")
        mask = load_image(mask_path)

        # NOTE WE CHANGE THE INTERPOLATOR!! nearest neighbor works better for masks
        mask_resampled = resample_image(mask, interpolator=sitk.sitkNearestNeighbor)

        da = DataAnalyzer(".")
        da.show_image(
            img_path,
            mask_path,
            mask_resampled,
            save=f"./imgs/resample_mask_test_{i}.png",
        )

        if verbose:
            resample_verbose_evaluation(img, mask_resampled, i)


def resample_verbose_evaluation(img_original, img_resampled, i):
    # print info to evaluate the images

    def absolute_difference(original_img, resampled_img):
        # this compares the original and resampled images voxel by voxel
        # useful to check the effect of interpolation in the resampling process,
        # since both are in the same space but the original is not interpolated

        # Reconvertir ambas imágenes a float para restarlas
        original = sitk.Cast(original_img, sitk.sitkFloat32)
        resampled = sitk.Cast(resampled_img, sitk.sitkFloat32)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(resampled)
        resampler.SetInterpolator(sitk.sitkLinear)  # ASSUMEs linear interpolation
        resampler.SetTransform(sitk.Transform())
        original = resampler.Execute(original)

        # Para comparar voxel a voxel, deben estar alineadas (mismo tamaño y spacing)
        # Si no lo están, primero registra o reinterpola la original al espacio de la resampleada
        diff = sitk.Abs(original - resampled)
        diff_arr = sitk.GetArrayFromImage(diff)
        print("MAE (error medio absoluto):", diff_arr.mean())
        print("Max diff:", diff_arr.max())

    print(f"Original image {i}:")
    describe_image(img_original)
    print(f"Resampled image {i}:")
    describe_image(img_resampled)
    absolute_difference(img_original, img_resampled)
    print("-" * 40)

def test_combine_zonal_masks():
    print("Testing combine_zonal_masks function...", end="\n\n")

    data_analyzer.regex = ".*\.nii\.gz"
    nfiles = 3

    for i in range(nfiles):
        
        zonal_mask_path = data_analyzer.pick_random(paths["picai_labels_zonal"], 1, type="file")
        print(f"Processing image: {zonal_mask_path}")

        zonal_mask = load_image(zonal_mask_path)
        combined_mask = combine_zonal_masks(zonal_mask, 1, 2)

        da = DataAnalyzer(".")
        da.show_image(
            zonal_mask,
            combined_mask,
            save=f"./imgs/combined_mask_test_{i}.png",
        )

def test_swap_zonal_mask_values():
    """
    Test the swap_zonal_mask_values function.
    This function swaps the values of the peripheral and transition zones in a zonal mask.
    """

    print("Testing swap_zonal_mask_values function...", end="\n\n")

    # we try it in the prostate158 dataset since it has inverted labels to picai
    data_analyzer.regex = "t2_anatomy_reader"
    path = "prostate158/prostate158_train/train"

    rfolders = data_analyzer.pick_random(
        path, 3, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        zonal_mask_path = data_analyzer.pick_random(folder, 1, type="file")
        print(f"Processing image: {zonal_mask_path}")

        zonal_mask = load_image(zonal_mask_path)
        swapped_mask = swap_zonal_mask_values(zonal_mask, 1, 2)

        da = DataAnalyzer(".")
        da.show_image(
            zonal_mask_path,
            swapped_mask,
            save=f"./imgs/swapped_mask_test_{i}.png",
        )


if __name__ == "__main__":
    import os
    from exploratoryAnalysis.DataAnalyzer import DataAnalyzer

    # we will pick several random images, apply preprocessing and save them in the ./imgs dir,
    # each with its corresponding index

    data_analyzer = DataAnalyzer("/home/guest/work/Datasets")
    data_analyzer.regex = ".*_t2w.mha"

    paths = {
        "picai_labels_wg": "picai_labels_all/picai_labels-main/anatomical_delineations/whole_gland/AI/Guerbet23",
        "picai_labels_zonal": "picai_labels_all/picai_labels-main/anatomical_delineations/zonal_pz_tz/AI/Yuan23",
        "picai_folds": "picai_folds/",
    }

    # Uncomment the function you want to test
    #normalization_test()
    roi_test()
    # n4_test()
    #test_resample_mask()
    #test_combine_zonal_masks()
    #test_swap_zonal_mask_values()

    # clean all .nii.gz files in the imgs folder after finished
    for file in os.listdir("./imgs"):
        if file.endswith(".nii.gz"):
            os.remove(os.path.join("./imgs", file))
