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

        output_path = f"./imgs/roi_image_{i}.nii.gz"
        sitk.WriteImage(roi, output_path)

        da = DataAnalyzer(".")
        da.show_image(img_path, output_path, save=f"./imgs/roi_test_{i}.png")


def n4_test():

    # use custom funtion to pick random directories
    rfolders = data_analyzer.pick_random(
        "picai_folds/picai_images_fold0/", 3, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        img_path = data_analyzer.pick_random(folder, 1, type="file")

        print(f"Processing image: {img_path}")

        img = sitk.ReadImage(img_path)
        img = ensure_3d(img)

        # example usage of n4_correction:
        img_out, log_bias = n4_bias_field_correction(img, return_log_bias=True)

        # study whether to interpolate contours of the masks?

        # Save the processed images
        output_path = f"./imgs/processed_image_{i}.nii.gz"
        sitk.WriteImage(img_out, output_path)
        logb_path = f"./imgs/log_bias_{i}.nii.gz"
        sitk.WriteImage(log_bias, logb_path)

        da = DataAnalyzer(".")
        da.show_image(img_path, logb_path, output_path, save=f"./imgs/test_{i}.png")

        # histograms help understand how the processing affects intensity distribution
        da.image_intensity_histogram(
            img_path, plot=True, save=f"./imgs/histogram_original_{i}.png"
        )
        da.image_intensity_histogram(
            output_path, plot=True, save=f"./imgs/histogram_processed_{i}.png"
        )


def normalization_test():

    # use custom funtion to pick random directories
    rfolders = data_analyzer.pick_random(
        "picai_folds/picai_images_fold0/", 3, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        img_path = data_analyzer.pick_random(folder, 1, type="file")
        print(f"Processing image: {img_path}")

        img = sitk.ReadImage(img_path)
        img = ensure_3d(img)
        img_array = sitk.GetArrayFromImage(img)

        # Example usage of norm:
        img_out = normalize_image(img, method="minmax")

        # Save the processed images
        output_path = f"./imgs/processed_image_{i}.nii.gz"
        sitk.WriteImage(img_out, output_path)

        da = DataAnalyzer(".")
        da.show_image(img_path, output_path, save=f"./imgs/test_{i}.png")

        # histograms help understand how the processing affects intensity distribution
        da.image_intensity_histogram(
            img_path, plot=True, save=f"./imgs/histogram_original_{i}.png"
        )
        da.image_intensity_histogram(
            output_path, plot=True, save=f"./imgs/histogram_processed_{i}.png"
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

        output_path = f"./imgs/resampled_image_{i}.nii.gz"
        sitk.WriteImage(img_resampled, output_path)

        da = DataAnalyzer(".")
        da.show_image(img_path, output_path, save=f"./imgs/resample_test_{i}.png")

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

        mask_path = list(
            data_analyzer.get_files(
                paths["picai_labels_zonal"], regex=os.path.basename(img_path)[0:13]
            )
        )[0]

        print(f"Using mask: {mask_path}")
        mask = load_image(mask_path)

        # NOTE WE CHANGE THE INTERPOLATOR!! nearest neighbor works better for masks
        mask_resampled = resample_image(mask, interpolator=sitk.sitkNearestNeighbor)

        output_path = f"./imgs/resampled_mask_{i}.nii.gz"
        sitk.WriteImage(mask_resampled, output_path)

        da = DataAnalyzer(".")
        da.show_image(
            img_path,
            mask_path,
            output_path,
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
        
        # prostate158/prostate158_train/train
        zonal_mask_path = data_analyzer.pick_random(paths["picai_labels_zonal"], 1, type="file")
        print(f"Processing image: {zonal_mask_path}")

        zonal_mask = load_image(zonal_mask_path)

        combined_mask = combine_zonal_masks(zonal_mask)

        output_path = f"./imgs/combined_mask_{i}.nii.gz"
        sitk.WriteImage(combined_mask, output_path)

        da = DataAnalyzer(".")
        da.show_image(
            zonal_mask_path,
            output_path,
            save=f"./imgs/combined_mask_test_{i}.png",
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
    # normalization_test()
    # roi_test()
    # n4_test()
    # test_resample_mask()
    test_combine_zonal_masks()

    # clean all .nii.gz files in the imgs folder after finished
    for file in os.listdir("./imgs"):
        if file.endswith(".nii.gz"):
            os.remove(os.path.join("./imgs", file))
