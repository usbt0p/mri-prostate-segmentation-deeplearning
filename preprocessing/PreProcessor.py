import numpy as np
import SimpleITK as sitk


def normalize_image(image: sitk.Image, method: str = "zscore") -> sitk.Image:
    """
    Normalize a SimpleITK image using Z-score or Min-Max normalization.

    Parameters:
        image (sitk.Image): Input image.
        method (str): Normalization method, 'zscore' or 'minmax'.

    Returns:
        sitk.Image: Normalized image.
    """
    array = sitk.GetArrayFromImage(image)  # shape: [slices, height, width]
    non_zero = array[array > 0]  # Avoid background

    if method == "zscore":
        mean = non_zero.mean()
        std = non_zero.std()
        norm_array = (array - mean) / std
    elif method == "minmax":
        min_val = non_zero.min()
        max_val = non_zero.max()
        norm_array = (array - min_val) / (max_val - min_val)
    else:
        raise ValueError("Normalization method must be 'zscore' or 'minmax'")

    norm_array[array == 0] = 0  # Optional: preserve background as 0

    norm_image = sitk.GetImageFromArray(norm_array)
    norm_image.CopyInformation(image)

    return norm_image


def create_automatic_mask(
    image: sitk.Image, threshold_method: str = "otsu"
) -> sitk.Image:
    """
    Create an automatic mask for the image.

    Note that this function may return bad results for images that do not
    present a clear distinction between foreground and background, like some
    of the t2w axial images in the PICAI dataset.
    """
    if threshold_method == "otsu":
        # Use Otsu thresholding
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        mask = otsu_filter.Execute(image)
    else:
        # Use percentile thresholding
        image_array = sitk.GetArrayFromImage(image)
        threshold = np.percentile(image_array[image_array > 0], 10)
        mask = sitk.BinaryThreshold(
            image,
            lowerThreshold=threshold,
            upperThreshold=image_array.max(),
            insideValue=1,
            outsideValue=0,
        )

    # TODO malke this function work
    # # Morphological operations to clean the mask
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
    mask = sitk.BinaryFillhole(mask)

    return sitk.Cast(mask, sitk.sitkUInt8)


def n4_bias_field_correction(
    image: sitk.Image,
    shrink_factor: int = 4,
    num_iterations: int = 50,
    num_fitting_levels: int = 4,
) -> sitk.Image:
    """
    Applies N4 bias field correction to the whole image without using a mask.

    Parameters:
        image: input image (Float32 or Float64).
        shrink_factor: factor to reduce resolution (default 4).
        num_iterations: iterations per fitting level.
        num_fitting_levels: levels in the multi-scale hierarchy.

    Returns:
        Image with bias field correction applied.
    """
    # Ensure float type
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Create mask if not supplied
    # if mask is not None:
    # TODO see if this is necessary
    # mask = create_automatic_mask(image, threshold_method='otsu')

    # Reduce resolution to speed up computation
    if shrink_factor > 1:
        small = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    else:
        small = image

    # Configure N4 filter
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations] * num_fitting_levels)

    # Run correction without mask: use the whole image
    corrected_small = corrector.Execute(small)

    # Reconstruct logarithmic bias field at original resolution
    log_bias = corrector.GetLogBiasFieldAsImage(image)

    # Apply full correction
    corrected = image / sitk.Exp(log_bias)

    # Copy spatial information
    corrected.CopyInformation(image)

    return corrected, log_bias


def ensure_3d(image: sitk.Image) -> sitk.Image:
    """
    Ensures the image is 3D. If it is 4D with a single volume in the fourth dimension, extracts it.
    If it has more than one volume, raises an error.
    This is necessary since SimpleITK likes to pull weird shenanigans with 4D images upon
    loading, even if they are actually 3D.

    Parameters:
        image (sitk.Image): Input image (3D or 4D).

    Returns:
        sitk.Image: 3D image.
    """
    dim = image.GetDimension()

    if dim == 3:
        return image

    elif dim == 4:
        size = list(image.GetSize())
        if size[3] != 1:
            raise ValueError(
                f"The image has {size[3]} volumes in the 4th dimension. Expected only one."
            )

        # Extract the 3D volume (index 0 in the 4th dimension)
        extract_size = size[:3] + [0]  # remove the 4th dimension
        extract_index = [0, 0, 0, 0]
        extracted = sitk.Extract(image, size=extract_size, index=extract_index)
        return extracted

    else:
        raise ValueError(f"Unsupported image dimension: {dim}")


def load_image(image_path: str) -> sitk.Image:
    """
    Reads a 3D image from a file path and does appropriate processing.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        sitk.Image: The read 3D image.
    """
    image = sitk.ReadImage(image_path)
    return ensure_3d(image)


def get_region_of_interest(image: sitk.Image, crop: float) -> sitk.Image:
    """
    Extract a region of interest (ROI) by cropping a percentage of the image size
    around the center.

    Parameters:
        image (sitk.Image): Input image.
        crop (float): Fraction of the image size to retain (0 < crop <= 1.0).

    Returns:
        sitk.Image: Cropped ROI image.
    """
    # inspired on: https://huggingface.co/MONAI/prostate_mri_anatomy/blob/0.3.5/scripts/center_crop.py
    if not (0 < crop <= 1.0):
        raise ValueError("Crop must be a value between 0 and 1.")

    # original size and center of the image
    original_size = image.GetSize()  # (x, y, z)
    center = [int(dim / 2) for dim in original_size]

    new_size = [int(dim * crop) for dim in original_size]

    # Compute the start index for cropping
    start = [max(0, c - ns // 2) for c, ns in zip(center, new_size)]
    roi = sitk.RegionOfInterest(image, size=new_size, index=start)

    return roi


def resample_image(
    image: sitk.Image,
    out_spacing: tuple = (
        0.5,
        0.5,
        5.0,
    ),  # TODO choose the best based on data exploration
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample the image to a given isotropic voxel size.

    Parameters:
        image (sitk.Image): Input image.
        out_spacing (tuple): Desired output spacing (z, y, x) in mm.
        interpolator: SimpleITK interpolator (default: sitkLinear).
        default_value (float): Value for areas outside the original image.

    Returns:
        sitk.Image: Resampled image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    out_spacing = tuple(float(s) for s in out_spacing)

    # Compute new size to preserve physical dimensions
    out_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, out_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetOutputPixelType(image.GetPixelID())

    resampled = resampler.Execute(image)
    return resampled


def register_images():
    # TODO register the images to a common space, e.g., using mutual information
    ...


def describe_image(img: sitk.Image):
    """
    Print basic information about the image, such as size, spacing, origin and direction.
    """

    print("Size (voxels):", img.GetSize())
    print("Spacing (mm):", tuple(round(s, 3) for s in img.GetSpacing()))
    print("Origin:", tuple(round(o, 3) for o in img.GetOrigin()))
    print("Direction:", tuple(round(d, 3) for d in img.GetDirection()))


# Example usage
if __name__ == "__main__":

    # TODO get ROI: CURRENT METHOD DOESNT WORK ON DATA THAT DOES NOT HAVE A MASK
    #
    # TODO figure out if different prepsocessing steps are order - invariant,
    # or if they should be applied in a specific order.
    #
    # for example, if we apply n4 bias field correction, should we apply it before or after
    # extracting the region of interest? after = less information to do the correction, but
    # faster, before = more information, but slower.
    # should alignignement / registration be done before or after the region of interest extraction?

    # TODO how to check proper functioning of the methods? analyze the intensity distribution?

    # TODO for n4 and normalize, some images might need background masking, but it
    # seems top work like shit. any fixes?

    # TODO determine the upscaling / downscaling process:
    # order matters here, when to do it?

    # TODO in resampling: must conserve the quality, i.e. not resample to a bigger
    # voxel size or details will be lost!!
    # do data exploration to find the voxel size counts and usee it to inform the new size

    # TODO other stuff:
    # 1. download the remaining datasets (script)
    # 2. make some analysis of them
    # 3. make a pipeline system / figure out if there is one that exists and lets you
    # use custom functions

    import os
    from DataAnalyzer import DataAnalyzer

    # we will pick several random images, apply preprocessing and save them in the ./imgs dir,
    # each with its corresponding index

    data_analyzer = DataAnalyzer("/home/guest/work/Datasets")
    data_analyzer.regex = ".*_t2w.mha"

    paths = {
        "picai_labels_wg": "picai_labels_all/picai_labels-main/anatomical_delineations/whole_gland/AI/Guerbet23",
        "picai_labels_zonal": "picai_labels_all/picai_labels-main/anatomical_delineations/zonal_pz_tz/AI/Yuan23",
        "picai_folds": "picai_folds/",
    }

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

            img_path = data_analyzer.pick_random(folder, 1, type="file")[
                0
            ]  # unpack because it returns a list
            print(f"Processing image: {img_path}")

            img = sitk.ReadImage(img_path)
            img = ensure_3d(img)

            # example usage of n4_correction:
            img_out, log_bias = n4_bias_field_correction(img)

            # example usage of automatic mask:
            # img_out = create_automatic_mask(img)

            # TODO this method seems to return garbage, review. Check in the papers if they use it
            # TODO: check if the method is necessary, test n4 without
            # if it is, incorporate into n4, if not remove it
            # then test n4, and then move on to registration, roi and resampling.
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

            img_path = data_analyzer.pick_random(folder, 1, type="file")[
                0
            ]  # unpack because it returns a list
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

    def test_automatic_mask():  # TODO work in progress, make useful
        # select a specific file that contains an obvious background
        img_path = next(
            data_analyzer.get_files(
                "picai_folds/picai_images_fold0/11392", regex=".*_t2w.mha"  # 10947,
            )
        )

        img_mask = create_automatic_mask(
            load_image(img_path), threshold_method="percentile"
        )

        # Save the mask
        mask_path = "./imgs/automask.nii.gz"
        sitk.WriteImage(img_mask, mask_path)

        data_analyzer.data_root = "."
        data_analyzer.show_image(img_path, mask_path, save="./imgs/automask.png")

    def test_resample_images(verbose=True):
        # use custom funtion to pick random directories
        rfolders = data_analyzer.pick_random(
            "picai_folds/picai_images_fold0/", 3, type="dir"
        )

        for i, folder in enumerate(rfolders, start=1):

            img_path = data_analyzer.pick_random(folder, 1, type="file")[0]
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

            img_path = data_analyzer.pick_random(folder, 1, type="file")[0]
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

        def describe_image(img):
            print("Size (voxels):", img.GetSize())
            print("Spacing (mm):", img.GetSpacing())
            print("Origin:", img.GetOrigin())
            print("Direction:", img.GetDirection())

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

    # Uncomment the function you want to test
    roi_test()
    # n4_test()
    # test_automatic_mask()
    # test_resample_mask()

    # clean all .nii.gz files in the imgs folder after finished
    for file in os.listdir("./imgs"):
        if file.endswith(".nii.gz"):
            os.remove(os.path.join("./imgs", file))
