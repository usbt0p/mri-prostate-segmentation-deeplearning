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

    # Morphological operations to clean the mask
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


def get_region_of_interest():
    # TODO select a region centered on the whole prostate gland, slightly bigger than it
    ...


def resample_image():
    # TODO resample the image to a common voxel size, e.g., 1mm isotropic
    ...


def register_images():
    # TODO register the images to a common space, e.g., using mutual information
    ...


# Example usage
if __name__ == "__main__":
    from DataAnalyzer import DataAnalyzer

    # we will pick several random images, apply preprocessing and save them in the ./imgs dir,
    # each with its corresponding index

    data_analyzer = DataAnalyzer("/home/guest/work/Datasets")
    data_analyzer.regex = ".*_t2w.mha"

    # use custom funtion to pick random directories
    rfolders = data_analyzer.pick_random(
        "picai_folds/picai_images_fold0/", 3, type="dir"
    )

    for i, folder in enumerate(rfolders, start=1):

        img_path = data_analyzer.pick_random(
            folder, 1, type="file")[0]  # unpack because it returns a list
        print(f"Processing image: {img_path}")

        img = sitk.ReadImage(img_path)
        img = ensure_3d(img)
        img_array = sitk.GetArrayFromImage(img)

        # Apply N4 correction
        img_out, log_bias = n4_bias_field_correction(img)

        # Example usage of norm:
        # img_out = normalize_image(img, method='minmax')

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
