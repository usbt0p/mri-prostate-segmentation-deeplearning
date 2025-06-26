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

def create_automatic_mask(image: sitk.Image) -> sitk.Image:
    """
    Create a mask using Gaussian blur, Otsu's thresholding, morphological operations,
    and extracting the largest connected component.
    """
    # Apply Gaussian blur
    blurred_image = sitk.SmoothingRecursiveGaussian(image, sigma=1.0)

    # Otsu's thresholding
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    binary_image = otsu_filter.Execute(blurred_image)

    # Morphological closing
    closed_image = sitk.BinaryMorphologicalClosing(binary_image, [10, 10, 10])

    # Connected component analysis
    connected_components = sitk.ConnectedComponent(closed_image)
    labeled_image = sitk.RelabelComponent(connected_components, sortByObjectSize=True)

    # Extract the largest connected component
    largest_component = sitk.BinaryThreshold(
        labeled_image, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0
    )

    return sitk.Cast(largest_component, sitk.sitkUInt8)


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
        0.5,
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
    ...

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