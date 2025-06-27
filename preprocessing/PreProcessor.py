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

def n4_bias_field_correction(
    image: sitk.Image,
    shrink_factor: int = 4,
    num_iterations: int = 50,
    num_fitting_levels: int = 4,
    return_log_bias: bool = False,
) -> sitk.Image:
    """
    Applies N4 bias field correction to the whole image without using a mask.

    Parameters:
        image: input image (Float32 or Float64).
        shrink_factor: factor to reduce resolution (default 4).
        num_iterations: iterations per fitting level.
        num_fitting_levels: levels in the multi-scale hierarchy.
        return_log_bias: if True, returns the logarithmic bias field as well. For visualization or debugging.

    Returns:
        Image with bias field correction applied.
        If return_log_bias is True, also returns the logarithmic bias field.
    """
    # Ensure float type
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Background masking might be needed here if no previous cropping has been done

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

    if return_log_bias:
        return corrected, log_bias
    else:
        return corrected


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
    # in principle this is not needed for t2w images and their masks
    ...

def combine_zonal_masks(
    zonal_mask: sitk.Image,
    pz_value: int = 1,
    tz_value: int = 2,
    background_value: int = 0,
) -> sitk.Image:
    """
    Combine zonal mask values (pz and tz) into a whole gland mask.
    Sets both zone annotations to 1, and the background to 0.

    Parameters:
        zonal_mask (sitk.Image): Input zonal mask image with two zones.
        pz_value (int): Value for the peripheral zone in the output mask.
        tz_value (int): Value for the transition zone in the output mask.
        background_value (int): Value for the background in the output mask.

    Returns:
        sitk.Image: Combined whole gland mask.
    """
    # Ensure the input is a 3D image
    zonal_mask = ensure_3d(zonal_mask)

    # Create an output mask with the same size and spacing as the input
    whole_gland_mask = sitk.Image(zonal_mask.GetSize(), sitk.sitkUInt8)
    whole_gland_mask.CopyInformation(zonal_mask)

    # Set the pixel values based on the zonal mask
    whole_gland_mask[zonal_mask == pz_value] = 1  # Peripheral zone
    whole_gland_mask[zonal_mask == tz_value] = 1  # Transition zone
    whole_gland_mask[zonal_mask == 0] = background_value  # Background

    return whole_gland_mask
    

def describe_image(img: sitk.Image):
    """
    Print basic information about the image, such as size, spacing, origin and direction.
    """
    print("__" * 30)
    print("Size (voxels):", img.GetSize())
    print("Spacing (mm):", tuple(round(s, 3) for s in img.GetSpacing()))
    print("Origin:", tuple(round(o, 3) for o in img.GetOrigin()))
    print("Direction:", tuple(round(d, 3) for d in img.GetDirection()))
    print("__" * 30, end="\n\n")

    return img # for pipeline compatibility


# Example usage
if __name__ == "__main__":
    ...
