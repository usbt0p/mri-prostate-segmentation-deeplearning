import os
import sys
from re import compile
import concurrent.futures

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import SimpleITK as sitk


class DataAnalyzer(object):
    """
    A class for analyzing and manipulating directories containing medical
    imaging data, such as MRI scans, and handling both 2D and 3D images.
    These directories are usually structured in a way that each patient has a
    separate folder containing their imaging data, but not in a standard way.

    This class provides methods to:
    - Get and yield directories and files, supporting regex filtering.
    - Extract metadata from files, including dimensions, spacing, orientation, and vendor information.
    - Visualize SimpleITK images from a given path or an Image object, save them and choose slices.
    - Plot histograms of image intensities. 
    - Find empty masks in a directory and count non-empty masks.
    - Select random files or directories from a given path.

    .. warning::
        This heavily relies on setting absolute paths in each function to work properly.
        Every function except inner helpers (prefixed with an underscore) expects a path
        relative to the data root directory.
        This is useful most of the time, but sometimes you might want to use absolute paths directly or 
        wor around with os.path.join and other functions.
    """

    def __init__(self, data_root):
        """
        Initialize the DataAnalyzer with a root directory for data.

        Parameters
        ----------
        data_root : str
            The root directory where the data is stored.
        """
        self.data_root = data_root
        if not os.path.exists(data_root):
            print(f"Data root {data_root} does not exist.")
            sys.exit(1)

        self.regex: str = None
        self.cpus = os.cpu_count()

    def abspath(self, path):
        """
        Get the absolute path by joining the data root with the given path.

        Parameters
        ----------
        path : str
            Relative path from the data root.

        Returns
        -------
        str
            Absolute path.
        """
        try:
            return os.path.join(self.data_root, path)
        except TypeError as e:
            print(f"Error joining path {self.data_root} with {path}: {e}")
            sys.exit(1)

    def get_dirs(self, path):
        """
        Yield directory names in the specified path.

        Parameters
        ----------
        path : str
            Relative path from the data root.

        Yields
        ------
        str
            Directory names in the specified path.
        """
        path = self.abspath(path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            sys.exit(1)
        for d in os.listdir(path):
            if os.path.isdir(os.path.join(path, d)):
                yield d

    def get_files(self, path, regex=None):
        """
        Yield files in the specified directory that match the given regex pattern.

        Parameters
        ----------
        path : str
            The directory path to search for files.
        regex : str, optional
            A regex pattern to filter the files. If None, all files are yielded.

        Yields
        ------
        str
            Filename that matches the regex pattern in the specified directory.
        """
        path = self.abspath(path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            sys.exit(1)

        if regex:
            regex = compile(regex)

        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)):
                if regex:  # use the regex if provided, if not just yield all files
                    if regex.match(f):
                        yield os.path.join(path, f)
                else:
                    # TODO this doesnt use self.regex
                    yield os.path.join(path, f)

    def show_image(
        self, *image_or_path, save: str = None, slice: int = None, title: str = None
    ):
        """
        Open and display one or more images.

        Only images supported by SimpleITK.ReadImage are supported.
        By default, 3D images show the middle slice.

        Parameters
        ----------
        *image_or_path : str or SimpleITK.Image
            Paths to the image files (relative to data root) or SimpleITK.Image objects.
        save : str, optional
            If provided, saves the figure to this path (as PNG).
        slice : int, optional
            The slice index to display for 3D images. If None, the middle slice is shown.
        title : str, optional
            Title for the image(s). If multiple images are provided, this is ignored.
        """
        try:
            # Create a figure with subplots for each image
            num_images = len(image_or_path)
            fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

            # Ensure axes is iterable even for a single image
            if num_images == 1:
                axes = [axes]

            for ax, image_path in zip(axes, image_or_path):

                if not isinstance(image_path, sitk.Image):
                    image_path = self.abspath(image_path)

                    # Load the image using SimpleITK
                    image = sitk.ReadImage(image_path)
                    title = title if title else os.path.basename(image_path)
                else:
                    image = image_path
                    title = title if title else "Image"  # TODO this is shitty

                # Convert the image to a numpy array for visualization
                image_array = sitk.GetArrayViewFromImage(image)

                
                if image_array.ndim == 3: # if the image is 3D
                    # check if slice is provided, if not, use the middle slice
                    if slice is not None:
                        image_array = image_array[slice, :, :]
                    else:
                        image_array = image_array[
                            image_array.shape[0] // 2, :, :
                        ]                
                elif image_array.ndim == 2: # in case the image is 2D
                    image_array = image_array[:, :]

                # Display the image on the corresponding axis
                ax.imshow(image_array, cmap="gray")
                ax.set_title(title)
                ax.axis("off")

            # Save the figure if requested
            if save is not None:
                if not save.endswith(".png"):
                    save += ".png"
                plt.savefig(save, bbox_inches="tight", pad_inches=0.1)

            plt.show()
        except Exception as e:
            print(f"Error opening the images: {e}")

    def is_empty_mask(self, path):
        """
        Check if a mask file is empty (contains only zero values).

        Parameters
        ----------
        path : str
            Path to the mask file (relative to data root).

        Returns
        -------
        bool
            True if the mask is empty, False otherwise.
        """
        path = self.abspath(path)
        try:
            mask = sitk.ReadImage(path)
            arr = sitk.GetArrayViewFromImage(mask)
            return arr.max() == 0
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return True  # Treat as empty if there's an error

    def count_and_find_non_empty_masks(self, folder):
        """
        Count non-empty masks in a folder using SimpleITK.

        Assumes the files inside the directory are actually masks.

        Parameters
        ----------
        folder : str
            Path to the folder containing mask files (relative to data root).

        Returns
        -------
        tuple
            (non_empty_count, empty_list)
            - non_empty_count : int
                Number of non-empty mask files.
            - empty_list : list of str
                List of empty mask filenames.
        """
        folder = self.abspath(folder)
        mask_files = self.get_files(folder, regex=self.regex)
        non_empty_count = 0
        empty_list = []
        for f in mask_files:
            path = os.path.join(folder, f)
            if self.is_empty_mask(path):
                empty_list.append(f)
            else:
                non_empty_count += 1
        return non_empty_count, empty_list

    def _get_header_value(self, filepath, key):
        """
        Read the header of an image file and return the value for the given key.

        Used as a fallback if the metadata is not available through SimpleITK.
        Only reads the header; breaks out of the line-reading loop when it encounters
        a line that cannot be decoded as UTF-8.

        Parameters
        ----------
        filepath : str
            Path to the image file.
        key : str
            Metadata key to search for.

        Returns
        -------
        str or None
            Value for the given key, or None if not found.
        """

        with open(filepath, "rb") as f:
            for line in f:
                try:
                    line = line.decode("utf-8").strip()

                    if line.startswith(key):
                        # e.g. AnatomicalOrientation = ASL
                        return line.split("=", 1)[-1].strip()
                except Exception:
                    print(f"Error decoding line in {filepath}: {line[:15]}...")
                    break  # we break when the data is not utf-8 encoded (hex or binary stuff in mri imaging)
        return None

    def parse_metadata_file(self, filepath):
        """
        Parse an image file and extract required fields using SimpleITK.

        Parameters
        ----------
        filepath : str
            Path to the image file (relative to data root).

        Returns
        -------
        dict
            Dictionary with extracted metadata fields.
        """
        abspath = self.abspath(filepath)
        info = {
            "filename": os.path.basename(filepath),
            "dim_size": None,
            "spacing": None,
            "orientation": None,
            "prostate_volume": None,
            "vendor": None,
            "mri_name": None,
            "psa_report": None,
        }
        try:
            image = sitk.ReadImage(abspath)
            # DimSize from image size
            info["dim_size"] = image.GetSize()
            # get spacing and round to 3 decimal places
            info["spacing"] = tuple(round(s, 3) for s in image.GetSpacing())
            # Try to get metadata fields if present
            keys = image.GetMetaDataKeys()

            # for key in keys:
            #     print(image.GetMetaData(key))

            if "PROSTATE_VOLUME_REPORT" in keys:
                value = image.GetMetaData("PROSTATE_VOLUME_REPORT")
                info["prostate_volume"] = (
                    float(value) if value.lower() != "nan" else None
                )
            if "0008|0070" in keys:
                info["vendor"] = image.GetMetaData("0008|0070")
            if "0008|1090" in keys:
                info["mri_name"] = image.GetMetaData("0008|1090")
            if "PSA_REPORT" in keys:
                value = image.GetMetaData("PSA_REPORT")
                info["psa_report"] = float(value) if value.lower() != "nan" else None

            # Try to get AnatomicalOrientation from header if not in keys
            if "AnatomicalOrientation" in keys:
                info["orientation"] = image.GetMetaData("AnatomicalOrientation")
            else:
                orientation = self._get_header_value(abspath, "AnatomicalOrientation")
                if orientation:
                    info["orientation"] = orientation
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return info

    def collect_metadata_to_dataframe(self, folder):
        """
        Collect metadata from all files in a folder into a pandas DataFrame.

        Parameters
        ----------
        folder : str
            Path to the folder containing files (relative to data root).

        Returns
        -------
        pandas.DataFrame
            DataFrame containing metadata for all files.
        """

        folder = self.abspath(folder)
        meta_files = self.get_files(folder, self.regex)
        records = []

        # loop over the files in the dir, construct path and get their metadata
        for f in meta_files:
            path = os.path.join(folder, f)
            record = self.parse_metadata_file(path)
            records.append(record)
        df = pd.DataFrame(records)
        return df

    def file_paths_gen(self, parent_dir):
        """
        Yield file paths of all files in subdirectories.
        Useful for iterating over patient directories to get a file.

        Parameters
        ----------
        parent_dir : str
            Path to the parent directory (relative to data root).

        Yields
        ------
        str
            File paths in subdirectories.
        """
        for subdir in self.get_dirs(parent_dir):
            subdir_path = os.path.join(parent_dir, subdir)
            meta_files = self.get_files(subdir_path, self.regex)
            for f in meta_files:
                yield os.path.join(subdir_path, f)

    def collect_metadata_from_subdirs(self, parent_dir, max_workers=None):
        """
        Collect metadata from all files in all subdirectories into a pandas DataFrame
        using parallel processing.

        Uses the maximum number of CPUs available or a specified number of workers.

        Parameters
        ----------
        parent_dir : str
            Path to the parent directory (relative to data root).
        max_workers : int, optional
            Number of worker processes to use.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing metadata for all files in subdirectories.
        """
        parent_dir = self.abspath(parent_dir)
        self.cpus if max_workers is None else max_workers

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            records = list(
                executor.map(self.parse_metadata_file, self.file_paths_gen(parent_dir))
            )
        return pd.DataFrame(records)

    def image_intensity_histogram(self, image_or_path, bins=128, plot=False, save=None):
        """
        Compute the histogram of pixel intensities for a given image.

        Parameters
        ----------
        image_or_path : str | SimpleITK.Image
            Path to the image file (relative to data root), or sitk.Image.
        bins : int, optional
            Number of bins for the histogram. Default is 128.
        plot : bool, optional
            If True, plot the histogram. Default is False.
        save : str, optional
            If provided, saves the plot to this path (as PNG).

        Returns
        -------
        tuple
            (histogram, bin_edges)
            - histogram : numpy.ndarray
                The computed histogram of pixel intensities.
            - bin_edges : numpy.ndarray
                The edges of the bins used in the histogram.
        """
        if isinstance(image_or_path, sitk.Image):
            image = image_or_path
        else:
            image_path = self.abspath(image_or_path)
            image = sitk.ReadImage(image_path)

        array = sitk.GetArrayFromImage(image).flatten()
        hist, bin_edges = np.histogram(
            array, bins=bins, range=(array.min(), array.max())
        )

        if plot:
            plt.figure(figsize=(6, 4))
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.title("Histogram of Pixel Intensities")
            plt.grid()
            if save is not None:
                if not save.endswith(".png"):
                    save += ".png"
                plt.savefig(save, bbox_inches="tight", pad_inches=0.1)
            plt.show()

        return hist, bin_edges

    def pick_random(self, path, num: int, type="file"):
        """
        Pick a random selection of files or directories from a given path and return their absolute paths (joined).

        Parameters
        ----------
        path : str
            Path to the folder (relative to data root).
        num : int
            Number of items to pick.
        type : {'file', 'dir'}, optional
            Whether to pick files or directories. Default is 'file'.

        Returns
        -------
        list of str
            List of absolute paths to the randomly selected items.
        """

        # if the folder has folders, pick a random one, and if not, then pick a random file
        path = self.abspath(path)
        if type == "file":
            items = list(self.get_files(path, self.regex))
        elif type == "dir":
            items = list(self.get_dirs(path))
        else:
            raise ValueError("type must be 'file' or 'dir'")

        choices = np.random.choice(items, num, replace=False)

        return (
            [os.path.join(path, c) for c in choices]
            if num > 1
            else os.path.join(path, choices[0])
        )
    

    def calculate_cube_bounds(self, mask_path):
        """
        Calculate the bounds of the smallest "cube" containing the mask, centered at the image center.

        "Cube" is not strictly a cube but a rectangular prism that is centered around the image center
        and extends equally in all directions in the x and y dimensions, while covering the full depth
        of the mask in the z dimension.

        With 'start_y', 'end_y', 'start_x', 'end_x', we calculate the 
        size of the bounding box as the maximum of the width and height, 
        and then calculate the proportion of that w.r.t. the image size, to 
        get a sense of how much we can crop the image.

        Parameters:
            mask (numpy.ndarray): Path to a 3D mask (z, y, x).

        Returns:
            tuple: (start_z, end_z, start_y, end_y, start_x, end_x, mask_path, bounding_box_size, proportion_of_image_size)
                representing the cube bounds and additional information about the bounding box size and its proportion of the image size.
        """
        mask_path = os.path.abspath(mask_path)  # ensure the path is absolute
        # read
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)

        center_z, center_y, center_x = np.array(mask.shape) // 2

        # Find non-zero regions in x and y dimensions
        non_zero_indices = np.argwhere(mask)
        min_y, max_y = non_zero_indices[:, 1].min(), non_zero_indices[:, 1].max()
        min_x, max_x = non_zero_indices[:, 2].min(), non_zero_indices[:, 2].max()

        # Calculate the farthest distance from the center in x and y directions
        max_distance = max(center_y - min_y, max_y - center_y, center_x - min_x, max_x - center_x)

        # Define the cube bounds
        start_y = max(center_y - max_distance, 0)
        end_y = min(center_y + max_distance + 1, mask.shape[1])
        start_x = max(center_x - max_distance, 0)
        end_x = min(center_x + max_distance + 1, mask.shape[2])

        # Include the full depth (z-axis)
        start_z = 0
        end_z = mask.shape[0]

        # we'll also calculate the bounding box size and its proportion of the image size
        # for later analysis
        width = end_x - start_x
        height = end_y - start_y
        bounding_box_size = max(width, height) # this shoulnt be needed since we are using a cube, but...
        
        image_size = max(mask.shape[1], mask.shape[2]  )
        proportion_of_image_size = bounding_box_size / image_size

        return (start_z, end_z, start_y, end_y, start_x, end_x, 
            mask_path, bounding_box_size, proportion_of_image_size)

    def overlay_bounding_box(self, mask, slice, start_y, end_y, start_x, end_x):
        """
        Overlay the mask as a bounding box in a slice.

        Parameters:
            mask (numpy.ndarray): The 3D mask.
            start_y (int): Start index for the y-axis.
            end_y (int): End index for the y-axis.
            start_x (int): Start index for the x-axis.
            end_x (int): End index for the x-axis.
        """
        # pick the slice from the mask
        slice_data = mask[slice, :, :]

        plt.figure(figsize=(6, 6))
        plt.imshow(slice_data, cmap='gray')

        # Draw the bounding box
        plt.plot([start_x, end_x, end_x, start_x, start_x],
                    [start_y, start_y, end_y, end_y, start_y],
                    color='red', linewidth=2)

        plt.title(f"Slice {slice} with Bounding Box")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.axis('on')
        plt.show()




if __name__ == "__main__":
    from time import perf_counter
    from os.path import join
    import random

    # create an analyzer object with the root path to the dataset and
    analyzer = DataAnalyzer("/home/guest/work/Datasets")

    paths = {
        "picai_labels_wg": "picai_labels_all/picai_labels-main/anatomical_delineations/whole_gland/AI/Guerbet23",
        "picai_labels_zonal": "picai_labels_all/picai_labels-main/anatomical_delineations/zonal_pz_tz/AI/Yuan23",
        "picai_folds": "picai_folds/",
    }

    # use this regex to filter the files
    analyzer.regex = "t2.nii.gz"  # (.*_t2w.mha$)|(.*_sag.mha$)|(.*_cor.mha$)"

    def test_histogram(analyzer):
        analyzer.image_intensity_histogram(
            "prostate158/prostate158_train/train/111/t2.nii.gz", plot=True
        )

    def test_show_image(analyzer, paths):

        d = "picai_folds/picai_images_fold0"
        dirs = list(analyzer.get_dirs(d))
        random_dir = random.choice(dirs)
        files_in_dir = analyzer.get_files(join(d, random_dir), ".*_t2w.mha$")

        # we have to do this because get_files is a generator
        name = list(files_in_dir)[0]
        i1 = join(d, random_dir, name)
        # get the corresponding nii.gz file that masks the image
        nii = name.split("_t2w")[0] + ".nii.gz"
        i2 = join(paths["picai_labels_zonal"], nii)

        analyzer.show_image(i1, i2, save="./test.png")

    def test_collect_metadata(analyzer):
        # Test collect_metadata_to_dataframe
        analyzer.regex = "t2.nii.gz"
        res = analyzer.collect_metadata_to_dataframe(
            "prostate158/prostate158_train/train/110"
        )
        print(res)

        # Test collect_metadata_from_subdirs
        start = perf_counter()
        analyzer.regex = "(.*_t2w.mha$)|(.*_sag.mha$)|(.*_cor.mha$)"
        df = analyzer.collect_metadata_from_subdirs(
            "picai_folds/picai_images_fold0", max_workers=analyzer.cpus
        )
        print(df)
        print(perf_counter() - start, "seconds")

    # Example usage:
    # test_histogram(analyzer)
    # test_show_image(analyzer, paths)
    test_collect_metadata(analyzer)
