import os
import sys
from re import compile
import concurrent.futures

import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk


class DataAnalyzer(object):
    """A class for analyzing medical imaging data.
    This class provides methods to read and analyze image files, extract metadata,
    and visualize images. It is designed to work with directories containing medical
    imaging data, such as MRI scans, and can handle both 2D and 3D images.
    """


    def __init__(self, data_root):
        """
        Initializes the DataAnalyzer with a root directory for data.

        Parameters:
        -----------
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
        return os.path.join(self.data_root, path)

    def get_dirs(self, path):
        """Generator that yields directory names in the specified path.
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
        Generator that yields files in the specified directory that match the given regex pattern.
        Parameters:
        -----------
        path : str
            The directory path to search for files.
        regex : str, optional
            A regex pattern to filter the files. If None, all files are yielded.
        Yields:
        -------
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
                if regex:
                    if regex.match(f):
                        yield f
                else:
                    yield f

    def show_image(self, image_path, save=False):
        """
        Opens and displays an image.
        Only images supported by SimpleITK.ReadImage are supported.
        By default, 3D images show the middle slice.

        Parameters:
        -----------
        image_path : str
            Path to the image file.
        """
        image_path = self.abspath(image_path)
        try:
            # Load the image using SimpleITK
            image = sitk.ReadImage(image_path)

            # Convert the image to a numpy array for visualization
            image_array = sitk.GetArrayViewFromImage(image)

            # Take a slice of the image (e.g., the middle slice)
            if image_array.ndim == 3:
                image_array = image_array[image_array.shape[0] // 2, :, :]
            elif image_array.ndim == 2:
                image_array = image_array[:, :]

            # Display the image using matplotlib
            plt.imshow(image_array, cmap="gray")
            plt.title(os.path.basename(image_path))
            plt.axis("off")
            if save:
                name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
                plt.savefig(name, bbox_inches="tight", pad_inches=0.1)
            plt.show()
        except Exception as e:
            print(f"Error opening the image: {e}")

    def count_and_find_non_empty_masks(self, folder):
        """Uses SimpleITK to count non-empty masks in a folder.
        Assumes the files inside the dir are actually masks.

        Returns:
        --------
        tuple: (non_empty_count, total_count, non_empty_list)
        - non_empty_count: Number of non-empty mask files.
        - total_count: Total number of mask files.
        - non_empty_list: List of non-empty mask filenames.
        """
        folder = self.abspath(folder)
        mask_files = self.get_files(folder)
        non_empty_count = 0
        empty_list = []
        for f in mask_files:
            path = os.path.join(folder, f)
            try:
                mask = sitk.ReadImage(path)
                # transform it into an array and check if it has any non-zero values
                arr = sitk.GetArrayViewFromImage(mask)
                if arr.max() > 0:
                    non_empty_count += 1
                else:
                    empty_list.append(f)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        return non_empty_count, empty_list

    def _get_header_value(self, filepath, key):
        """
        Reads the header of an image file and returns the value for the given key.
        Used as a fallback if the metadata is not available trough SimpleITK.

        Implemented to read only the header, the class will break out of the line-reading
        loop when it encounters a line that cannot be decoded as UTF-8.
        """

        with open(filepath, "rb") as f:
            for line in f:
                try:
                    line = line.decode("utf-8").strip()

                    if line.startswith(key):
                        # e.g. AnatomicalOrientation = ASL
                        return line.split("=", 1)[-1].strip()
                except Exception:
                    print(f"Error decoding line in {filepath}: {line}")
                    break  # we break when the data is not utf-8 encoded (hex or binary stuff in mri imaging)
        return None

    def parse_metadata_file(self, filepath):
        """
        Parses an image file and extracts required fields using SimpleITK.
        """
        abspath = self.abspath(filepath)
        info = {
            "filename": os.path.basename(filepath),
            "orientation": None,
            "dim_size": None,
            "prostate_volume": None,
            "vendor": None,
            "mri_name": None,
            "psa_report": None,
        }
        try:
            image = sitk.ReadImage(abspath)
            # DimSize from image size
            info["dim_size"] = image.GetSize()
            # Try to get metadata fields if present
            keys = image.GetMetaDataKeys()

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
        Collects metadata from all files in a folder into a pandas DataFrame.
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

    def _file_paths_gen(self, parent_dir):
        """Generator that yields file paths of all files in subdirectories"""
        for subdir in self.get_dirs(parent_dir):
            subdir_path = os.path.join(parent_dir, subdir)
            meta_files = self.get_files(subdir_path, self.regex)
            for f in meta_files:
                yield os.path.join(subdir_path, f)

    def collect_metadata_from_subdirs(self, parent_dir, max_workers=None):
        """
        Collects metadata from all files in all subdirectories into a
        pandas DataFrame using parallel processing. Uses the maximu
        number of CPUs available or a specified number of workers unless specified otherwise.
        """
        parent_dir = self.abspath(parent_dir)
        self.cpus if max_workers is None else max_workers

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            records = list(
                executor.map(self.parse_metadata_file, 
                self._file_paths_gen(parent_dir))
            )
        return pd.DataFrame(records)


if __name__ == "__main__":
    from time import perf_counter

    # create an analyzer object with the root path to the dataset and
    analyzer = DataAnalyzer("/home/guest/work/Datasets")

    # use this regex to filter the files
    analyzer.regex = "(.*_t2w.mha$)|(.*_sag.mha$)|(.*_cor.mha$)"
    res = analyzer.collect_metadata_to_dataframe("picai_folds/picai_images_fold0/10189")
    print(res)

    start = perf_counter()
    df = analyzer.collect_metadata_from_subdirs("picai_folds/picai_images_fold0")
    print(df)
    print(perf_counter() - start, "seconds")
