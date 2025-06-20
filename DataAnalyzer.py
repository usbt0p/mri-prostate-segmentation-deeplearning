import os
import sys
from re import compile
import concurrent.futures

#import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import SimpleITK as sitk


class DataAnalyzer(object):
    """A class for analyzing medical imaging data.
    This class provides methods to read and analyze image files, extract metadata,
    and visualize images. It is designed to work with directories containing medical
    imaging data, such as MRI scans, and can handle both 2D and 3D images.

    Beware this heavily relies on setting absolte paths in each function to work properly.
    Every function except inner helpers (_functio_name) expects a path relative to the data root directory.
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

    def show_image(self, *image_paths, save : str =None):
        """
        Opens and displays an image.
        Only images supported by SimpleITK.ReadImage are supported.
        By default, 3D images show the middle slice.

        Parameters:
        -----------
        image_path : str
            Path to the image file.
        """
        try:
            # Create a figure with subplots for each image
            num_images = len(image_paths)
            fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
            
            # Ensure axes is iterable even for a single image
            if num_images == 1:
                axes = [axes]
            
            for ax, image_path in zip(axes, image_paths):
                image_path = self.abspath(image_path)
                
                # Load the image using SimpleITK
                image = sitk.ReadImage(image_path)
                
                # Convert the image to a numpy array for visualization
                image_array = sitk.GetArrayViewFromImage(image)
                
                # Take a slice of the image (e.g., the middle slice)
                if image_array.ndim == 3:
                    # TODO add support for choosing the slice
                    # For now, we just take the middle slice
                    image_array = image_array[image_array.shape[0] // 2, :, :]
                elif image_array.ndim == 2:
                    image_array = image_array[:, :]
                
                # Display the image on the corresponding axis
                ax.imshow(image_array, cmap="gray")
                ax.set_title(os.path.basename(image_path))
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
        """Checks if a mask file is empty (contains only zero values).
    
        Parameters:
        -----------
        path : str
            Path to the mask file.
    
        Returns:
        --------
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
        """Uses SimpleITK to count non-empty masks in a folder.
        Assumes the files inside the dir are actually masks.
    
        Returns:
        --------
        tuple: (non_empty_count, empty_list)
        - non_empty_count: Number of non-empty mask files.
        - empty_list: List of empty mask filenames.
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
    
    def image_intensity_histogram(self, image_path, bins=128, plot=False, save=None):
        """
        Computes the histogram of pixel intensities for a given image.
        
        Parameters:
        -----------
        image_path : str
            Path to the image file.
        bins : int, optional
            Number of bins for the histogram. Default is 256.
        
        Returns:
        --------
        tuple: (histogram, bin_edges)
            histogram: The computed histogram of pixel intensities.
            bin_edges: The edges of the bins used in the histogram.
        """
        image_path = self.abspath(image_path)
        image = sitk.ReadImage(self.abspath(image_path))
        array = sitk.GetArrayFromImage(image).flatten()
        hist, bin_edges = np.histogram(array, bins=bins, range=(array.min(), array.max()))
        
        if plot:
            plt.figure(figsize=(6,4))
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.title('Histogram of Pixel Intensities')
            plt.grid()
            if save is not None:
                if not save.endswith(".png"):
                    save += ".png"
                plt.savefig(save, bbox_inches="tight", pad_inches=0.1)
            plt.show()

        return hist, bin_edges

    def pick_random(self, path,  num : int, type="file"):
        
        # if the folder has folders, pick a random one, and if not, then pick a random file
        path = self.abspath(path)
        if type == "file":
            items = list(self.get_files(path, self.regex))
        elif type == "dir":
            items = list(self.get_dirs(path))
        else:
            raise ValueError("type must be 'file' or 'dir'")
        
        choices = np.random.choice(items, num, replace=False)

        return [os.path.join(path, c) for c in choices]
        
    

if __name__ == "__main__":
    from time import perf_counter
    from os.path import join

    # create an analyzer object with the root path to the dataset and
    analyzer = DataAnalyzer("/home/guest/work/Datasets")

    paths = {
        "picai_labels_wg" : "picai_labels_all/picai_labels-main/anatomical_delineations/whole_gland/AI/Guerbet23",
        "picai_labels_zonal" : "picai_labels_all/picai_labels-main/anatomical_delineations/zonal_pz_tz/AI/Yuan23",
        "picai_folds" : "picai_folds/"
    }

    # use this regex to filter the files
    analyzer.regex = "t2.nii.gz" #(.*_t2w.mha$)|(.*_sag.mha$)|(.*_cor.mha$)"
    # res = analyzer.collect_metadata_to_dataframe("prostate158/prostate158_train/train/110") #"picai_folds/picai_images_fold0/10189")
    # print(res)

    analyzer.image_intensity_histogram("prostate158/prostate158_train/train/111/t2.nii.gz", plot=True)

    # start = perf_counter()
    # df = analyzer.collect_metadata_from_subdirs("picai_folds/picai_images_fold0")
    # print(df)
    # print(perf_counter() - start, "seconds")

    # import random

    # # pick a random folder and set the name of a file inside as the prefix
    # d = "picai_folds/picai_images_fold0"
    # dirs = list(analyzer.get_dirs(d))
    # random_dir = random.choice(dirs)
    # files_in_dir = analyzer.get_files(join(d, random_dir), ".*_t2w.mha$")

    # # we have to do this because get files is a generator
    # name = list(files_in_dir)[0] 
    # i1 = join(d, random_dir, name)
    # # get the corresponding nii.gz file that masks the image
    # nii = name.split("_t2w")[0] + ".nii.gz"
    # i2 = join(paths['picai_labels_zonal'], nii)

    # analyzer.show_image(i1, i2, save="./test.png")
