from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple, Any
from os import cpu_count
from tqdm import tqdm

class Pipeline:
    """
    A simple pipeline class to chain image processing functions.
    This allows for a sequence of operations to be applied to an image in a
    structured manner.
    """

    def __init__(self) -> None:
        """
        Initializes an empty pipeline with no processing steps.
        """
        self.steps = []
        self.max_workers = cpu_count()
        self.show_progress = False

    def add(self, func: Callable, *args: Any, **kwargs: Any) -> "Pipeline":
        """
        Adds a processing function to the pipeline.

        Args:
            func (Callable): The function to be added to the pipeline.
            *args (Any): Positional arguments to pass to the function.
            **kwargs (Any): Keyword arguments to pass to the function.

        Returns:
            Pipeline: The pipeline instance, allowing for method chaining.
        """
        self.steps.append((func, args, kwargs))
        return self

    def _process_single(self, image: Any) -> Any:
        """
        Processes a single image through the pipeline.

        Args:
            image (Any): The input image to process.

        Returns:
            Any: The processed image after applying all pipeline steps.
        """
        for func, args, kwargs in self.steps:
            image = func(image, *args, **kwargs)
        return image

    def run(self, images: Any, parallel: bool = False, max_workers: int = 4) -> Any:
        """
        Executes the pipeline on the given image(s).

        Args:
            images (Any): The input image or list of images to process.
            parallel (bool): Whether to process the images in parallel.
            max_workers (int): The maximum number of workers for parallel processing.

        Returns:
            Any: The processed image(s) after applying all pipeline steps.
        """
        if isinstance(images, list):
            if parallel:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    if self.show_progress:
                        results = list(tqdm(executor.map(self._process_single, images), 
                                            total=len(images), desc="Processing"))
                    else:
                        results = list(executor.map(self._process_single, images))
                return results
            else:
                if self.show_progress:
                    images = tqdm(images, desc="Processing", total=len(images))
                return [self._process_single(image) for image in images]
        else:
            return self._process_single(images)

    def __call__(self, images: Any, parallel: bool = True) -> Any:
        """
        Allows the pipeline to be called like a function.

        Args:
            images (Any): The input image or list of images to process.
            parallel (bool): Whether to process the images in parallel.
            max_workers (int): The maximum number of workers for parallel processing.

        Returns:
            Any: The processed image(s) after applying all pipeline steps.
        """
        return self.run(images, parallel=parallel, max_workers=self.max_workers)

    def __repr__(self) -> str:
        """
        Returns a string representation of the pipeline steps.

        Returns:
            str: A string listing all processing steps in the pipeline.
        """
        return "Pipeline with steps:\n" + "\n".join(
            f"{i+1}. {func.__name__}({', '.join(map(str, args))}, {', '.join(f'{k}={v}' for k, v in kwargs.items())})"
            for i, (func, args, kwargs) in enumerate(self.steps)
        )
    
if __name__ == "__main__":

    from preprocessing.PreProcessor import *
    from exploratoryAnalysis.DataAnalyzer import DataAnalyzer
    # TODO rename to preprocessors later

    analyzer = DataAnalyzer("/home/guest/work/Datasets")

    paths = {
        "picai_labels_wg": "picai_labels_all/picai_labels-main/anatomical_delineations/whole_gland/AI/Guerbet23",
        "picai_labels_zonal": "picai_labels_all/picai_labels-main/anatomical_delineations/zonal_pz_tz/AI/Yuan23",
        "picai_folds": "picai_folds/picai_images_fold0",
    }

    analyzer.regex = ".*_t2w.mha" # this is the regex used for picai
    rdir = analyzer.pick_random(paths["picai_folds"], 1, type="dir")
    img_pth = analyzer.pick_random(rdir, 1, type="file")

    # Example usage of the Pipeline class
    pipeline = Pipeline()

    describe_image(load_image(img_pth))  # Initial description of the image

    pipeline.add(load_image) \
            .add(get_region_of_interest, crop=0.6) \
            .add(resample_image) \
            .add(n4_bias_field_correction) \
            .add(describe_image)
            #.add(normalize_image, method="minmax") \ # TODO this can be done in the training step

    processed_image = pipeline(img_pth)

    analyzer.show_image(img_pth, processed_image, save="processed_image.png")