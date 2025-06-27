from typing import Callable, List, Tuple, Any

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

    def run(self, image: Any) -> Any:
        """
        Executes the pipeline on the given image.

        Args:
            image (Any): The input image to process.

        Returns:
            Any: The processed image after applying all pipeline steps.
        """

        # TODO decide if this is going to work with paths or images, or both
        for func, args, kwargs in self.steps:
            image = func(image, *args, **kwargs)
        return image
    
    def __call__(self, image: Any) -> Any:
        """
        Allows the pipeline to be called like a function.

        Args:
            image (Any): The input image to process.

        Returns:
            Any: The processed image after applying all pipeline steps.
        """

        return self.run(image)
    
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
            #.add(normalize_image, method="minmax") \

    processed_image = pipeline(img_pth)

    analyzer.show_image(img_pth, processed_image, save="processed_image.png")