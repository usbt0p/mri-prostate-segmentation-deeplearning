from exploratoryAnalysis.DataAnalyzer import DataAnalyzer
from preprocessing.PreProcessor import *
from preprocessing.Pipeline import Pipeline
from Utils import *

from time import perf_counter
import json
import os

# use this path to set the abspath from which to read the data
RAW_DATA_ROOT = "/media/guest/PORT-DISK/Datasets/Prostate-Datasets/decathlon/Task05_Prostate"

# these paths are used to save the preprocessed data following nnUNet's conventions
OUT_ROOT = "/media/guest/PORT-DISK/Datasets/nnUNet_raw/"
assert os.path.exists(OUT_ROOT), f"Output root {OUT_ROOT} does not exist."

DATASET_NAME = "Dataset003_decathlon"

# Format for images: {CASE_IDENTIFIER(includes an id)}_{XXXX}.{FILE_ENDING}
# Format for labels: {CASE_IDENTIFIER(includes an id)}.{FILE_ENDING}
PREFIX_NAME = "deca_"  # ... + case_id + ...
FILE_ENDING = ".nii.gz"

# this path points to the root of the specific dataset train and test (or to only one if they are mixed)
paths = {
    "imagesTr" : os.path.join(RAW_DATA_ROOT, "imagesTr"),
    "imagesTs" : os.path.join(RAW_DATA_ROOT, "imagesTs"),
    "labelsTr" : os.path.join(RAW_DATA_ROOT, "labelsTr")
    # there are no labels for the test set
}

output_paths = {
    "images": os.path.join(OUT_ROOT, DATASET_NAME, "imagesTr"),
    "labels": os.path.join(OUT_ROOT, DATASET_NAME, "labelsTr"),
}

# these are some of the preprocessing parameters one might want to change
spacing = (0.5, 0.5, 3.0)  # Resampling factor for x, y, z dimensions
crop_factor = 0.55  # according to data analysis 
# if True, swaps the values of the zonal masks (PZ=1, TZ=2) to (PZ=2, TZ=1) or vice versa
bool_swap_mask_values = False

########################  INPUT PART  ###########################

# Initialize DataAnalyzer
analyzer = DataAnalyzer(RAW_DATA_ROOT)

# add regex to only get the valid
image_paths = list(analyzer.get_files(paths["imagesTr"], regex="^p"))

# TODO beware we must somehow make the prediction also using the WG labels!!!

label_paths = list(analyzer.get_files(paths["labelsTr"], regex="^p"))

assert len(image_paths) == len(label_paths), f"Number of images ({len(image_paths)}) does not match number of labels ({len(label_paths)})"

####################### OUTPUT part ########################

# Ensure output directories exist
os.makedirs(output_paths["images"], exist_ok=True)
os.makedirs(output_paths["labels"], exist_ok=True)

# use list comprehensions to create output filenames
out_images = [
    create_filename(
        output_paths["images"],
        idx,
        prefix=PREFIX_NAME,
        ending=FILE_ENDING,
        channel_id="_0000",
    )
    for idx in range(len(image_paths))
]

out_labels = [
    create_filename(
        output_paths["labels"],
        idx,
        prefix=PREFIX_NAME,
        ending=FILE_ENDING,
        channel_id="",
    )
    for idx in range(len(label_paths))
]

################  JSON part  ########################
# populate the json

# determine the path to save the JSON files
json_path = os.path.join(OUT_ROOT, DATASET_NAME)
os.makedirs(json_path, exist_ok=True)

# first, create the preprocessing JSON
images_json = {in_pth: out_pth for in_pth, out_pth in zip(image_paths, out_images)}
labels_json = {in_pth: out_pth for in_pth, out_pth in zip(label_paths, out_labels)}

preprocessing_json = {
    "imagesTr": images_json,
    "labelsTr": labels_json,
    "metadata": {
        "pipeline_images": None,
        "pipeline_labels": None,
        "preprocessing_time": 0,
    },
}

json_file = os.path.join(json_path, "preprocessing.json")
with open(json_file, "w") as f:
    json.dump(preprocessing_json, f, indent=4)

# Print confirmation
print(f"Preprocessing JSON saved to {json_file}")

# next, load the data template JSON and populate it
# this is nnUnet's expected metadata json
with open("./loadingData/data.template.json", "r") as f:
    data_json = json.load(f)
data_json["numTraining"] = len(images_json)
data_json["file_ending"] = FILE_ENDING

# Save the data JSON to a file
data_json_file = os.path.join(json_path, "dataset.json")
with open(data_json_file, "w") as f:
    json.dump(data_json, f, indent=4)
# Print confirmation
print(f"Data JSON saved to {data_json_file}")

del images_json, labels_json, data_json, preprocessing_json  # free some memory

################ NOW FOR THE ACTUAL PREPROCESSING ########################

# this can also be separated into a different script,
# loading the in and out paths from the preprocessing.json file

# Initialize pipelines with the desired preprocessing steps

pipeline_images = Pipeline()
pipeline_images.add(load_image
    ).add(reorient_image, target_direction="RPS"
    ).add(resample_image, interpolator=sitk.sitkLinear, out_spacing=spacing
    ).add(get_region_of_interest, crop=crop_factor
    ).add(n4_bias_field_correction)

pipeline_labels = Pipeline()
pipeline_labels.add(load_image
    ).add(reorient_image, target_direction="RPS"
    ).add(resample_image, interpolator=sitk.sitkNearestNeighbor, out_spacing=spacing
    ).add(get_region_of_interest, crop=crop_factor)

if bool_swap_mask_values:
    pipeline_labels.add(swap_zonal_mask_values)

# Preprocess images and labels using the pipeline
# zip images and labels to keep them associated
img_lbl_pairs = list(zip(image_paths, label_paths))
workers = os.cpu_count()  # Use all available CPU cores

# And, now, Process and save pairs in parallel
start_time = perf_counter()
# results are tuples of (image_path, label_path)
paired_results = preprocess_pairs_parallel(
    img_lbl_pairs, pipeline_images, pipeline_labels, workers=workers
)

# Ensure output directories exist
os.makedirs(output_paths["images"], exist_ok=True)
os.makedirs(output_paths["labels"], exist_ok=True)

# save the images and labels to disk, also in parallel
# same as before with the parallel pairs, keeping association
out_i, out_l = save_pairs_parallel(
    paired_results, out_images, out_labels, workers=workers
)

# count the saving as part of the preprocessing time
end_time = perf_counter()
print(f"Saved {len(out_i)} images and {len(out_l)} labels to disk.")

# Save metadata about the pipeline to preprocessing.json
with open(json_file, "r") as f:
    dataset_json = json.load(f)
    # add pipeline info to json
dataset_json["metadata"]["pipeline_images"] = str(pipeline_images)
dataset_json["metadata"]["pipeline_labels"] = str(pipeline_labels)
dataset_json["metadata"]["preprocessing_time"] = end_time - start_time

# Save the updated JSON to a file
with open(json_file, "w") as f:
    json.dump(dataset_json, f, indent=4)
print(f"Updated dataset JSON saved to {json_file}")

# visual check of some of the images and labels
show_random_from_json(json_file)
