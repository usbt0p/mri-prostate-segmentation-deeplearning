"""
This script merges the multiple JSON files containing images and , created in the data loading
and preprocessing steps, into a single JSON file.
It also generates a holdout set from the merged data, which can be used for testing or validation purposes.
"""

import json
import random
import os

def merge_json_files(json_files, output_file):
    """
    Merge multiple JSON files containing images and labels into a single JSON file.
    Useful for having a complete data inventory, for later splitting or just analysis.

    Parameters:
    - json_files: list of str
        List of absolute paths to the JSON files to be merged.
    - output_file: str
        Path where the merged JSON file will be saved.

    """
    merged_data = {
        "images": [],
        "labels": [], 
        "sources": []  # Optional: to keep track of the dirs used
    }

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Extraer las imágenes y etiquetas preprocesadas
            merged_data["images"].extend(data["imagesTr"].values())
            merged_data["labels"].extend(data["labelsTr"].values())

    merged_data["sources"] = json_files  # Keep track of the sources
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

    return merged_data

def generate_holdout(data, test_ratio, out_dir, shuffle=True):
    """
    Generate a holdout set from the merged data.
    
    Parameters:
    - data: dict
        The merged JSON data containing images and labels.
    - test_ratio: float
        The proportion of the dataset to include in the holdout set.
    """
    total_images = len(data["images"])
    holdout_size = int(total_images * test_ratio)
    
    holdout_idxs = random.sample(range(total_images), holdout_size)

    test_holdout = {
        "images": [data["images"][i] for i in holdout_idxs],
        "labels": [data["labels"][i] for i in holdout_idxs],
    }

    train_data = {
        "images": [data["images"][i] for i in range(total_images) if i not in holdout_idxs],
        "labels": [data["labels"][i] for i in range(total_images) if i not in holdout_idxs],
    }

    if shuffle:
        pairs = list(zip(train_data["images"], train_data["labels"]))
        random.shuffle(pairs)
        train_data["images"], train_data["labels"] = zip(*pairs)
    
    # Save the holdout data
    holdout_file = os.path.join(out_dir, "test_holdout.json")
    with open(holdout_file, 'w') as f:
        json.dump(test_holdout, f, indent=4)
    print(f"Holdout data saved to {holdout_file}, size: {holdout_size} images")

    # Save the training data
    train_file = os.path.join(out_dir, "train_data.json")
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    print(f"Training data saved to {train_file}, size: {len(train_data['images'])} images")


if __name__ == "__main__":

    json_files = [
        "/media/guest/PORT-DISK/Datasets/nnUNet_raw/Dataset001_picai/preprocessing.json",
        "/media/guest/PORT-DISK/Datasets/nnUNet_raw/Dataset002_prostate158/preprocessing.json",
        "/media/guest/PORT-DISK/Datasets/nnUNet_raw/Dataset003_decathlon/preprocessing.json"
    ]

    # Archivo de salida
    output_file = "./data_jsons/all_data.json"

    inventory = merge_json_files(json_files, output_file)
    print(f"Joined JSON saved at: {output_file}")
    print(f"Total images: {len(inventory['images'])}, Total labels: {len(inventory['labels'])}")

    generate_holdout(inventory, test_ratio=0.15, out_dir="./data_jsons")