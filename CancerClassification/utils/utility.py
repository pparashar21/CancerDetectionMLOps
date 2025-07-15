import os
import sys
import boto3
import yaml
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from patchify import patchify
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Union
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from functools import partial

from CancerClassification.constants import *
from CancerClassification.utils.logger import logging
from CancerClassification.utils.exception_handler import ExceptionHandler

@ensure_annotations
def read_yaml(path: str) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox dictionary.

    Args:
        path (str): Path to the YAML file.

    Returns:
        ConfigBox: Parsed configuration.
    """
    try:
        with open(path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file loaded successfully from: {path}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty or malformed")
    except Exception as e:
        raise e

@ensure_annotations
def get_class_names(bucket_name: str, root_key: str) -> list:
    """
    Retrieves class names from the given S3 bucket prefix.

    Args:
        bucket_name (str): Name of the S3 bucket.
        root_key (str): Root prefix path in the S3 bucket.

    Returns:
        List[str]: List of class folder names.
    """
    s3 = boto3.client("s3")
    class_names = []

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=root_key, Delimiter="/")

    if "CommonPrefixes" in response:
        for folder in response['CommonPrefixes']:
            top_level_folder = folder['Prefix'].split("/")[-2]
            if top_level_folder != "ALL":
                inner_prefix = folder["Prefix"]
                inner_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=inner_prefix, Delimiter="/")
                if "CommonPrefixes" in inner_response:
                    for subfolder in inner_response["CommonPrefixes"]:
                        class_name = subfolder["Prefix"].split("/")[-2]
                        class_names.append(class_name)

    return class_names

@ensure_annotations
def load_hyperparameters(path: str, s3_bucket: str, data_folder: str) -> dict:
    """
    Loads hyperparameters from YAML and appends computed fields.

    Args:
        path (str): Path to YAML file.
        s3_bucket (str): S3 bucket to fetch class names.
        data_folder (str): Folder inside S3 bucket for training data.

    Returns:
        Dict: Dictionary with hyperparameters.
    """
    with open(path, 'r') as f:
        hp = yaml.safe_load(f)

    hp["NUM_PATCHES"] = (hp["IMAGE_SIZE"] ** 2) // (hp["PATCH_SIZE"] ** 2)
    hp["FLAT_PATCHES_SHAPE"] = (
        hp["NUM_PATCHES"],
        hp["PATCH_SIZE"] * hp["PATCH_SIZE"] * hp["NUM_CHANNELS"]
    )
    hp["CLASS_NAMES"] = get_class_names(bucket_name=s3_bucket, root_key=data_folder)
    return hp

@ensure_annotations
def get_image_paths_from_s3(bucket_name: str, root_prefix: str, split: float = 0.09) -> tuple:
    """
    Fetches and splits S3 image paths into train, validation, and test sets.

    Args:
        bucket_name (str): S3 bucket name.
        root_prefix (str): Prefix for images.
        split (float): Split ratio for validation and test.

    Returns:
        Tuple[List[str], List[str], List[str]]: Train, validation, and test image paths.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    image_paths = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=root_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                parts = key[len(root_prefix):].split("/")
                if len(parts) >= 3 and parts[0] != "ALL":
                    image_paths.append(f"s3://{bucket_name}/{key}")

    split_count = int(len(image_paths) * split)
    train, valid = train_test_split(image_paths, test_size=split_count, random_state=42)
    train, test = train_test_split(train, test_size=split_count, random_state=42)
    return train, valid, test

@ensure_annotations
def process_image_label(s3_path: Union[str, bytes], hp: dict) -> tuple:
    """
    Processes a single image from S3 and returns patches and class index.

    Args:
        s3_path (str): S3 path of the image.
        hp (dict): Hyperparameters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Image patches and class index.
    """
    s3_path = s3_path.decode() if isinstance(s3_path, bytes) else s3_path
    assert s3_path.startswith("s3://"), "Path must be an S3 URI."

    s3 = boto3.client("s3")
    _, _, bucket, *key_parts = s3_path.split("/")
    key = "/".join(key_parts)

    response = s3.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (hp["IMAGE_SIZE"], hp["IMAGE_SIZE"])) / 255.0
    patch_shape = (hp["PATCH_SIZE"], hp["PATCH_SIZE"], hp["NUM_CHANNELS"])
    patches = patchify(image, patch_shape, step=hp["PATCH_SIZE"])
    patches = np.reshape(patches, hp["FLAT_PATCHES_SHAPE"]).astype(np.float32)

    class_name = key_parts[-2]
    class_idx = np.array(hp["CLASS_NAMES"].index(class_name), dtype=np.int32)
    return patches, class_idx

def parse_fn_factory(hp: dict):
    """
    Creates a parse function for TF datasets using provided hyperparameters.

    Args:
        hp (dict): Hyperparameters used in image processing.

    Returns:
        Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
    """
    wrapped_func = partial(process_image_label, hp=hp)

    def parse(path: tf.Tensor) -> tuple:
        patches, label = tf.numpy_function(
            func=wrapped_func,
            inp=[path],
            Tout=[tf.float32, tf.int32]
        )
        label = tf.one_hot(label, hp["NUM_CLASSES"])
        patches.set_shape(hp["FLAT_PATCHES_SHAPE"])
        label.set_shape([hp["NUM_CLASSES"]])
        return patches, label

    return parse

def tf_dataset(images: List[str], hp: dict, batch_size: int = 32) -> tf.data.Dataset:
    """
    Converts image paths to a TensorFlow dataset pipeline.

    Args:
        images (List[str]): List of S3 image paths.
        hp (dict): Hyperparameters.
        batch_size (int): Batch size.

    Returns:
        tf.data.Dataset: Prepared TensorFlow dataset.
    """
    parse_fn = parse_fn_factory(hp) 
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def visualise_patches(patches: np.ndarray, patch_size: int = 64, num_channels: int = 3) -> None:
    """
    Visualises the first 64 patches of a given image.

    Args:
        patches (np.ndarray): Flattened patch tensor.
        patch_size (int): Size of each patch.
        num_channels (int): Number of channels (e.g., 3 for RGB).
    """
    reshaped_patches = patches[:64].reshape((-1, patch_size, patch_size, num_channels))
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(8, 8)
    for i in range(64):
        ax = plt.subplot(gs[i])
        ax.imshow(reshaped_patches[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def _download_from_s3(bucket: str, key: str, dst: Path) -> Path:
    """Download *key* from *bucket* to *dst* if it does not already exist."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading s3://{bucket}/{key} → {dst} …")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(dst))
    return dst


def _load_and_preprocess(img_path: Path, hp: dict) -> np.ndarray:
    """
    Load an image file, resize, normalise and convert it to the
    `(1, NUM_PATCHES, PATCH_DIM)` tensor expected by the ViT model.
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (hp["IMAGE_SIZE"], hp["IMAGE_SIZE"]))
    img = img / 255.0                                           
    patch_shape = (hp["PATCH_SIZE"],
                   hp["PATCH_SIZE"],
                   hp["NUM_CHANNELS"])
    patches = patchify(img, patch_shape,
                       step=hp["PATCH_SIZE"])                 
    patches = patches.reshape(hp["NUM_PATCHES"],
                              hp["PATCH_SIZE"]
                              * hp["PATCH_SIZE"]
                              * hp["NUM_CHANNELS"])            

    return np.expand_dims(patches.astype(np.float32), axis=0)  