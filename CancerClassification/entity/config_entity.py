from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf

@dataclass
class DataIngestionConfig:
    kaggle_dataset_slug: str
    s3_bucket: str
    s3_data_key: str
    kaggle_folder_name: str

@dataclass
class DataPreparationConfig:
    s3_bucket: str
    s3_data_key: str
    class_structure: str
    split_ratio: float
    local_splits_dir: Path

@dataclass
class ModelTrainerConfig:
    train_ds: tf.data.Dataset
    valid_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    class_names: list[str]