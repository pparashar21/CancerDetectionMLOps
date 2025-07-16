"""config_entity.py
===================

Strongly-typed configuration “contracts” for every stage of the pipeline.

Each dataclass below represents the exact set of fields expected by a
specific pipeline component. Keeping them in one place helps catch typos early
(vs. using plain dictionaries) and provides IDE autocompletion.

Typical usage
-------------
```python
from CancerClassification.entity.config_entity import DataPreparationConfig

cfg = DataPreparationConfig(
    s3_bucket="my-bucket",
    s3_data_key="Datasets/HistImages",
    class_structure="Datasets/HistImages/ALL/",
    split_ratio=0.09,
    local_splits_dir=Path("./data_splits"),
"""
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
    s3_bucket:str
    s3_model_path: str
    s3_model_csv:str

@dataclass
class ModelInferenceConfig:
    s3_bucket:str
    s3_model_weights: str
    s3_model_weights_best:str