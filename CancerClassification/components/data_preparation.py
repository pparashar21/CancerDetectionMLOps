"""
data_preparation.py
===================

Turn raw S3 image keys into **TensorFlow `tf.data.Dataset` objects** that the
ViT model can train on.

High-level workflow
-------------------
1. **Read hyper-parameters** (image size, patch size, batch size, …) from
   *params.yaml* and augment them with values discovered at runtime
   (e.g. `CLASS_NAMES`).
2. **List images in S3** and create three python lists:
   *train*, *validation*, *test*.
3. **Map → Batch → Prefetch** those lists into `tf.data.Dataset`s using the
   parsing utilities in `CancerClassification.utils.utility`.

Typical usage
-------------
The object is usually instantiated by `model_trainer.py`, but you can also run
this file directly for a quick sanity-check:

```bash
python -m CancerClassification.components.data_preparation
"""
import sys
import os
import tensorflow as tf

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from CancerClassification.utils.logger import logging
from CancerClassification.utils.exception_handler import ExceptionHandler
from CancerClassification.constants import *
from CancerClassification.utils.utility import (
    load_hyperparameters,
    get_image_paths_from_s3,
    tf_dataset
)
from CancerClassification.config.configuration import configManager
from CancerClassification.entity.config_entity import DataPreparationConfig


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def run(self):
        try:
            logging.info("Starting data preparation pipeline...")

            # Load hyperparameters and compute derived fields
            try:
                hp = load_hyperparameters(
                    path=PARAMS_FILE_PATH,
                    s3_bucket=self.config.s3_bucket,
                    data_folder=self.config.class_structure
                )
                logging.info("Hyperparameters loaded and computed successfully.")
            except Exception as e:
                logging.error(f"Failed to load hyperparameters: {ExceptionHandler(e, sys)}")
                raise e

            # Get train, validation, and test image paths from S3
            try:
                train_paths, valid_paths, test_paths = get_image_paths_from_s3(
                    bucket_name=self.config.s3_bucket,
                    root_prefix=self.config.class_structure,
                    split=self.config.split_ratio
                )
                logging.info("Image paths fetched and split successfully from S3.")
                logging.debug(f"Train samples: {len(train_paths)}, Validation: {len(valid_paths)}, Test: {len(test_paths)}")
            except Exception as e:
                logging.error(f"Failed to retrieve image paths: {ExceptionHandler(e, sys)}")
                raise e

            # Convert image paths to TensorFlow datasets
            try:
                train_ds = tf_dataset(train_paths, hp, batch_size=hp["BATCH_SIZE"])
                valid_ds = tf_dataset(valid_paths, hp, batch_size=hp["BATCH_SIZE"])
                test_ds = tf_dataset(test_paths, hp, batch_size=hp["BATCH_SIZE"])
                logging.info("TensorFlow datasets created successfully.")
            except Exception as e:
                logging.error(f"Failed to create TensorFlow datasets: {ExceptionHandler(e, sys)}")
                raise e

            logging.info(" Data preparation pipeline completed successfully.")
            return train_ds, valid_ds, test_ds, hp["CLASS_NAMES"]

        except Exception as e:
            logging.error(f"Pipeline crashed: {ExceptionHandler(e, sys)}")
            raise e


if __name__ == "__main__":
    try:
        logging.info("Initialising config manager...")
        c = configManager()
        dpc = c.get_data_preparation_config()
        dp = DataPreparation(dpc)

        logging.info("Running DataPreparation...")
        train, valid, test, class_names = dp.run()
        print(class_names)

        # Print a single batch shape
        for i, j in train:
            print(i.shape, j.shape)
            break

    except Exception as e:
        logging.error(f"Fatal error in __main__: {ExceptionHandler(e, sys)}")