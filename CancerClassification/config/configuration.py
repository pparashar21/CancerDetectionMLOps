import os
from CancerClassification.constants import *
from CancerClassification.utils.utility import read_yaml
from CancerClassification.entity.config_entity import (DataIngestionConfig, DataPreparationConfig)

class configManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            kaggle_dataset_slug =  config.kaggle_dataset_slug,
            s3_bucket = config.aws_bucket,
            s3_data_key = config.data_key,
            kaggle_folder_name = config.kaggle_folder_name
        )
        return data_ingestion_config 
    
    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation

        data_prep_config = DataPreparationConfig(
            s3_bucket = config.s3_bucket,
            s3_data_key= config.s3_data_key,
            class_structure = os.path.join(config.s3_data_key, config.class_structure),
            split_ratio= config.split_ratio,
            local_splits_dir= config.local_splits_dir
        )
        return data_prep_config