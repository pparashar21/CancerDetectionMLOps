"""
data_ingestion.py
=================

End-to-end routine for **downloading the Kaggle dataset, un-zipping it, and
pushing the raw images to S3** so the rest of the pipeline can stay 100 %
cloud-native.

Typical usage
-------------
```bash
python -m CancerClassification.components.data_ingestion
"""
import sys
import os
import zipfile
import boto3
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except:
    KaggleApi = None
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from CancerClassification.utils.logger import logging
from CancerClassification.utils.exception_handler import ExceptionHandler
from CancerClassification.constants import *
from CancerClassification.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.api = None

    def download_data_local(self) -> None:
        try:
            if KaggleApi is None:
                    raise ImportError ("KaggleApi could not be imported. Possibly running in CI environment.")
            self.api = KaggleApi()
            self.api.authenticate()
            logging.info("Starting local download of .zip")
            self.api.dataset_download_files(self.config.kaggle_dataset_slug, path=ROOT_DIR, unzip=False)
            logging.info("Local download done!")
        except Exception as e:
            logging.error(ExceptionHandler(e, sys))

    def upload_to_S3(self) -> None:
        try:
            logging.info('Starting data upload to S3')
            s3 = boto3.client('s3')
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue
                    file_data = zip_ref.read(file_info.filename)
                    s3_key = os.path.join(self.config.s3_data_key, file_info.filename).replace("\\", "/")
                    s3.put_object(Bucket=self.config.s3_bucket, Key=s3_key, Body=file_data)
            
            os.remove(ZIP_PATH)
            logging.info(f"Uploaded data sucessfully to {s3_key}!")

        except Exception as e:
            logging.error(ExceptionHandler(e, sys))

