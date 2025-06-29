import sys
import os
import zipfile
import boto3
from kaggle.api.kaggle_api_extended import KaggleApi
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
        self.api = KaggleApi()
        self.api.authenticate()

    def download_data_local(self) -> None:
        try:
            logging.info("Starting local download of .zip")
            self.api.dataset_download_files(KAGGLE_DATASET_SLUG, path=ROOT_DIR, unzip=False)
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
                    s3_key = os.path.join(DATA_INGESTION_FOLDER_KEY, file_info.filename).replace("\\", "/")
                    s3.put_object(Bucket=AWS_BUCKET, Key=s3_key, Body=file_data)
            
            os.remove(ZIP_PATH)
            logging.info(f"Uploaded data sucessfully to {s3_key}!")

        except Exception as e:
            logging.error(ExceptionHandler(e, sys))


if __name__ == "__main__":
    #to write test cases here...
    pass
