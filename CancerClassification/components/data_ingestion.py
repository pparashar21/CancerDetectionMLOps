import os
import zipfile
import gdown
from CancerClassification.utils import logger
from CancerClassification.utils.utility import get_size
from CancerClassification.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """ 
        Fetch data from url
        """

        try:
            url = self.config.source_URL
            zip_file = self.config.local_data_file

            os.makedirs("data/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {url} into file {zip_file}")

            file_id = url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id, zip_file)
            logger.info(f"Downloaded data from {url} into file {zip_file}")

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        unzip_dir : str
        Extracts the zip file into the data directory
        Returns none
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)