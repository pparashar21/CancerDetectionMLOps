#creating a file to store all the constants (values that will not change throughout project code)
from pathlib import Path
from datetime import datetime

LOGS_DIRECTORY:str = "Logs"
LOGS_FILENAME:str = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"

#kaggle constants to extract data
KAGGLE_DATASET_SLUG:str = "obulisainaren/multi-cancer"
KAGGLE_FOLDER_NAME:str = "Multi Cancer/Multi Cancer/"

#AWS buckets and keys for different stages
AWS_BUCKET:str = "multi-cancer-mlops-prasoon"
DATA_INGESTION_FOLDER_KEY:str = "Data"

CONFIG_FILE_PATH:Path = Path("/Users/pparashar21/Desktop/Projects/CancerDetectionMLOps/config/config.yaml")
PARAMS_FILE_PATH:Path = Path("/Users/pparashar21/Desktop/Projects/CancerDetectionMLOps/params.yaml")

