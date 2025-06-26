# Contains functionalities that we will use frequenlty
import pandas as pd
import os
import sys
from box.exceptions import BoxValueError
import kagglehub
from kagglehub import KaggleDatasetAdapter
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
from CancerClassification.constants import *
from CancerClassification.utils.logger import logging
from CancerClassification.utils.exception_handler import ExceptionHandler

def get_data_from_kaggle(file_name:str) -> pd.DataFrame:
    """
    Downloading a kaggle dataset as a Pandas DataFrame
    
    Args:
        file_name (str): The file name to be downloaded from Kaggle inside the mentioned KAGGLE_DATASET_SLUG

    Returns:
        df (pd.DataFrame) : The downloaded DataFrame from Kaggle
    """
    pass
    try:
        logging.info(f"Extracting {file_name} from Kaggle")
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            KAGGLE_DATASET_SLUG,
            file_name,
        )
        logging.info(f"Shape of the dataframe: {df.shape}")
        logging.info(f"Column names: {df.columns}")
        logging.info(f"Preview of the DataFrame:\n{df.head()}")
        logging.info("Data fetched successfully from Kaggle.")
        
        return df
    except Exception as e:
        logging.error(ExceptionHandler(e,sys))
        raise ExceptionHandler(e, sys)
    
@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    """
    Reads YAML file and returns ConfigBox
    Error handling: Check for empty yaml file
    """

    try:
        with open(path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file : {path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(paths:list, log=True):
    """
    Create list of directories

    paths : List of paths of directories to be created
    log : To choose to print to log or not, by default it will print (True)
    """

    for p in paths:
        os.makedirs(p, exist_ok = True)
        if log:
            logging.info(f"Directory created at path : {p}")


@ensure_annotations
def save_json(path:Path, data:dict):
    """
    Save json files

    path : path to store json objects
    data : dictionary storing our json data
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"JSON file stored at path : {path}")

@ensure_annotations
def load_json(path : Path) -> ConfigBox:
    """
    Load json files and store as ConfigBix data type

    path : path o the required JSON
    """

    with open(path) as f:
        content = json.load(f)

    logging.info(f"JSON object loaded successfully from path : {path}")
    return ConfigBox(content)

@ensure_annotations
def save_binary(data : Any, path : Path):
    """
    Save binary files 

    data : data of the binary file
    path : path to store the binary file
    """

    joblib.dump(value = data, filename = path)
    logging.info(f"Binary file saved at path : {path}")

@ensure_annotations
def load_binary(path: Path) -> Any:
    """ 
    Load binary files

    path : path of the binary file to be loaded
    """

    data = joblib.load(path)
    logging.info(f"Loaded binary file from path : {path}")
    return data

@ensure_annotations
def get_size(path : Path) -> str:
    """
    Get the size of a file in KB

    path : path to the source file
    """

    size_kb = round(os.path.getsize(path)/1024)
    return f"{size_kb} KB"

@ensure_annotations
def decodeImage(imgstr, file):
    imgdata = base64.b64decode(imgstr)
    with open(file, 'wb') as f:
        f.write(imgdata)
        f.close()

# To pass from html to base64
@ensure_annotations
def encodeImage(croppedImgPath):
    with open(croppedImgPath, "rb") as f:
        return base64.b64encode(f.read())    
