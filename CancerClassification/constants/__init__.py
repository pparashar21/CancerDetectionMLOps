#creating a file to store all the constants (values that will not change throughout project code)
import os
from pathlib import Path
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Log file directories
LOGS_DIRECTORY:Path = os.path.join(ROOT_DIR, "Logs")
LOGS_FILENAME:str = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"

# Local file directories
CONFIG_FILE_PATH:Path = os.path.join(ROOT_DIR, "config/config.yaml")
PARAMS_FILE_PATH:Path = os.path.join(ROOT_DIR, "config/params.yaml")
ZIP_PATH:Path = os.path.join(ROOT_DIR, '/multi-cancer.zip')

