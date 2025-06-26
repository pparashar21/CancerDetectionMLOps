import os
import logging
from datetime import datetime
from CancerClassification.constants import *

logs_dir = os.path.join(os.getcwd(), LOGS_DIRECTORY)
os.makedirs(logs_dir, exist_ok=True)

LOGS_FILE_PATH = os.path.join(logs_dir, LOGS_FILENAME)

logging.basicConfig(
    filename= LOGS_FILE_PATH,
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)