import sys
from CancerClassification.utils.logger import logging
from CancerClassification.utils.exception_handler import ExceptionHandler
from CancerClassification.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f">>>>> Stage, {STAGE_NAME} started! <<<<<")
    # obj = DataIngestionTrainingPipeline()
    # obj.main()
    a = 1/0
    logging.info(f">>>>> Stage, {STAGE_NAME} completed! <<<<<")

except Exception as e:
    logging.exception(ExceptionHandler(e, sys))