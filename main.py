from CancerClassification import logger
from CancerClassification.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> Stage, {STAGE_NAME} started! <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage, {STAGE_NAME} completed! <<<<<")
    #check

except Exception as e:
    logger.exception(e)
    raise e