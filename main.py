import sys
from CancerClassification.config.configuration import configManager
from CancerClassification.components.data_preparation import DataPreparation
from CancerClassification.components.model_trainer import ModelTrainer
from CancerClassification.utils.logger import logging
from CancerClassification.utils.exception_handler import ExceptionHandler

try:
    logging.info(">>>>>>>>>>>>>>>>>>>>Initialising config manager<<<<<<<<<<<<<<<<<<<<<<<<<")
    c = configManager()
    dpc = c.get_data_preparation_config()
    dp = DataPreparation(dpc)

    logging.info(">>>>>>>>>>>>>>>>>>>>Running Data Preparation<<<<<<<<<<<<<<<<<<<<<<<<<")
    train, valid, test, class_names = dp.run()
    print(class_names)

    logging.info(">>>>>>>>>>>>>>>>>>>>Data Preparation done!<<<<<<<<<<<<<<<<<<<<<<<<<")
    logging.info(">>>>>>>>>>>>>>>>>>>>Starting Model Building<<<<<<<<<<<<<<<<<<<<<<<<<")
    trainer = ModelTrainer(train, valid, test, class_names)
    logging.info(">>>>>>>>>>>>>>>>>>>>Model Architecture Created<<<<<<<<<<<<<<<<<<<<<<<<<")
    logging.info(trainer.model.summary)
    logging.info(">>>>>>>>>>>>>>>>>>>>Starting Model Training<<<<<<<<<<<<<<<<<<<<<<<<<")
    trainer.run()


except Exception as e:
    logging.exception(ExceptionHandler(e, sys))