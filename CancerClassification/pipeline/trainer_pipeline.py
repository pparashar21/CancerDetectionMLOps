"""trainer_pipeline.py
=====================

High-level script that chains data preparation and model training for
the cancer-classification project.

Workflow
--------
1. Instantiate :class:`configManager` to obtain all runtime settings.
2. Launch :class:`DataPreparation` - builds train/val/test `tf.data.Dataset`s.
3. Pass those datasets to :class:`ModelTrainer`, which:
   * constructs the ViT architecture,
   * trains it (with early-stopping & checkpoints),
   * evaluates on the test split,
   * uploads metrics and the SavedModel directory to S3.

The script is intended to be executed from the CLI:

```bash
python -m CancerClassification.pipeline.trainer_pipeline
"""

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