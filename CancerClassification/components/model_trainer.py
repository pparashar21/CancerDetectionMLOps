"""
model_trainer.py
----------------

Trains and evaluates the ViT model.
Uploads trained model and metrics to S3.

Usage:
    python -m CancerClassification.components.model_trainer
"""

import os
import io
import json
import boto3
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow.keras import callbacks  # type: ignore

from CancerClassification.utils.utility import read_yaml, load_hyperparameters
from CancerClassification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from CancerClassification.config.configuration import configManager
from CancerClassification.components.data_preparation import DataPreparation
from CancerClassification.components.model_builder import build_ViT


class ModelTrainer:
    def __init__(self, train_ds, val_ds, test_ds, class_names):
        self.cfg_mgr = configManager()
        self.dp_cfg = self.cfg_mgr.get_data_preparation_config()
        self.cfg = self.cfg_mgr.get_model_trainer_config()
        self.hp = load_hyperparameters(
            PARAMS_FILE_PATH,
            self.dp_cfg.s3_bucket,
            self.dp_cfg.class_structure
        )
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.class_names = class_names
        self.model = build_ViT(self.hp)
        self.s3 = boto3.client('s3')

    def compile_model(self):
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=self.hp.get("LEARNING_RATE", 1e-4),
            clipvalue = 1.0
        )
        self.model.compile(optimizer=optimiser, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self):
        cb = [
            callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
            callbacks.ModelCheckpoint(filepath="vit_best.h5", monitor='val_accuracy', save_best_only=True),
            callbacks.TensorBoard(log_dir="./logs")
        ]

        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.hp.get("EPOCHS", 50),
            callbacks=cb,
            verbose=1
        )

    def evaluate(self):
        results = self.model.evaluate(self.test_ds, verbose=1, return_dict=True)
        print("Test results:", results)

        # Save results to CSV buffer
        df = pd.DataFrame([results])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        # Upload to S3
        self.s3.put_object(
            Bucket=self.cfg.s3_bucket,
            Key=self.cfg.s3_model_csv,
            Body=csv_buffer.getvalue()
        )

        print(f"Evaluation metrics uploaded to s3://{self.cfg.s3_bucket}/{self.cfg.s3_model_csv}")
        return results

    def save_model(self):
        # Save to a temporary path
        save_path = Path("saved_model")
        self.model.save(str(save_path))

        # Upload to S3
        for root, _, files in os.walk(save_path):
            for file in files:
                local_path = Path(root) / file
                relative_path = local_path.relative_to(save_path)
                s3_key = os.path.join(self.cfg.s3_model_path, str(relative_path))

                self.s3.upload_file(
                    Filename=str(local_path),
                    Bucket=self.cfg.s3_bucket,
                    Key=s3_key
                )
        print(f"Model saved and uploaded to s3://{self.cfg.s3_model_path}")

    def run(self):
        self.compile_model()
        self.train()
        self.evaluate()
        self.save_model()


if __name__ == "__main__":
    c = configManager()
    dpc = c.get_data_preparation_config()
    dp = DataPreparation(dpc)
    train, valid, test, class_names = dp.run()
    trainer = ModelTrainer(train, valid, test, class_names)
    trainer.run()