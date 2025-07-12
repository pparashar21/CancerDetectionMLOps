"""
model_trainer.py
----------------

Trains and evaluates the ViT model.
Separated from model_builder.py to ensure modular structure.

Usage:
    python -m CancerClassification.components.model_trainer
"""

import tensorflow as tf
from pathlib import Path
from tensorflow.keras import callbacks #type: ignore

from CancerClassification.utils.utility import read_yaml, load_hyperparameters, create_directory
from CancerClassification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from CancerClassification.config.configuration import configManager
from CancerClassification.components.data_preparation import DataPreparation
from CancerClassification.components.model_builder import build_vit  # previously ViT

class ModelTrainer:
    def __init__(self):
        self.cfg_mgr = configManager()
        self.dp_cfg = self.cfg_mgr.get_data_preparation_config()
        self.dp = DataPreparation(self.dp_cfg)
        self.cfg = self.cfg_mgr.get_model_trainer_config()
        self.hp = load_hyperparameters(
            PARAMS_FILE_PATH, 
            self.dp_cfg.s3_bucket, 
            self.dp_cfg.class_structure
        )
        self.model = build_vit(self.hp)

    def compile_model(self):
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=self.hp.get("LEARNING_RATE", 1e-4),
            weight_decay=self.hp.get("WEIGHT_DECAY", 1e-6)
        )
        self.model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        train_ds, val_ds, test_ds, class_names = self.dp.run()
        log_dir = self.cfg.tensorboard_log_dir
        ckpt_dir = self.cfg.checkpoint_dir
        create_directory([log_dir, ckpt_dir])

        cb = [
            callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
            callbacks.ModelCheckpoint(filepath=f"{ckpt_dir}/vit_best.h5", monitor='val_accuracy', save_best_only=True),
            callbacks.TensorBoard(log_dir=log_dir)
        ]

        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.hp.get("EPOCHS", 50),
            callbacks=cb,
            verbose=1
        )

    def evaluate(self):
        _, _, test_ds, _ = self.dp.run()
        results = self.model.evaluate(test_ds, verbose=1, return_dict=True)
        print("Test results:", results)
        return results

    def save_model(self):
        model_path = self.cfg.saved_model_path
        create_directory([Path(model_path).parent])
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def run(self):
        self.compile_model()
        self.train()
        self.evaluate()
        self.save_model()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()