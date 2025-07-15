"""
model_inference.py
------------------
Utility script to download the latest **ViT** weights from S3 and run
inference on a single image.

The script prints the predicted class label (and associated probability)
for the supplied image.
"""

from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
import numpy as np

from CancerClassification.components.model_builder import build_ViT
from CancerClassification.utils.utility import load_hyperparameters, _download_from_s3, _load_and_preprocess, get_class_names
from CancerClassification.config.configuration import configManager
from CancerClassification.utils.logger import logging
from CancerClassification.constants import *

def run_inference(
    image_path: Path,
    bucket: str,
    weights_key: str
) -> tuple[str, float]:
    
    """Load weights from S3, predict the class of image_path and return it."""
    logging.info("Inference module started:")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cfg_mgr = configManager()
        dp_cfg = cfg_mgr.get_data_preparation_config()
        hp = load_hyperparameters(
            PARAMS_FILE_PATH,
            dp_cfg.s3_bucket,
            dp_cfg.class_structure
        )
        logging.info("Hyperparameters loaded")
        class_names = get_class_names(dp_cfg.s3_bucket, dp_cfg.class_structure)
        model = build_ViT(hp)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        
        weights_local = tmpdir_path / Path(weights_key).name
        _download_from_s3(bucket, weights_key, weights_local)
        model.load_weights(weights_local)
        logging.info("Model weights loaded into the system")

    x = _load_and_preprocess(image_path, hp)
    logging.info("Image preprocessed!")
    probs = model.predict(x, verbose=0).squeeze()

    print(probs)
    if probs.ndim == 0:
        probs = np.stack([1 - probs, probs])

    predicted_idx = int(np.argmax(probs))
    predicted_label = class_names[predicted_idx]
    confidence = float(probs[predicted_idx])

    return predicted_label, confidence

def parse_args() -> argparse.Namespace:  # noqa: D401
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run inference with ViT model")
    parser.add_argument("--image_path", required=True, type=Path, help="Path to the input image")
    return parser.parse_args()


def main() -> None:  # noqa: D401
    """Console script entry-point."""
    config = configManager()
    cmi = config.get_model_inference_config()
    args = parse_args()
    label, conf = run_inference(
        image_path=args.image_path,
        bucket=cmi.s3_bucket,
        weights_key=cmi.s3_model_weights_best,
    )
    logging.info("\nPrediction from the model")
    logging.info(f"image     : {args.image_path}")
    logging.info(f"predicted : {label}")
    logging.info(f"confidence: {conf:.4f}")


if __name__ == "__main__":
    main()
