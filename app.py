import os
import tempfile
from pathlib import Path
from typing import Tuple

import boto3
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from CancerClassification.components.model_builder import build_ViT
from CancerClassification.utils.utility import (
    _download_from_s3,
    _load_and_preprocess,
    get_class_names,
    load_hyperparameters,
)
from CancerClassification.config.configuration import configManager
from CancerClassification.constants import PARAMS_FILE_PATH

CFG_MGR = configManager()
DP_CFG = CFG_MGR.get_data_preparation_config()
INF_CFG = CFG_MGR.get_model_inference_config()

S3_BUCKET = INF_CFG.s3_bucket
WEIGHTS_KEY = INF_CFG.s3_model_weights  

def _load_model() -> Tuple["tf.keras.Model", dict, list[str]]:
    """Download weights (only once) and build/compile the ViT model."""
    import tensorflow as tf  
    hp = load_hyperparameters(PARAMS_FILE_PATH, DP_CFG.s3_bucket, DP_CFG.class_structure)
    class_names = get_class_names(DP_CFG.s3_bucket, DP_CFG.class_structure)

    model = build_ViT(hp)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    cache_dir = Path.home() / ".cache" / "mcancer_vit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_weights = cache_dir / Path(WEIGHTS_KEY).name

    if not local_weights.exists():
        _download_from_s3(S3_BUCKET, WEIGHTS_KEY, local_weights)  

    model.load_weights(local_weights)
    return model, hp, class_names

@st.cache_resource(show_spinner=False)
def get_model():
    return _load_model()


st.set_page_config(page_title="Multi-Cancer Classifier", layout="centered")
st.title("🔬 Multi-Cancer ViT Classifier")
st.caption("Upload a histopathology image. The ViT model will predict which of "
           "the %d cancer classes it belongs to." )

uploaded = st.file_uploader("Select a .jpg / .png file", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True, caption="Input image")

    with st.spinner("Running inference… this usually takes < 1 s"):
        # Persist uploaded file to a tmp path because existing util expects path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            tmp_path = Path(tmp.name)

        model, hp, class_names = get_model()
        x = _load_and_preprocess(tmp_path, hp)
        probs = model.predict(x, verbose=0).squeeze()

        # Ensure probs is (num_classes,) even for binary head
        if probs.ndim == 0:
            probs = np.stack([1 - probs, probs])

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        confidence = float(probs[pred_idx])

    st.markdown("## 🩺 Prediction")
    st.write(f"**Class:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2%}")

    with st.expander("Show all class probabilities"):
        import pandas as pd
        df = pd.DataFrame({"class": class_names, "probability": probs})
        st.dataframe(df.sort_values("probability", ascending=False), use_container_width=True)

###############################################################################
#                             SECURITY NOTES                                  #
###############################################################################
# • Set AWS credentials as environment variables *in the deployment platform*:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and (optionally) AWS_SESSION_TOKEN.
#   ▸ On Streamlit Cloud you can add them via *Settings → Secrets* and load with
#     boto3 by default; the variables are injected automatically.
#   ▸ On a private EC2 / container ensure the variables are in the environment
#     *or* attach an IAM role with the minimum S3 read-only permissions
#     (s3:GetObject on the weights key).
#
# • The model weights are cached on the server file-system and the model object
#   is cached in memory, so subsequent user requests are instant and *no new*
#   S3 calls are made—making inference fast and cost-effective.
