"""
model_builder.py
----------------

Component to train the Vision Transformer model for Multi-Cancer Detection.

This script is intended to be executed after `data_preparation.py`. It:
    • Loads the processed datasets produced by `DataPreparation`
    • Builds a Vision Transformer (ViT) with hyper-parameters from `params.yaml`
    • Compiles, trains and evaluates the network
    • Persists the best model and logs

Execute as a standalone module:

    python -m CancerClassification.components.model_trainer
"""

from CancerClassification.config.configuration import configManager
from CancerClassification.components.data_preparation import DataPreparation
from CancerClassification.constants import *
from CancerClassification.utils.utility import read_yaml, load_hyperparameters
import tensorflow as tf
from tensorflow.keras import layers, models #type:ignore



class ClassToken(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        return tf.cast(cls, dtype=inputs.dtype)


def mlp(x, cf):
    x = layers.Dense(cf['MLP_DIM'], activation='gelu')(x)
    x = layers.Dropout(cf['DROPOUT_RATE'])(x)
    x = layers.Dense(cf['HIDDEN_DIM'])(x)
    x = layers.Dropout(cf['DROPOUT_RATE'])(x)
    return x


def transformer_encoder(x, cf):
    skip_1 = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=cf['NUM_HEADS'], key_dim=cf['HIDDEN_DIM'])(x, x)
    x = layers.Add()([x, skip_1])

    skip_2 = x
    x = layers.LayerNormalization()(x)
    x = mlp(x, cf)
    x = layers.Add()([x, skip_2])
    return x


def build_ViT(cf):
    input_shape = (cf['NUM_PATCHES'], cf['PATCH_SIZE'] * cf['PATCH_SIZE'] * cf['NUM_CHANNELS'])
    inputs = layers.Input(input_shape)
    patch_embed = layers.Dense(cf['HIDDEN_DIM'])(inputs)

    positions = tf.range(start=0, limit=cf['NUM_PATCHES'], delta=1)
    pos_emb = layers.Embedding(input_dim=cf['NUM_PATCHES'], output_dim=cf['HIDDEN_DIM'])(positions)
    embed = patch_embed + pos_emb

    token = ClassToken()(embed)
    x = layers.Concatenate(axis=1)([token, embed])

    for _ in range(cf['NUM_LAYERS']):
        x = transformer_encoder(x, cf)

    x = layers.LayerNormalization()(x)
    x = x[:, 0, :]
    x = layers.Dense(cf['NUM_CLASSES'], activation='softmax')(x)
    return models.Model(inputs, x, name='ViT')

if __name__ == "__main__":
    config = read_yaml(CONFIG_FILE_PATH)
    params = read_yaml(PARAMS_FILE_PATH)    
    print("Initialising config manager...")
    c = configManager()
    dpc = c.get_data_preparation_config()
    dp = DataPreparation(dpc)
    hp = load_hyperparameters(
    path=PARAMS_FILE_PATH,
    s3_bucket=dp.config.s3_bucket,
    data_folder=dp.config.class_structure
    )
    model = build_ViT(hp)
    model.summary()