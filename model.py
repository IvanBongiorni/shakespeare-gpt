"""
Building GPT architecture
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import maximal
from maximal.layers import (
    PositionalEmbedding, GPTLayer
)

from config import config


def build_model() -> tf.keras.models.Model:
    """
    Builds a GPT using Maximal and TensorFlow.
    Args:   / (just needs config params)
    Returns: GPT model (tf.keras.models.Model)
    """
    # Define nodes of the graph
    input_batch = Input(shape=(config.INPUT_LENGTH,), dtype=tf.int32)

    embedding = PositionalEmbedding(config.INPUT_LENGTH, config.VOCAB_SIZE, config.DEPTH)

    gpt_layers = [GPTLayer(depth=config.DEPTH, heads=config.HEADS, ff_nodes=config.FF_NODES) for _ in range(config.N_LAYERS)]

    classification_layer = Dense(config.VOCAB_SIZE)

    # Build the computational graph
    x = embedding(input_batch)

    for layer in gpt_layers:
        x = layer(x)

    classification = classification_layer(x)

    return Model(
        inputs=input_batch,
        outputs=classification
    )


def load_or_build_model(verbose: bool =False) -> tf.keras.models.Model:
    """
    Checks if a model with name MODEL_NAME is already stored in /saved_models
    folder. If present, loads the existing one (to train it further). If not, it
    builds a new one.

    Args:
        verbose (bool): print model.summary() or not - defaults to False
    """
    filenames = os.listdir(os.path.join(os.getcwd(), "saved_models"))

    if config.MODEL_NAME in filenames:
        print(f"Loading existing model: {config.MODEL_NAME}.h5")
        gpt = tf.keras.models.load_model(os.path.join(os.getcwd(), "saved_models", config.MODEL_NAME))
    else:
        print(f"Creating a new model: {config.MODEL_NAME}.h5")
        gpt = build_model()

    if verbose:
        print(gpt.summary())

    return gpt