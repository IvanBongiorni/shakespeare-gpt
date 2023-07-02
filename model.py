"""
Building GPT architecture
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import maximal
from maximal.layers import (
    PositionalEmbedding, GPTLayer
)

from config import config


def build_model():
    """
    Builds a GPT using Maximal and TensorFlow.
    Args:   / (just needs config params)
    Returns: GPT model (tf.keras.models.Model)
    """
    # Define nodes of the graph
    input_batch = Input(shape=(INPUT_LENGTH,), dtype=tf.int32)
    embedding = PositionalEmbedding(INPUT_LENGTH, VOCAB_SIZE, DEPTH)
    gpt_layers = [GPTLayer(depth=DEPTH, heads=HEADS, ff_nodes=FF_NODES) for _ in range(N_LAYERS)]
    classification_layer = Dense(VOCAB_SIZE)

    # Build the computational graph
    x = embedding(input_batch)

    for layer in gpt_layers:
        x = layer(x)

    classification = classification_layer(x)

    return Model(
        inputs=input_batch,
        outputs=classification
    )


def load_model():
    """
    If a model with a given name already exists
    :return:
    """
    return gpt


def load_or_build_model():

    # check if the model is

    #

    return gpt