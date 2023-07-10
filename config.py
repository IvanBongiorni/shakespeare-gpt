"""
Configurations
"""
from utils import StrMessages


class config(StrMessages):
    MODEL_NAME = "gpt_maximal_00"

    # NLG
    N_GENERATION = 1000
    TEMPERATURE = 1.0
    TOP_K_SAMPLE = 10

    # Training hyperparams
    LEARNING_RATE = 10e-5
    N_EPOCHS = 1
    BATCH_SIZE = 64

    # Model architecture
    VOCAB_SIZE = None  #placeholder
    INPUT_LENGTH = 128
    DEPTH = 512
    HEADS = 4
    FF_NODES = 1024
    N_LAYERS = 4

    SHOW_LOSS_HISTORY = True

    CORPUS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"

