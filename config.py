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

    # Model architecture
    INPUT_LENGTH = 128
    DEPTH = 512
    HEADS = 4
    FF_NODES = 1024
    N_LAYERS = 4

    SHOW_LOSS_HISTORY = True

    CORPUS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"

