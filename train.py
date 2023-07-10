"""
Training
"""
import os
import requests
import time
from pdb import set_trace as BP

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

from config import config
from model import load_or_build_model


def numerical_encoding(text: str, char_dict: dict) -> np.array:
    """
    First breaks text into a list of chars, then converts each to
    its numerical idx (np.array)

    Args:
        text (str): corpus to be vectorized
        char_dict (dict): dictionary to map chars to indexes

    Returns:
        chars_list (np.array): vectorized corpus
    """
    chars_list = [ char for char in text ]
    chars_list = [ char_dict[char] for char in chars_list ]
    chars_list = np.array(chars_list)
    return chars_list


def get_text_matrix(sequence: np.array, len_input: int) -> np.array:
    """
    Generates a matrix containing all sequences
    of length INPUT_LENGTH to be fed into the Network

    Args:
        sequence (np.array): array to be processed
        len_input (int): length od model input
    """
    # create empty matrix
    X = np.empty((len(sequence)-len_input, len_input))

    # fill each row/time window from input sequence
    for i in range(X.shape[0]):
        X[i,:] = sequence[i : i+len_input]

    return X


def process_corpus() -> (np.array, dict):
    """
    Text preprocessing steps: 1. Downloads corpus; 2. extracts set of
    unique chars; 3. Map every char to its int; 3. vectorize text;
    4. process vectorized text into a 2D array for model training
    (a sliding window of text is produced)

    Returns:
        X (np.array): 2D array for model training
        char2idx (dict): dictionary to preserve char-index mapping
    """
    page = requests.get(config.CORPUS_URL)
    text = page.text

    # Store list of unique characters
    unique_chars = list(set(text))
    unique_chars.sort()

    # Map every letter in our alphabet to an int
    char2idx = {char[1]: char[0] for char in enumerate(unique_chars)}

    # vectorize text
    encoded_text = numerical_encoding(text, char2idx)

    # Sequence of vectorized chars to 2D array
    X = get_text_matrix(encoded_text, config.INPUT_LENGTH + 1)

    return X, char2idx


@tf.function
def train_on_batch(gpt, optimizer, x, y):
    with tf.GradientTape() as tape:

        batch_loss = tf.reduce_sum(
            tf.keras.losses.sparse_categorical_crossentropy(
                y, gpt(x),
                from_logits=True)
        )

    gradients = tape.gradient(batch_loss, gpt.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gpt.trainable_variables))
    return batch_loss


def main():
    """
    Model training.

    """
    X, char2idx = process_corpus()

    config.VOCAB_SIZE = len(char2idx)

    loss_history = []

    gpt = load_or_build_model(verbose=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    print(f"\nStart model training for {config.N_EPOCHS} epochs\n")

    for epoch in range(config.N_EPOCHS):
        start = time.time()

        # Reshuffle data at each epoch to randomize mini-batch composition
        reshuffle = np.random.choice(X.shape[0], X.shape[0], replace=False)
        X = X[reshuffle]

        for iteration in range(X.shape[0] // config.BATCH_SIZE):
            # take new minibatch (with 1 char shift from x to y)
            take = iteration * config.BATCH_SIZE
            x = X[take:take + config.BATCH_SIZE, :-1]  # chars [0:128]
            y = X[take:take + config.BATCH_SIZE, 1:]  # chars [1:129]

            # training step
            current_loss = train_on_batch(gpt, optimizer, x, y)

            # periodically store batch loss into history
            if iteration % 100 == 0:
                loss_history.append(current_loss)
                print(f"\t{iteration}\tLoss: {current_loss}")

        print("{}.  \t  Loss: {}  \t  Time: {}ss".format(
            epoch + 1, current_loss.numpy(), round(time.time() - start, 2)))


    if config.SHOW_LOSS_HISTORY:
        # Visualize Loss history
        plt.figure(figsize=(15, 7))
        plt.plot(loss_history)
        plt.title('Loss History')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (Sparse CCE)')
        plt.show()

    # BP()

    # Save model
    gpt.save(os.path.join(os.getcwd(), "saved_models", config.MODEL_NAME+".h5"))
    print(f"Model {config.MODEL_NAME}.h5 saved in {os.path.join(os.getcwd(),'saved_models')}")

    return None


if __name__ == "__main__":
    main()