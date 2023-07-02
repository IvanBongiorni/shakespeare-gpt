"""
Training
"""
import requests

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import config
from model import load_or_build_model


# globals
gpt = load_or_build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)


def numerical_encoding(text, char_dict):
    """
    First breaks text into a list of chars, then converts each to
    its numerical idx (np.array)
    """
    chars_list = [ char for char in text ]
    chars_list = [ char_dict[char] for char in chars_list ]
    chars_list = np.array(chars_list)
    return chars_list


def get_text_matrix(sequence, len_input):
    """
    This generates a matrix containing all the sequences
    of length INPUT_LENGTH to be fed into the Network
    """
    # create empty matrix
    X = np.empty((len(sequence)-len_input, len_input))

    # fill each row/time window from input sequence
    for i in range(X.shape[0]):
        X[i,:] = sequence[i : i+len_input]

    return X


def process_corpus():
    page = requests.get(config.CORPUS_URL)
    text = page.text

    # Store list of unique characters
    unique_chars = list(set(text))
    unique_chars.sort()

    # Map every letter in our alphabet to an int
    char2idx = {char[1]: char[0] for char in enumerate(unique_chars)}

    # Produce a reverse dictionary to go back from int to str later
    idx2char = {v: k for k, v in char2idx.items()}

    encoded_text = numerical_encoding(text, char2idx)

    X = get_text_matrix(encoded_text, INPUT_LENGTH + 1)

    return X


@tf.function
def train_on_batch(x, y):
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
    X = process_corpus()

    loss_history = []

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
            current_loss = train_on_batch(x, y)

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

    # Save model
    gpt.save(os.path.join(os.getcwd(), "saved_models", config.MODEL_NAME))

    return None



if __name__ == "__main__":
    main()