"""
Inference
"""
import os

import numpy as np
import tensorflow as tf
from maximal.layers import PositionalEmbedding, GPTLayer

from tqdm import tqdm

from config import config
from train import process_corpus


def generate_text(
        gpt: tf.keras.models.Model,
        prompt: str,
        char2idx: dict,
        idx2char: dict,
        n: int = config.N_GENERATION,
        temperature: float = config.TEMPERATURE,
        k: int = config.TOP_K_SAMPLE
    ) -> str:
    """
    Inference time for the GPT.

    Args:
        prompt (str): input text
        char2idx (dict): char -> idx mapping
        idx2char (dict): idx -> char mapping (inverse of original char2idx)
        n (int): number of tokens to be generated
        temperature (float): noise in the output probability
                            (>1. = noisy sampling; <1. = conservative sampling.)
        k (int): restricts to number of top-k tokens to be sampled from

    Returns:
        generated_text (str): GPT completion
    """
    # If prompt is shorter than INPUT_LENGTH raise error (no padding in this simple tutorial)
    assert len(prompt) >= config.INPUT_LENGTH, f"Prompt must be of {config.INPUT_LENGTH} character length"

    # If prompt is longer than INPUT_LENGTH crop it to last piece
    if len(prompt) > config.INPUT_LENGTH:
        prompt = prompt[-config.INPUT_LENGTH:]

    generated_text = []

    for i in tqdm(range(n)):
        # vectorize prompt and adjust np.array shape
        vectorized_text = [char2idx[c] for c in prompt]
        vectorized_text = np.array(vectorized_text).reshape((1, len(vectorized_text)))

        # next token prediction
        pred = gpt.predict(vectorized_text, verbose=0)
        pred = np.squeeze(pred[:, -1, :])

        # temperature scaling
        pred /= temperature

        # restrict sampling to top k tokens
        probs, indices = tf.math.top_k(pred, k, sorted=True)

        # sample token id
        probs = tf.nn.softmax(probs).numpy()
        pred_id = np.random.choice(indices, p=probs)

        # update prompt
        next_char = idx2char[pred_id]
        prompt = prompt[1:] + next_char
        generated_text.append(next_char)

    generated_text = ''.join(generated_text)

    return generated_text


def main():
    """
    Natural Language Generation.
    Starts an infinite loop that can be broken only via Ctrl+C or by
    typing "exit" as prompt.
    """
    # TODO: this is an overkill - change that, pickle dict locally
    _, char2idx = process_corpus()
    idx2char = {v: k for k, v in char2idx.items()}

    print(f"Loading model: {config.MODEL_NAME}.h5")
    gpt = tf.keras.models.load_model(
        os.path.join(os.getcwd(), "saved_models", config.MODEL_NAME+".h5"),
        custom_objects = {"PositionalEmbedding": PositionalEmbedding, "GPTLayer": GPTLayer}
    )
    print("Completed.")

    print(config.MSG_GREETINGS)

    # Start infinite loop
    while True:
        prompt = input("\nUser:\n")

        if len(prompt) < config.INPUT_LENGTH:
            print(f"Please provide a prompt of {config.INPUT_LENGTH}")

            # If prompt too short send a shakespearean message
            print(config.MSG_INPUT_TOO_SHORT.format(config.INPUT_LENGTH))
            continue
        elif prompt == "exit":
            print(config.MSG_FAREWELL)
            quit()
        else:
            generated_text = generate_text(gpt, prompt, char2idx, idx2char)
            print(f"\nShakespeare-GPT:\n{generated_text}\n")


if __name__ == "__main__":
    main()