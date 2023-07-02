"""
Inference
"""
import numpy as np
import tensorflow as tf
import maximal

from config import config
from models import load_or_build_model


def nlg():
    print("Loading model...")

    gpt = load_or_build_model()

    print("Completed.")

    print(config.MSG_GREETINGS)

    while true:
        prompt = input("User: ")

        if prompt < config.INPUT_LENGTH:
            print(f"Please provide a prompt of {config.INPUT_LENGTH}")

            # If prompt too short send a shakespearean message
            print(config.MSG_INPUT_TOO_SHORT.format(config.INPUT_LENGTH))
            continue

        generated_text = generate_text(prompt, config)
        print(f"\nShakespeare-GPT: {generated_text}\n")