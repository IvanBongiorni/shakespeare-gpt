# Shakespeare GPT
This is a tutorial repository to show how easy it is to build a custom GPT model using [Maximal](https://github.com/IvanBongiorni/maximal) and *TensorFlow 2.x* libraries.

For a more in depth documentation on how to build Transformer neural networks with **Maximal** and TensorFlow, refer to the [Official documentation](https://ivanbongiorni.github.io/maximal/) (Tutorials section).

## How to start
You must train a model via: `python train.py`. All hyperparameters can be tweaked in `config.py`.
#### WARNING: a GPU is strongly suggested to run this code. 

## How to generate code
Once a model is saved and stored in `/saved_models` folder, you can run: `python chat.py` to load the model and generate text via command line.

In the first public version of this repository, prompt length must be at least as long as the model's input size (controllable in `config.py`).
This is a constraint that will be removed in further versions.

To end the generation either use `Ctr+C` or type `exit`.
