import os
import time
import utils
from sentence_transformers import SentenceTransformer
from model import FeedForwardClassifier
import torch


def load_model(checkpoint_path: str, args: dict) -> FeedForwardClassifier:
    """
    Load the trained model from the checkpoint
    :param checkpoint_path: path to the saved checkpoint file
    :return: loaded model
    """
    model = FeedForwardClassifier(args['model']['input_size'], args['model']['hidden_size'])

    # Load the trained parameters from the checkpoint file
    checkpoint = torch.load(checkpoint_path)

    # Load the parameters into the model
    model.load_state_dict(checkpoint['state_dict'])

    return model


if __name__ == "__main__":

    # load the experimental config
    src_dir = os.path.dirname(os.getcwd())
    experiment_config = utils.load_yaml_file(file_path=os.path.join(src_dir, "configuration"))

    # load the sentence transformer
    print("Loading sentence transformers model...")
    encoder = SentenceTransformer(experiment_config['dataset']['embeddings_name'])

    # load the pretrained model from the pipeline
    print("Loading the trained model...")
    checkpoint_path = os.path.join(src_dir, 'models', 'model.pt')
    model = load_model(checkpoint_path, experiment_config)

    while True:
        user_input = input("Enter a sentence (type 'exit' to quit): ")
        start_time = time.time()

        # Check if the user wants to exit
        if user_input.lower() == 'exit' or user_input.lower() == 'quit':
            break

        enc = encoder.encode(user_input)
        prediction = model(torch.tensor(enc))[0].tolist()

        end_time = time.time()
        ms = (end_time - start_time) * 1000
        if prediction <= 0.5: print(f"True News! == Time (ms): {ms}")
        if prediction >= 0.5: print(f"Fake News! == Time (ms): {ms}")
