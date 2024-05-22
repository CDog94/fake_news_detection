import time
from model_definition import FeedForwardClassifier
import torch
import argparse
from transformers import pipeline


def load_model(checkpoint_path: str) -> FeedForwardClassifier:
    """
    Load the trained model from the checkpoint
    :param checkpoint_path: path to the saved checkpoint file
    :return: loaded model
    """
    model = FeedForwardClassifier(input_size=384, hidden_size=128)

    # Load the parameters into the model
    model.load_state_dict(torch.load(checkpoint_path))

    return model


def gather_inputs() -> argparse.Namespace:
    """
    Gather input arguments from the user
    :return: args
    """
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument('-embeddings_name', type=str, help='First argument', default='sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    parser.add_argument('-model_weights', type=str, help='Second argument', default='model.pt')

    return parser.parse_args()


if __name__ == "__main__":

    args = gather_inputs()

    feature_extractor = pipeline(
        task="feature-extraction",
        framework="pt",
        model=args.embeddings_name
    )

    model = load_model(args.model_weights)

    while True:
        user_input = input("Enter a sentence (type 'exit' to quit): ")
        start_time = time.time()

        if user_input.lower() == 'exit' or user_input.lower() == 'quit':
            break

        enc = feature_extractor(user_input, return_tensors="pt")[0].numpy().mean(axis=0)
        prediction = model(torch.tensor(enc))[0].tolist()

        end_time = time.time()
        ms = (end_time - start_time) * 1000
        if prediction <= 0.5: print(f"True News! == Time (ms): {ms}")
        if prediction >= 0.5: print(f"Fake News! == Time (ms): {ms}")
