import os
import utils
import model as m
import data_representation as da
import evaluation as e
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_pipeline(args: dict):
    """
    Run the ML pipeline using the arguments provided.
    :param args: a dictionary of arguments.
    :return: Nothing.
    """
    dataset = da.get_dataset(args=args)
    predicted_proba = m.train_test_model(args=args, dataset=dataset)
    e.evaluate_model(
        predictions=predicted_proba,
        dataset=dataset,
        save_dir=os.path.join(experiment_config['src_dir'], 'results')
    )


if __name__ == "__main__":

    src_dir = os.path.dirname(os.getcwd())
    experiment_config = utils.load_yaml_file(file_path=os.path.join(src_dir, "configuration"))
    experiment_config['src_dir'] = src_dir
    experiment_config['device'] = device
    run_pipeline(args=experiment_config)