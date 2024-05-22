import os
from tqdm import tqdm
import pandas as pd
import joblib
from torch.utils.data import Dataset, DataLoader
import utils
import torch
from transformers import pipeline


def get_dataset(args: dict) -> list:
    """
    A function which gets the dataset, either loads it or creates it from scratch.
    :param args: a set of arguments pertaining to the dataset
    :return: a dictionary of the dataset in its associated splits
    """
    if args['dataset']['load_dataset']:
        dataset = joblib.load(os.path.join(args['src_dir'], 'data', 'preprocessed', 'dataset.pkl'))
        return dataset
    else:
        dataset_cfg = args['dataset']
        dataset_dir = os.path.join(args['src_dir'], 'data', 'raw')

        # load the dataset
        dataset = utils.load_dataset_and_combine(
            true_data=os.path.join(dataset_dir, dataset_cfg['positive']),
            fake_data=os.path.join(dataset_dir, dataset_cfg['negative']),
        )

        # downsample the dataset for now.
        dataset = utils.downsample_dataset(dataset)

        dataset = dataset.sample(2500)

        # apply preprocessing functions
        dataset = utils.clean_dataset(dataset=dataset)

        # remove irrelevant data (for now).
        dataset = dataset.drop(columns=['title', 'subject', 'date'])

        # get sentence embeddings
        dataset = get_sentence_embeddings(args=args, dataset=dataset, feature='text')

        # randomly shuffle the dataset
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        # split the dataset
        dataset = utils.split_dataset(dataset=dataset)

        # convert to torch tensors
        dataset = prepare_dataset_for_model(dataset=dataset)

        # write to disk
        joblib.dump(dataset, os.path.join(args['src_dir'], 'data', 'preprocessed', 'dataset.pkl'))

        return dataset


def get_sentence_embeddings(args: dict, dataset: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Generates sentence embedding for each sentence in the dataset.
    :param args: arguments pertaining to the dataset
    :param dataset: a pandas dataframe containing the sentences
    :param feature: a string indicating the name of the feature
    :return: a dataframe of the sentence embeddings
    """
    sentences = [' '.join(s) for s in dataset[feature].tolist()]

    with torch.no_grad():

        feature_extractor = pipeline(
            task="feature-extraction",
            framework="pt",
            model=args['dataset']['embeddings_name'],
            device=0,
        )

        for i, each_sentence in enumerate(tqdm(sentences, desc='Generating sentence embeddings')):
            emb = feature_extractor(each_sentence, return_tensors="pt")
            avg_pool = emb[0].numpy().mean(axis=0).tolist()
            sentences[i] = avg_pool

    dataset[feature] = sentences
    return dataset


def prepare_dataset_for_model(dataset: list) -> list:
    """
    Prepares a dataset for a Pytorch model
    :param dataset: dictionary of data splits
    :return: a fully prepared dataset
    """
    for k, split in enumerate(dataset):
        dataset[k] = to_pytorch_dataset(dataset=split)
    return dataset


def to_pytorch_dataset(dataset: dict) -> dict:
    """
    Prepare a dataset for a Pytorch model
    :param dataset: a dictionary of data splits
    :return: a fully prepared dataset
    """
    train, test, valid = dataset['training'], dataset['testing'], dataset['validating']

    X_tr, y_tr = train['text'].to_numpy(), train['fake'].to_numpy()
    X_te, y_te = test['text'].to_numpy(), test['fake'].to_numpy()
    X_val, y_val = valid['text'].to_numpy(), valid['fake'].to_numpy()

    # custom pytorch datasets
    train_dataset = CustomDataset(X_tr, y_tr)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_te, y_te)

    # to data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return {'train': train_loader, 'test': test_loader, 'valid': val_loader}


class CustomDataset(Dataset):
    """
    A custom dataset wrapper for the Pytorch Dataset.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])