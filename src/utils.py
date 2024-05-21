import yaml
import pandas as pd
import os
import nltk
import multiprocessing as mp

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import KFold, train_test_split
import string
from tqdm import tqdm

tqdm.pandas()
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


def load_yaml_file(file_path: str) -> dict:
    """
    A function which loads an experimental configuration.

    :param file_path: the location of the configuration file
    :return: a dictionary of experiment arguments
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")


def load_dataset_and_combine(fake_data: object, true_data: object) -> pd.DataFrame:
    """
    A function which loads true and fake data and combines them into a dataframe.

    :param fake_data: fake news articles
    :param true_data: true news articles
    :return: a combined dataframe
    """
    fake_data = pd.read_csv(fake_data)
    fake_data['fake'] = 1
    true_data = pd.read_csv(true_data)
    true_data['fake'] = 0
    return pd.concat([fake_data, true_data])


def process_text(text: str) -> str:
    """
    Applies all the preprocessing steps to a text.
    :param text: input str
    :return: preprocessed output str
    """

    text = lower(text)
    text = remove_special_characters(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = stem(text)
    return text


def process_column(column, num_cpus):
    """
    A function which applies multiprocessing to each column.

    :param dataset: input dataset which is unclean
    :return: a cleaned, usuable dataset
    """
    with mp.Pool(processes=num_cpus) as pool:
        processed_column = list(tqdm(pool.map(process_text, column), total=len(column)))
    return processed_column


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard NLP-based cleaning functions using nltk. and multiprocessing using Dask

    :param dataset: input dataset which is unclean
    :return: a cleaned, usuable dataset
    """

    dataset = check_nan_rows_remove(dataset=dataset)

    for column in ['title', 'text']:
        dataset[column] = process_column(dataset[column], os.cpu_count())

    return dataset


def check_nan_rows_remove(dataset: pd.DataFrame, columns=['title', 'text']) -> pd.DataFrame:
    """
    Checks if the dataset has any NaN rows and removes them

    :param dataset: pandas dataframe of combined data
    :param columns: list of column names to check
    :return: dataset without nan incides
    """
    nan_indices = dataset[columns].isna().any(axis=1)
    return dataset[~nan_indices]


def lower(text: str) -> str:
    """
    Convert text to lowercase.

    :param text: Input text.
    :return: Lowercased text.
    """
    return text.lower()


def remove_special_characters(text: str) -> object:
    """
    Remove special characters and punctuation from text.

    :param text: Input text.
    :return: Text with special characters and punctuation removed.
    """
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def tokenize(text: str) -> list:
    """
    Tokenize text into words.

    :param text: Input text.
    :return: List of tokens.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """
    Remove stopwords from a list of tokens.

    :param tokens: List of tokens.
    :return: List of tokens with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]


def stem(tokens: list) -> list:
    """
    Apply stemming to a list of tokens.

    :param tokens: List of tokens.
    :return: List of stemmed tokens.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def split_dataset(dataset: pd.DataFrame, k: int =5, type: str = 'classic_split') -> []:
    """
    A function which splits the dataset into training, testing and validation using something akin to a nested CV
    paradigm
    :param dataset: a combined dataset
    :param k: used to determine the number of nested splits
    :return: dictionary with a split dataset
    """
    splits = []
    if type == 'nested_cv':
        outer_cv = KFold(n_splits=k)
        for outer_train_idx, test_idx in outer_cv.split(dataset):
            train_df, test_df = dataset.iloc[outer_train_idx], dataset.iloc[test_idx]
            train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['fake'])

            splits.append(
                {
                    'training': train_df,
                    'testing': test_df,
                    'validating': val_df
                }
            )
    elif type == 'classic_split':

        train, test = train_test_split(dataset, test_size=0.2, stratify=dataset['fake'])
        train, valid = train_test_split(train, test_size=0.2, stratify=train['fake'])
        splits.append(
            {
                'training': train,
                'testing': test,
                'validating': valid
            }
        )

    return splits


def downsample_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Downsamples a dataset
    :param dataset: dataframe of combined data
    :return: a downsampled dataset
    """
    positives = dataset[dataset['fake']==1]
    negatives = dataset[dataset['fake']==0].sample(len(positives))
    return pd.concat([positives, negatives])