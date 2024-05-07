import os
import logging
from bertopic import BERTopic
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import utils as utils

def compute_summary_stats(dataset: pd.DataFrame, results_dir: str):
    """
    Computes basic summary statistics on the dataset
    :param dataset: pandas dataframe of the combined dataset
    :param results_dir: where to write results
    """
    dataset = utils.check_nan_rows_remove(dataset)

    samples_per_class = dataset.groupby('fake').size()
    fake_after_nan = samples_per_class[1]
    true_after_nan = samples_per_class[0]
    data = {'Category': ['fake', 'real'], 'Value': [fake_after_nan, true_after_nan]}
    sns.barplot(x='Category', y='Value', data=data)
    plt.title(f'Bar Chart: Fake vs. True After NaN. N={fake_after_nan + true_after_nan}')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.savefig(os.path.join(results_dir, 'summary.png'))
    plt.close()

    # compute statistics on category counts within each class.
    count_df = dataset.groupby(['subject', 'fake']).size().unstack(fill_value=0)
    sns.heatmap(count_df, annot=True, fmt='g')
    plt.title('Sample breakdown Fake vs. True articles')
    plt.savefig(os.path.join(results_dir, 'category_breakdown.png'))
    plt.close()

    # compute statistics on dates
    dataset['date'] = pd.to_datetime(dataset['date'], format='mixed')
    dataset['month'] = dataset['date'].dt.month
    dataset['year'] = dataset['date'].dt.year
    monthly_counts = dataset.groupby(['year', 'month', 'fake']).size().unstack(fill_value=0).reset_index()

    for year in dataset.year.unique().tolist():
        sns.barplot(data=monthly_counts[monthly_counts['year'] == year], x='month', y=1, color='green',
                    label=f'Fake ({year})')
        sns.barplot(data=monthly_counts[monthly_counts['year'] == year], x='month', y=0, color='orange',
                    label=f'True ({year})', bottom=monthly_counts[monthly_counts['year'] == year][1])

        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.title(f'Monthly Distribution of Fake and True News ({year})')
        plt.legend(title='News Type')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'monthly_distribution_{year}.png'))
        plt.close()


def investigate_topic_distribtion(dataset: pd.DataFrame, sample_size: int = 2500) -> pd.DataFrame:
    """
    A function which uses LDA to invesgiate themes within the dataset
    :param sample_size: sample used for topic modeller
    :param dataset: input dataset of news articles
    :return: distributions of topics within the dataset
    """
    # look at all data combined
    dataset = dataset.sample(sample_size)
    text = dataset['title'].tolist()
    text = [' '.join(t) for t in text]

    topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
    topic_model.fit(text)

    # look at the top 15 most interesting topics
    topic_model.visualize_barchart(top_n_topics=15).show()

    # look at fake and true data
    topic_info = pd.DataFrame(
        {
            'topics': topic_model.get_document_info(text)['Topic'],
            'labels': dataset['fake'].tolist()
        }
    )

    # look at the intersection between fake, true data in terms of topics
    intersection_count = sum(topic_info['labels'] & topic_info['topics'])

    # Calculate remaining counts for 0 and 1
    count_0 = sum(topic_info['labels'] == 0) - intersection_count
    count_1 = sum(topic_info['labels'] == 1) - intersection_count

    # Output results
    print("Intersection Count:", intersection_count)
    print("Remaining Count for 0:", count_0)
    print("Remaining Count for 1:", count_1)


def run_exploratory_analysis(config: dict):
    """
    A main function which runs all of the exploratory analysis
    :param config: configuration dictionary of exploratory analysis args
    :return: nothing.
    """
    dataset_cfg = config['dataset']

    # load the datasets and combine
    dataset_dir = os.path.join(config['src_dir'], 'data', 'raw')
    combined_dataset = utils.load_dataset_and_combine(
        true_data=os.path.join(dataset_dir, dataset_cfg['positive']),
        fake_data=os.path.join(dataset_dir, dataset_cfg['negative']),
    )

    # compute summary statistics on the samples
    results_dir = os.path.join(config['src_dir'], 'results', 'exploratory_data_analysis')
    compute_summary_stats(dataset=combined_dataset, results_dir=results_dir)

    # apply some preprocessing steps to perform more advanced EDA
    cleaned_data = utils.clean_dataset(dataset=combined_dataset)

    # apply topic modelling to understand themes within the dataset
    if config['dataset']['topic_modeller']: investigate_topic_distribtion(dataset=cleaned_data)


if __name__ == "__main__":
    src_dir = os.path.dirname(os.getcwd())
    experiment_config = utils.load_yaml_file(file_path=os.path.join(src_dir, "configuration"))
    experiment_config['src_dir'] = src_dir
    run_exploratory_analysis(config=experiment_config)
