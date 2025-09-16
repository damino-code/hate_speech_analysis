import datasets
import warnings
warnings.filterwarnings('ignore')


def download_dataset(config: str = 'default'):
    """Download the Measuring Hate Speech dataset and return the dataset dict.

    Args:
        config: Dataset configuration/subset. Use 'default' per dataset card.

    Returns:
        A datasets.DatasetDict with the available splits, or None on failure.
    """
    print("Downloading the Measuring Hate Speech dataset...")
    try:
        dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', config)
        print("Dataset loaded successfully!")
        print(f"Dataset splits: {list(dataset.keys())}")
        if 'train' in dataset:
            print(f"Train set size: {len(dataset['train'])} rows")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


if __name__ == "__main__":
    # CLI usage: simply download to verify access
    _ = download_dataset('default')
