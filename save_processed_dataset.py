import pandas as pd
from download_dataset import download_dataset
from classifcation_changes import combine_columns

def main():
    dataset = download_dataset('default')
    if dataset is None:
        print('Dataset download failed.')
        return
    df = dataset['train'].to_pandas()
    df = combine_columns(df)
    df.to_csv('processed_dataset.csv', index=False)
    print('Processed dataset saved as processed_dataset.csv')

if __name__ == "__main__":
    main()
