import pandas as pd
import matplotlib.pyplot as plt
from download_dataset import download_dataset
from classifcation_changes import combine_columns

def main():
    dataset = download_dataset('default')
    if dataset is None:
        print('Dataset download failed.')
        return
    df = dataset['train'].to_pandas()
    df = combine_columns(df)

    # Columns to analyze
    columns = [col for col in df.columns if col.startswith('target_') or col.startswith('annotator_')]

    for col in columns:
        # Skip boolean columns
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        print(f'\nAnalysis for {col}:')
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column (e.g., age)
            mean = df[col].mean()
            var = df[col].var()
            print(f'Mean: {mean:.2f}, Variance: {var:.2f}')
            plt.figure()
            df[col].hist(bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{col}_hist.png')
        else:
            # Categorical column
            value_counts = df[col].value_counts(dropna=True, normalize=True) * 100
            print(value_counts)
            plt.figure()
            value_counts.plot(kind='bar')
            plt.title(f'Percentage Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Percentage')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{col}_bar.png')
    print('\nPlots saved as *_bar.png and *_hist.png')

if __name__ == "__main__":
    main()
