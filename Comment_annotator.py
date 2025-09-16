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

    print(f'Total unique comment_id: {df["comment_id"].nunique()}')
    print(f'Total unique annotator_id: {df["annotator_id"].nunique()}')

    # Number of annotators per comment
    annotators_per_comment = df.groupby('comment_id')['annotator_id'].nunique()
    plt.figure()
    annotators_per_comment.hist(bins=30)
    plt.title('Number of Annotators per Comment')
    plt.xlabel('Annotators per Comment')
    plt.ylabel('Number of Comments')
    plt.grid(True, alpha=0.3)
    plt.savefig('annotators_per_comment.png')
    print('Saved: annotators_per_comment.png')

    # Number of comments per annotator
    comments_per_annotator = df.groupby('annotator_id')['comment_id'].nunique()
    plt.figure()
    comments_per_annotator.hist(bins=30)
    plt.title('Number of Comments per Annotator')
    plt.xlabel('Comments per Annotator')
    plt.ylabel('Number of Annotators')
    plt.grid(True, alpha=0.3)
    plt.savefig('comments_per_annotator.png')
    print('Saved: comments_per_annotator.png')

if __name__ == "__main__":
    main()
