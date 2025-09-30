import pandas as pd
from download_dataset import download_dataset
from classifcation_changes import combine_columns

# Select 10 random comments and show specified columns

def main():
    dataset = download_dataset('default')
    if dataset is None:
        print('Dataset download failed.')
        return
    df = dataset['train'].to_pandas()
    df = combine_columns(df)

    # Select 10 random comments regardless of annotator gender uniqueness
    sample = df.sample(n=10, random_state=42)
    columns = [
        'comment_id', 'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide',
        'attack_defend', 'hatespeech', 'hate_speech_score',
        'annotator_race', 'annotator_severity', 'annotator_age', 'annotator_gender'
    ]
    result = sample[columns].copy()
    result['comment_text'] = sample['text']

    # Print and save to file
    print(result)
    result.to_csv('selected_comments.csv', index=False)
    print('\nResults saved to selected_comments.csv')

if __name__ == "__main__":
    main()
