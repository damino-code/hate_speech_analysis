import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from download_dataset import download_dataset


def explore_dataset(dataset):
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)

    df = dataset['train'].to_pandas()

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of unique comments: {df['comment_id'].nunique()}")
    print(f"Number of unique annotators: {df['annotator_id'].nunique()}")

    print(f"\nColumns in the dataset:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")

    print(f"\nKey columns and their data types:")
    key_columns = ['hate_speech_score', 'sentiment', 'respect', 'insult', 'humiliate',
                   'status', 'dehumanize', 'violence', 'genocide', 'attack_defend', 'hatespeech']
    for col in key_columns:
        if col in df.columns:
            print(f"{col:20s}: {df[col].dtype} | Range: {df[col].min():.2f} to {df[col].max():.2f}")

    return df


def analyze_hate_speech_scores(df):
    print("\n" + "="*50)
    print("HATE SPEECH SCORE ANALYSIS")
    print("="*50)

    if 'hate_speech_score' not in df.columns:
        print("hate_speech_score column not found!")
        return

    scores = df['hate_speech_score']

    print(f"Mean hate speech score: {scores.mean():.3f}")
    print(f"Median hate speech score: {scores.median():.3f}")
    print(f"Standard deviation: {scores.std():.3f}")

    hate_speech = scores > 0.5
    counter_supportive = scores < -1
    neutral_ambiguous = (scores >= -1) & (scores <= 0.5)

    print(f"\nScore categorization:")
    print(f"Hate speech (> 0.5): {hate_speech.sum():,} ({hate_speech.mean()*100:.1f}%)")
    print(f"Counter/supportive speech (< -1): {counter_supportive.sum():,} ({counter_supportive.mean()*100:.1f}%)")
    print(f"Neutral/ambiguous (-1 to 0.5): {neutral_ambiguous.sum():,} ({neutral_ambiguous.mean()*100:.1f}%)")

    print(f"\nExample hate speech comments (score > 0.5):")
    for _, row in df[hate_speech].head(3).iterrows():
        print(f"Score: {row['hate_speech_score']:.2f} | Text: {row['text'][:100]}...")

    print(f"\nExample counter/supportive comments (score < -1):")
    for _, row in df[counter_supportive].head(3).iterrows():
        print(f"Score: {row['hate_speech_score']:.2f} | Text: {row['text'][:100]}...")


def analyze_target_groups(df):
    print("\n" + "="*50)
    print("TARGET IDENTITY GROUP ANALYSIS")
    print("="*50)

    target_columns = [col for col in df.columns if col.startswith('target_') and col.endswith('_bool')]
    print(f"Found {len(target_columns)} target identity group columns:")

    for col in target_columns:
        true_count = df[col].sum()
        percentage = (true_count / len(df)) * 100
        print(f"{col:30s}: {true_count:6,} ({percentage:5.1f}%)")


def analyze_annotator_demographics(df):
    print("\n" + "="*50)
    print("ANNOTATOR DEMOGRAPHICS")
    print("="*50)

    if 'annotator_gender' in df.columns:
        print(f"Annotator gender distribution:")
        for gender, count in df['annotator_gender'].value_counts().head(10).items():
            print(f"  {gender}: {count:,}")

    if 'annotator_educ' in df.columns:
        print(f"\nAnnotator education distribution:")
        for educ, count in df['annotator_educ'].value_counts().head(10).items():
            print(f"  {educ}: {count:,}")

    if 'annotator_age' in df.columns:
        age_stats = df['annotator_age'].describe()
        print(f"\nAnnotator age statistics:")
        print(f"  Mean: {age_stats['mean']:.1f}")
        print(f"  Median: {age_stats['50%']:.1f}")
        print(f"  Min: {age_stats['min']:.1f}")
        print(f"  Max: {age_stats['max']:.1f}")


def show_sample_comments(df, n_samples=5):
    print("\n" + "="*50)
    print(f"SAMPLE COMMENTS (showing {n_samples} random samples)")
    print("="*50)

    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    for _, row in sample_df.iterrows():
        print(f"\nComment ID: {row['comment_id']}")
        print(f"Hate Speech Score: {row['hate_speech_score']:.3f}")
        print(f"Text: {row['text']}")
        print("-" * 80)


def create_visualizations(df):
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Measuring Hate Speech Dataset Analysis', fontsize=16)

    if 'hate_speech_score' in df.columns:
        axes[0, 0].hist(df['hate_speech_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Hate Speech Threshold (>0.5)')
        axes[0, 0].axvline(x=-1, color='green', linestyle='--', label='Counter Speech Threshold (<-1)')
        axes[0, 0].set_xlabel('Hate Speech Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Hate Speech Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts()
        axes[0, 1].pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Distribution by Platform')

    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        axes[1, 0].bar(sentiment_counts.index, sentiment_counts.values, alpha=0.7, color='lightcoral')
        axes[1, 0].set_xlabel('Sentiment Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Sentiment Scores')
        axes[1, 0].grid(True, alpha=0.3)

    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        axes[1, 1].hist(text_lengths, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_xlabel('Text Length (characters)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Text Lengths')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hate_speech_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'hate_speech_analysis.png'")
    plt.show()


def main():
    dataset = download_dataset('default')
    if dataset is None:
        return

    df = explore_dataset(dataset)
    analyze_hate_speech_scores(df)
    analyze_target_groups(df)
    analyze_annotator_demographics(df)
    show_sample_comments(df, n_samples=3)
    create_visualizations(df)


if __name__ == "__main__":
    main()
