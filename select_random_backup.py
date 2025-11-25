import pandas as pd
import numpy as np

# Select 200 comments with highest variance from processed_dataset.csv

def load_processed_dataset():
    """Load the processed dataset"""
    print("📚 Loading processed_dataset.csv...")
    try:
        df = pd.read_csv('processed_dataset.csv')
        print(f"✅ Loaded dataset with {len(df)} rows")
        return df
    except FileNotFoundError:
        print("❌ processed_dataset.csv not found!")
        print("Make sure the file exists in the current directory.")
        return None

def calculate_variance_score(df):
    """Calculate variance score for each comment across annotator demographic columns"""
    print("📊 Calculating variance scores across annotator demographics...")
    
    # Annotator demographic columns to calculate variance across
    variance_columns = [
        'annotator_gender', 'annotator_ideology', 'annotator_age', 
        'annotator_race', 'annotator_religion', 'annotator_sexuality'
    ]
    
    # Check which columns exist in the dataset
    existing_columns = [col for col in variance_columns if col in df.columns]
    missing_columns = [col for col in variance_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️  Missing columns: {missing_columns}")
    
    if not existing_columns:
        print("❌ No annotator demographic columns found!")
        return df
    
    print(f"✅ Using columns for variance: {existing_columns}")
    
    # For categorical/string columns, we need to convert to numeric first
    df_numeric = df.copy()
    
    for col in existing_columns:
        if df[col].dtype == 'object':
            # Convert categorical to numeric codes
            df_numeric[col + '_numeric'] = pd.Categorical(df[col]).codes
            existing_columns[existing_columns.index(col)] = col + '_numeric'
    
    # Calculate variance for each comment across annotator demographic columns
    variance_scores = df_numeric[existing_columns].var(axis=1)
    df['variance_score'] = variance_scores
    
    print(f"✅ Variance scores calculated. Range: {variance_scores.min():.3f} to {variance_scores.max():.3f}")
    return df

def select_high_variance_comments(df, n=200):
    """Select n comments with highest variance, ensuring no duplicate comment_ids"""
    print(f"\n🎯 Selecting {n} comments with highest variance...")
    
    # First, remove duplicates by comment_id, keeping the one with highest variance
    print(f"📋 Original dataset: {len(df)} rows")
    
    # Group by comment_id and keep the row with highest variance for each comment
    df_unique = df.loc[df.groupby('comment_id')['variance_score'].idxmax()]
    print(f"📊 Unique comments: {len(df_unique)} (after removing duplicate IDs)")
    
    # Sort by variance score (highest first) and select top n
    df_sorted = df_unique.sort_values('variance_score', ascending=False)
    
    if len(df_sorted) < n:
        print(f"⚠️  Only {len(df_sorted)} unique comments available, using all")
        selected = df_sorted
    else:
        selected = df_sorted.head(n)
    
    print(f"✅ Selected {len(selected)} comments")
    print(f"📈 Variance range in selection: {selected['variance_score'].min():.3f} to {selected['variance_score'].max():.3f}")
    
    return selected

def main():
    # Load processed dataset
    df = load_processed_dataset()
    if df is None:
        return
    
    print(f"📊 Dataset columns: {list(df.columns)}")
    print(f"� Dataset shape: {df.shape}")

    # Calculate variance scores for each comment
    df = calculate_variance_score(df)
    
    # Select 200 comments with highest variance (no duplicate IDs)
    sample = select_high_variance_comments(df, n=200)
    
    # Define columns to include in output
    available_columns = [
        'comment_id', 'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide',
        'attack_defend', 'hatespeech', 'hate_speech_score', 'variance_score',
        'annotator_race', 'annotator_severity', 'annotator_age', 'annotator_gender'
    ]
    
    # Only include columns that exist in the dataset
    columns = [col for col in available_columns if col in df.columns]
    missing_columns = [col for col in available_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️  Missing columns: {missing_columns}")
    
    result = sample[columns].copy()
    if 'text' in sample.columns:
        result['comment_text'] = sample['text']
    else:
        print("⚠️  'text' column not found in dataset")

    # Print summary and save to file
    print(f"\n📋 SELECTION SUMMARY:")
    print(f"Total comments selected: {len(result)}")
    print(f"Unique comment IDs: {result['comment_id'].nunique()}")
    print(f"Average variance score: {result['variance_score'].mean():.3f}")
    print(f"Variance distribution:")
    print(f"  - Min: {result['variance_score'].min():.3f}")
    print(f"  - Max: {result['variance_score'].max():.3f}")
    print(f"  - Median: {result['variance_score'].median():.3f}")
    
    # Show top 5 highest variance comments
    print(f"\n🔥 TOP 5 HIGHEST VARIANCE COMMENTS:")
    for i, row in result.head(5).iterrows():
        if 'comment_text' in result.columns:
            text_preview = row['comment_text'][:60] + "..." if len(row['comment_text']) > 60 else row['comment_text']
        else:
            text_preview = "No text available"
        print(f"  {row['comment_id']}: Variance={row['variance_score']:.3f} - '{text_preview}'")
    
    result.to_csv('selected_comments.csv', index=False)
    print(f'\n💾 Results saved to selected_comments.csv')

if __name__ == "__main__":
    main()
