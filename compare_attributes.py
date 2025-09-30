import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSV files
annotator_df = pd.read_csv('comparison_annotator.csv')
chatgpt5_df = pd.read_csv('comparison_chatgpt5.csv')

# Strip spaces from column names
annotator_df.columns = annotator_df.columns.str.strip()
chatgpt5_df.columns = chatgpt5_df.columns.str.strip()

attributes = [
    'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize',
    'violence', 'genocide', 'attack_defend', 'hatespeech', 'hate_speech_score'
]

# Ensure only the required columns are used
annotator_df = annotator_df[['comment_id'] + attributes]
chatgpt5_df = chatgpt5_df[['comment_id'] + attributes]

# Merge on comment_id
merged = pd.merge(annotator_df, chatgpt5_df, on='comment_id', suffixes=('_annotator', '_chatgpt5'))

# Create output directory if it doesn't exist
output_dir = 'comparison_attributes'
os.makedirs(output_dir, exist_ok=True)

# Plot comparison for each attribute
for attr in attributes:
    plt.figure(figsize=(8,4))
    # Discrete x values (comment_id) and y values (discrete categories)
    x = merged['comment_id'].astype(str)
    y1 = merged[f'{attr}_annotator']
    y2 = merged[f'{attr}_chatgpt5']
    plt.scatter(x, y1, label='Annotator', marker='o', color='blue')
    plt.scatter(x, y2, label='ChatGPT-5', marker='x', color='orange')
    plt.title(f'Comparison for {attr}')
    plt.xlabel('comment_id')
    plt.ylabel(attr)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{attr}.png'))
    plt.close()

print('Comparison plots saved in comparison_attributes/')
