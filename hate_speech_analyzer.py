#!/usr/bin/env python3
"""
Simple Hate Speech Analyzer for your dataset
Usage: python hate_speech_analyzer.py
"""

import os
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def load_hate_speech_model():
    """Load a hate speech detection model"""
    try:
        # Using a smaller, accessible model
        classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-hate-latest"
        )
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def analyze_dataset(csv_file="processed_dataset.csv"):
    """Analyze your hate speech dataset"""
    try:
        # Load your dataset
        df = pd.read_csv(csv_file)
        print(f"Loaded dataset with {len(df)} rows")
        
        # Load the classifier
        classifier = load_hate_speech_model()
        if not classifier:
            return
        
        # Analyze a sample of comments (to avoid rate limits)
        sample_size = min(50, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        print(f"Analyzing {sample_size} sample comments...")
        
        results = []
        for idx, row in sample_df.iterrows():
            if 'comment' in df.columns:
                text = row['comment']
            elif 'text' in df.columns:
                text = row['text']
            else:
                # Use first text column found
                text_cols = [col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()]
                if text_cols:
                    text = row[text_cols[0]]
                else:
                    continue
            
            try:
                result = classifier(text[:512])  # Limit text length
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'prediction': result[0]['label'],
                    'confidence': result[0]['score']
                })
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Display results
        print("\n=== Analysis Results ===")
        print(results_df.head(10))
        
        # Show distribution
        if len(results_df) > 0:
            print("\n=== Prediction Distribution ===")
            print(results_df['prediction'].value_counts())
            
            # Save results
            results_df.to_csv('hate_speech_analysis_results.csv', index=False)
            print("\nResults saved to 'hate_speech_analysis_results.csv'")
        
        return results_df
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return None

def main():
    """Main function"""
    print("=== Hate Speech Analysis Tool ===")
    
    # Check if dataset exists
    if os.path.exists('processed_dataset.csv'):
        print("Found processed_dataset.csv")
        results = analyze_dataset('processed_dataset.csv')
    elif os.path.exists('selected_comments.csv'):
        print("Found selected_comments.csv")
        results = analyze_dataset('selected_comments.csv')
    else:
        print("No dataset found. Please ensure you have:")
        print("- processed_dataset.csv, or")
        print("- selected_comments.csv")
        print("in the current directory.")
        
        # Demo with sample text
        print("\nRunning demo with sample texts...")
        classifier = load_hate_speech_model()
        if classifier:
            sample_texts = [
                "I love this community, everyone is so helpful!",
                "I disagree with this policy but respect different opinions.",
                "This is a test comment for analysis."
            ]
            
            for text in sample_texts:
                result = classifier(text)
                print(f"Text: {text}")
                print(f"Prediction: {result[0]['label']} (confidence: {result[0]['score']:.3f})")
                print()

if __name__ == "__main__":
    main()
