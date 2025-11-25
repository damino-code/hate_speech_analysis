#!/usr/bin/env python3
"""
Compare Demographic-Aware Hate Speech Analysis Results
Compares demographic perspective analysis with human annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os

def load_demographic_data():
    """Load demographic analysis results"""
    print("📊 LOADING DEMOGRAPHIC ANALYSIS DATA")
    print("=" * 50)
    
    # Find demographic analysis files
    demographic_files = list(Path(".").glob("demographic_hate_analysis_*.csv"))
    if not demographic_files:
        raise FileNotFoundError("No demographic analysis files found (demographic_hate_analysis_*.csv)")
    
    # Use the most recent file
    latest_file = max(demographic_files, key=lambda x: x.stat().st_mtime)
    print(f"🎯 Loading demographic results: {latest_file}")
    
    df_demographic = pd.read_csv(latest_file)
    print(f"   📈 {len(df_demographic)} demographic predictions")
    
    # Extract demographic profile from filename
    filename_parts = latest_file.stem.split('_')
    demographic_profile = None
    for i, part in enumerate(filename_parts):
        if part == "analysis":
            demographic_profile = '_'.join(filename_parts[i+1:-2])  # Skip timestamp parts
            break
    
    if not demographic_profile:
        demographic_profile = "unknown"
    
    return df_demographic, latest_file, demographic_profile

def load_and_filter_processed_data(comment_ids):
    """Load processed dataset and filter to matching comment IDs"""
    print(f"📋 Loading processed dataset: processed_dataset.csv")
    df_processed_full = pd.read_csv("processed_dataset.csv")
    df_processed_full['comment_id'] = df_processed_full['comment_id'].astype(str)
    
    comment_ids_str = [str(cid) for cid in comment_ids]
    
    # Filter to only the comments that were analyzed demographically
    df_filtered = df_processed_full[df_processed_full['comment_id'].isin(comment_ids_str)].copy()
    
    # Average multiple annotations per comment (since there are multiple annotators)
    df_processed = df_filtered.groupby('comment_id').agg({
        'hatespeech': 'mean',
        'hate_speech_score': 'mean', 
        'sentiment': 'mean',
        'respect': 'mean',
        'insult': 'mean',
        'text': 'first'  # Take the first text (should be same for all annotators)
    }).reset_index()
    
    print(f"   📈 Original dataset: {len(df_processed_full)} total annotations")
    print(f"   🎯 Filtered to: {len(df_filtered)} matching annotations")  
    print(f"   📊 Averaged to: {len(df_processed)} unique comments (one per comment_id)")
    
    return df_processed

def merge_demographic_and_processed(df_demographic, df_processed):
    """Merge demographic and processed datasets"""
    print("\n🔗 MERGING DATASETS")
    print("=" * 30)
    
    # Ensure comment_id is same type
    df_demographic['comment_id'] = df_demographic['comment_id'].astype(str)
    df_processed['comment_id'] = df_processed['comment_id'].astype(str)
    
    # Merge on comment_id
    df_merged = pd.merge(
        df_demographic[['comment_id', 'rating', 'category', 'confidence', 'demographic_profile', 'explanation', 'text']], 
        df_processed[['comment_id', 'hatespeech', 'hate_speech_score', 'sentiment', 'respect', 'insult']], 
        on='comment_id', 
        how='inner'
    )
    
    print(f"✅ Merged {len(df_merged)} comments with both demographic and human annotations")
    
    return df_merged

def convert_ratings_for_comparison(df_merged):
    """Convert ratings to consistent formats for comparison"""
    print("\n🔄 CONVERTING RATINGS FOR COMPARISON")
    print("=" * 40)
    
    # Convert demographic ratings to binary (0=no hate, 1=hate, treat unclear as no hate for binary comparison)
    df_merged['demographic_hate_binary'] = (df_merged['rating'] >= 2).astype(int)
    df_merged['demographic_hate_category'] = df_merged['category']
    
    # Convert human ratings to binary (threshold at 0.5)
    df_merged['human_hate_binary'] = (df_merged['hatespeech'] >= 0.5).astype(int) 
    df_merged['human_hate_category'] = df_merged['hatespeech'].apply(
        lambda x: 'hate' if x >= 0.5 else 'no_hate'
    )
    
    print("📊 Demographic ratings distribution:")
    print(df_merged['demographic_hate_category'].value_counts())
    
    print("\n📊 Human ratings distribution:")
    print(df_merged['human_hate_category'].value_counts())
    
    return df_merged

def calculate_comparison_metrics(df_merged):
    """Calculate comparison metrics"""
    print("\n📈 CALCULATING COMPARISON METRICS")
    print("=" * 40)
    
    # Binary classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_true = df_merged['human_hate_binary']
    y_pred = df_merged['demographic_hate_binary']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    print(f"✅ Accuracy:  {metrics['accuracy']:.3f}")
    print(f"✅ Precision: {metrics['precision']:.3f}")
    print(f"✅ Recall:    {metrics['recall']:.3f}")
    print(f"✅ F1 Score:  {metrics['f1']:.3f}")
    
    return metrics

def create_demographic_comparison_visualization(df_merged, demographic_profile, metrics):
    """Create visualization comparing demographic perspective with human annotations"""
    print("\n🎨 CREATING DEMOGRAPHIC COMPARISON VISUALIZATION")
    print("=" * 50)
    
    # Create output directory
    os.makedirs('demographic_comparison_outputs', exist_ok=True)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Rating Distribution Comparison
    ratings_data = []
    
    # Demographic ratings
    for rating in [0, 1, 2]:
        count = len(df_merged[df_merged['rating'] == rating])
        ratings_data.append({
            'Rating': f'{rating} ({["No Hate", "Unclear", "Hate"][rating]})',
            'Count': count,
            'Source': f'Demographic ({demographic_profile})'
        })
    
    # Human ratings (convert to 3-category for comparison)
    human_cats = []
    for _, row in df_merged.iterrows():
        if row['hatespeech'] < 0.3:
            human_cats.append(0)
        elif row['hatespeech'] < 0.7:
            human_cats.append(1)
        else:
            human_cats.append(2)
    
    for rating in [0, 1, 2]:
        count = human_cats.count(rating)
        ratings_data.append({
            'Rating': f'{rating} ({["No Hate", "Unclear", "Hate"][rating]})',
            'Count': count,
            'Source': 'Human Annotators'
        })
    
    # Create DataFrame and plot
    ratings_df = pd.DataFrame(ratings_data)
    
    sns.barplot(data=ratings_df, x='Rating', y='Count', hue='Source', ax=ax1)
    ax1.set_title(f'Rating Distribution Comparison\n{demographic_profile.replace("_", " ").title()} vs Human Annotators')
    ax1.set_ylabel('Number of Comments')
    ax1.legend()
    
    # Add count labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%d')
    
    # Plot 2: Confusion Matrix (Binary Classification)
    cm = metrics['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['No Hate', 'Hate'],
                yticklabels=['No Hate', 'Hate'])
    ax2.set_title(f'Confusion Matrix\n{demographic_profile.replace("_", " ").title()} vs Human (Binary)')
    ax2.set_xlabel('Demographic Prediction')
    ax2.set_ylabel('Human Annotation')
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            ax2.text(j + 0.3, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'demographic_comparison_outputs/demographic_comparison_{demographic_profile}_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved: {plot_filename}")
    return plot_filename

def show_detailed_comment_analysis(df_merged, demographic_profile):
    """Show detailed comment-by-comment comparison"""
    print(f"\n🔍 DETAILED COMMENT ANALYSIS ({demographic_profile})")
    print("=" * 50)
    
    # Sort by comment ID for easier reading
    df_sorted = df_merged.sort_values('comment_id')
    
    print(f"📊 Analyzing {len(df_sorted)} comments:")
    print(f"   Demographic ({demographic_profile}): 0-2 scale (0=No Hate, 1=Unclear, 2=Hate)")
    print("   Human: 0-1 scale (averaged across annotators)")
    print()
    
    # Count agreements and disagreements (using binary classification)
    agreements = df_sorted[df_sorted['demographic_hate_binary'] == df_sorted['human_hate_binary']]
    disagreements = df_sorted[df_sorted['demographic_hate_binary'] != df_sorted['human_hate_binary']]
    
    print(f"✅ Binary Agreements: {len(agreements)} ({len(agreements)/len(df_sorted)*100:.1f}%)")
    print(f"❌ Binary Disagreements: {len(disagreements)} ({len(disagreements)/len(df_sorted)*100:.1f}%)")
    
    # Show detailed breakdown
    print("\n📋 SAMPLE COMMENT BREAKDOWN:")
    print("ID".ljust(10) + "Demo".ljust(6) + "Human".ljust(8) + "Match".ljust(8) + "Text Preview")
    print("-" * 80)
    
    for _, row in df_sorted.head(15).iterrows():  # Show first 15 for readability
        comment_id = str(row['comment_id'])[:8]
        demo_rating = int(row['rating'])
        human_score = f"{row['hatespeech']:.2f}"
        match = "✓" if row['demographic_hate_binary'] == row['human_hate_binary'] else "✗"
        text_preview = row['text'][:45] + "..." if len(row['text']) > 45 else row['text']
        
        print(f"{comment_id:<10}{demo_rating:<6}{human_score:<8}{match:<8}{text_preview}")
    
    if len(df_sorted) > 15:
        print(f"... and {len(df_sorted) - 15} more comments")
    
    # Show specific disagreement examples
    if len(disagreements) > 0:
        print(f"\n🔴 DISAGREEMENT EXAMPLES:")
        for _, row in disagreements.head(3).iterrows():
            text_preview = row['text'][:70] + "..." if len(row['text']) > 70 else row['text']
            print(f"  ID {row['comment_id']}: Demographic={row['rating']}, Human={row['hatespeech']:.2f}")
            print(f"    Text: '{text_preview}'")
            if row['explanation']:
                explanation_preview = row['explanation'][:100] + "..." if len(row['explanation']) > 100 else row['explanation']
                print(f"    Reasoning: {explanation_preview}")
            print()

def save_demographic_comparison_results(df_merged, demographic_profile, metrics):
    """Save detailed comparison results"""
    print(f"\n💾 SAVING DETAILED RESULTS")
    print("=" * 30)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed comparison CSV
    comparison_filename = f"demographic_detailed_comparison_{demographic_profile}_{timestamp}.csv"
    df_merged.to_csv(comparison_filename, index=False)
    
    # Save summary report
    report_filename = f"demographic_comparison_report_{demographic_profile}_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write(f"DEMOGRAPHIC HATE SPEECH ANALYSIS COMPARISON REPORT\n")
        f.write(f"Profile: {demographic_profile}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"SUMMARY STATISTICS\n")
        f.write(f"-" * 20 + "\n")
        f.write(f"Total comments analyzed: {len(df_merged)}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n") 
        f.write(f"Recall: {metrics['recall']:.3f}\n")
        f.write(f"F1 Score: {metrics['f1']:.3f}\n\n")
        
        f.write(f"DEMOGRAPHIC RATING DISTRIBUTION\n")
        f.write(f"-" * 35 + "\n")
        f.write(str(df_merged['demographic_hate_category'].value_counts()))
        f.write(f"\n\n")
        
        f.write(f"HUMAN RATING DISTRIBUTION\n")
        f.write(f"-" * 30 + "\n")
        f.write(str(df_merged['human_hate_category'].value_counts()))
        f.write(f"\n")
    
    print(f"📄 Report saved: {report_filename}")
    print(f"📊 Detailed data saved: {comparison_filename}")
    
    return report_filename, comparison_filename

def main():
    """Main comparison function"""
    print("🎯 DEMOGRAPHIC HATE SPEECH COMPARISON")
    print("🧠 Demographic Perspective vs Human Annotations")
    print("=" * 60)
    
    try:
        # Load demographic analysis data
        df_demographic, demographic_file, demographic_profile = load_demographic_data()
        
        # Load and filter processed data to matching comment IDs
        df_processed = load_and_filter_processed_data(df_demographic['comment_id'])
        
        # Merge datasets
        df_merged = merge_demographic_and_processed(df_demographic, df_processed)
        
        # Convert ratings for comparison
        df_merged = convert_ratings_for_comparison(df_merged)
        
        # Calculate metrics
        metrics = calculate_comparison_metrics(df_merged)
        
        # Create visualization
        plot_file = create_demographic_comparison_visualization(df_merged, demographic_profile, metrics)
        
        # Show detailed analysis
        show_detailed_comment_analysis(df_merged, demographic_profile)
        
        # Save results
        report_file, comparison_file = save_demographic_comparison_results(df_merged, demographic_profile, metrics)
        
        print(f"\n🎉 DEMOGRAPHIC COMPARISON COMPLETE!")
        print("=" * 40)
        print(f"📊 Analyzed {len(df_merged)} comments")
        print(f"👤 Demographic Profile: {demographic_profile}")
        print(f"📈 Overall accuracy: {metrics['accuracy']:.1%}")
        print(f"🎨 Visualization: {plot_file}")
        print(f"📄 Report: {report_file}")
        print(f"📊 Data: {comparison_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        raise

if __name__ == "__main__":
    main()