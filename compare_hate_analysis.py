#!/usr/bin/env python3
"""
Hate Speech Analysis Comparison Tool
Compares llama-cpp analysis results with processed dataset annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load both datasets for comparison - only comments from llama-cpp analysis"""
    print("📊 LOADING DATA FOR COMPARISON")
    print("=" * 50)
    
    # Load llama-cpp results
    llama_files = list(Path(".").glob("llamacpp_hate_analysis_*.csv"))
    if not llama_files:
        raise FileNotFoundError("No llama-cpp analysis files found (llamacpp_hate_analysis_*.csv)")
    
    # Use the most recent file
    latest_llama_file = max(llama_files, key=lambda x: x.stat().st_mtime)
    print(f"🦙 Loading Llama-cpp results: {latest_llama_file}")
    
    df_llama = pd.read_csv(latest_llama_file)
    print(f"   📈 {len(df_llama)} llama-cpp predictions")
    
    # Get the comment IDs from llama-cpp analysis
    llama_comment_ids = df_llama['comment_id'].astype(str).tolist()
    print(f"   🎯 Targeting {len(llama_comment_ids)} specific comment IDs")
    
    # Load processed dataset and filter to only those comment IDs
    print(f"📋 Loading processed dataset: processed_dataset.csv")
    df_processed_full = pd.read_csv("processed_dataset.csv")
    df_processed_full['comment_id'] = df_processed_full['comment_id'].astype(str)
    
    # Filter to only the comments that were analyzed by llama-cpp
    df_filtered = df_processed_full[df_processed_full['comment_id'].isin(llama_comment_ids)].copy()
    
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
    
    return df_llama, df_processed, latest_llama_file

def merge_datasets(df_llama, df_processed):
    """Merge datasets on comment_id for comparison"""
    print("\n🔗 MERGING DATASETS")
    print("=" * 30)
    
    # Ensure comment_id is same type
    df_llama['comment_id'] = df_llama['comment_id'].astype(str)
    df_processed['comment_id'] = df_processed['comment_id'].astype(str)
    
    # Merge on comment_id
    df_merged = pd.merge(
        df_llama[['comment_id', 'rating', 'category', 'confidence', 'method', 'text']], 
        df_processed[['comment_id', 'hatespeech', 'hate_speech_score', 'sentiment', 'respect', 'insult']], 
        on='comment_id', 
        how='inner'
    )
    
    print(f"✅ Merged {len(df_merged)} comments with both predictions and annotations")
    print(f"📊 Coverage: {len(df_merged)/len(df_llama)*100:.1f}% of llama-cpp predictions have annotations")
    
    return df_merged

def convert_ratings(df_merged):
    """Convert ratings to consistent formats for comparison"""
    print("\n🔄 CONVERTING RATINGS")
    print("=" * 25)
    
    # Llama-cpp uses: 2.0=hate, 1.0=unclear, 0.0=no_hate
    # Processed uses: hatespeech (0.0-1.0), hate_speech_score (-∞ to +∞)
    
    # Convert llama ratings to binary
    df_merged['llama_hate_binary'] = (df_merged['rating'] >= 2.0).astype(int)
    df_merged['llama_hate_category'] = df_merged['rating'].map({
        2.0: 'hate', 1.0: 'unclear', 0.0: 'no_hate'
    })
    
    # Convert processed ratings to binary (threshold at 0.5)
    df_merged['processed_hate_binary'] = (df_merged['hatespeech'] >= 0.5).astype(int)
    df_merged['processed_hate_category'] = df_merged['hatespeech'].apply(
        lambda x: 'hate' if x >= 0.5 else 'no_hate'
    )
    
    print(f"📊 Llama-cpp ratings distribution:")
    print(df_merged['llama_hate_category'].value_counts())
    print(f"\n📊 Processed ratings distribution:")
    print(df_merged['processed_hate_category'].value_counts())
    
    return df_merged

def calculate_metrics(df_merged):
    """Calculate comparison metrics"""
    print("\n📈 CALCULATING METRICS")
    print("=" * 25)
    
    # Confusion matrix values
    tp = len(df_merged[(df_merged['llama_hate_binary'] == 1) & (df_merged['processed_hate_binary'] == 1)])
    tn = len(df_merged[(df_merged['llama_hate_binary'] == 0) & (df_merged['processed_hate_binary'] == 0)])
    fp = len(df_merged[(df_merged['llama_hate_binary'] == 1) & (df_merged['processed_hate_binary'] == 0)])
    fn = len(df_merged[(df_merged['llama_hate_binary'] == 0) & (df_merged['processed_hate_binary'] == 1)])
    
    # Calculate metrics
    accuracy = (tp + tn) / len(df_merged) if len(df_merged) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'total_samples': len(df_merged)
    }
    
    print(f"✅ Accuracy:  {accuracy:.3f}")
    print(f"✅ Precision: {precision:.3f}")
    print(f"✅ Recall:    {recall:.3f}")
    print(f"✅ F1 Score:  {f1:.3f}")
    
    return metrics

def create_visualizations(df_merged, metrics, output_file):
    """Create focused comparison visualizations"""
    print("\n🎨 CREATING FOCUSED VISUALIZATIONS")
    print("=" * 35)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Rating Distribution Comparison
    # Create bins for rating values
    bins = np.linspace(0, 2, 21)  # 20 bins from 0 to 2
    
    # Plot human ratings distribution (hatespeech attribute 0-1, scaled to 0-2)
    human_scaled = df_merged['hatespeech'] * 2  # Scale 0-1 to 0-2 for comparison
    ax1.hist(human_scaled, bins=bins, alpha=0.6, color='blue', 
             label=f'Human (scaled 0-2)\nn={len(df_merged)}', edgecolor='black', linewidth=0.5)
    
    # Plot llama ratings distribution (rating attribute 0-2)
    ax1.hist(df_merged['rating'], bins=bins, alpha=0.6, color='red', 
             label=f'Llama-cpp (0-2)\nn={len(df_merged)}', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Rating Value')
    ax1.set_ylabel('Number of Comments')
    ax1.set_title(f'Rating Distribution Comparison ({len(df_merged)} comments)')
    ax1.set_xlim(0, 2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add vertical lines for key thresholds
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Threshold (1.0)')    # 2. Confusion Matrix - Just count the 50 comments by categories
    # Convert human hatespeech (0-1) to binary: <0.5=No Hate, >=0.5=Hate
    human_binary = (df_merged['hatespeech'] >= 0.5).astype(int)
    # Convert llama rating (0-2) to binary: <1=No Hate, >=1=Hate  
    llama_binary = (df_merged['rating'] >= 1).astype(int)
    
    # Create 2x2 confusion matrix counting the comments
    confusion_matrix = np.zeros((2, 2))
    for h, l in zip(human_binary, llama_binary):
        confusion_matrix[int(h), int(l)] += 1
    
    # Plot confusion matrix
    im = ax2.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    
    # Add text annotations showing comment counts
    for i in range(2):
        for j in range(2):
            count = int(confusion_matrix[i, j])
            text = ax2.text(j, i, f'{count}\ncomments',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['No Hate', 'Hate'])
    ax2.set_yticklabels(['No Hate', 'Hate'])
    ax2.set_xlabel('Llama-cpp Prediction')
    ax2.set_ylabel('Human Annotation')
    ax2.set_title(f'Confusion Matrix\n(Total: {len(df_merged)} comments)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Number of Comments')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"focused_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved: {plot_filename}")
    
    plt.show()
    
    return plot_filename

def show_comment_by_comment_analysis(df_merged):
    """Show detailed comment-by-comment comparison with IDs"""
    print("\n🔍 COMMENT-BY-COMMENT ANALYSIS")
    print("=" * 35)
    
    # Sort by comment ID for easier reading
    df_sorted = df_merged.sort_values('comment_id')
    
    print(f"📊 Analyzing {len(df_sorted)} comments with ratings:")
    print("   Human: 0-1 scale (converted to 0=No Hate, ≥0.5=Hate)")
    print("   Llama: 0-2 scale (0=No Hate, 1=Unclear, 2=Hate)")
    print()
    
    # Convert human scores for easier comparison
    df_sorted = df_sorted.copy()
    df_sorted['human_rating'] = df_sorted['hatespeech'].apply(lambda x: 0 if x < 0.5 else 2)
    
    # Count agreements and disagreements
    agreements = df_sorted[df_sorted['human_rating'] == df_sorted['rating']]
    disagreements = df_sorted[df_sorted['human_rating'] != df_sorted['rating']]
    
    print(f"✅ Agreements: {len(agreements)} ({len(agreements)/len(df_sorted)*100:.1f}%)")
    print(f"❌ Disagreements: {len(disagreements)} ({len(disagreements)/len(df_sorted)*100:.1f}%)")
    
    # Show detailed breakdown
    print("\n� DETAILED BREAKDOWN BY COMMENT ID:")
    print("ID".ljust(8) + "Human".ljust(8) + "Llama".ljust(8) + "Match".ljust(8) + "Text Preview")
    print("-" * 70)
    
    for _, row in df_sorted.head(20).iterrows():  # Show first 20 for readability
        comment_id = str(row['comment_id'])[:6]
        human_score = f"{row['hatespeech']:.2f}"
        human_rating = int(row['human_rating'])
        llama_rating = int(row['rating'])
        match = "✓" if human_rating == llama_rating else "✗"
        text_preview = row['text'][:40] + "..." if len(row['text']) > 40 else row['text']
        
        print(f"{comment_id:<8}{human_score:<8}{llama_rating:<8}{match:<8}{text_preview}")
    
    if len(df_sorted) > 20:
        print(f"... and {len(df_sorted) - 20} more comments")
    
    # Show specific disagreement examples with comment IDs
    if len(disagreements) > 0:
        print(f"\n🔴 DISAGREEMENT EXAMPLES:")
        for _, row in disagreements.head(5).iterrows():
            text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
            print(f"  ID {row['comment_id']}: Human={row['hatespeech']:.2f}, Llama={row['rating']}")
            print(f"    Text: '{text_preview}'")
            print()

def save_detailed_results(df_merged, metrics, output_file):
    """Save detailed comparison results"""
    print("\n💾 SAVING DETAILED RESULTS")
    print("=" * 30)
    
    # Create summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"hate_analysis_comparison_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("HATE SPEECH ANALYSIS COMPARISON REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Llama-cpp file: {output_file}\n")
        f.write(f"Dataset: processed_dataset.csv\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 20 + "\n")
        for metric, value in metrics.items():
            if metric not in ['total_samples']:
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
        
        f.write(f"\nSAMPLE SIZE: {metrics['total_samples']} comments\n")
        
        f.write("\nRATING DISTRIBUTIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("Llama-cpp predictions:\n")
        f.write(str(df_merged['llama_hate_category'].value_counts()))
        f.write("\n\nHuman annotations:\n")
        f.write(str(df_merged['processed_hate_category'].value_counts()))
        f.write("\n")
    
    # Save detailed comparison CSV
    comparison_filename = f"detailed_comparison_{timestamp}.csv"
    df_merged.to_csv(comparison_filename, index=False)
    
    print(f"📄 Report saved: {report_filename}")
    print(f"📊 Detailed data saved: {comparison_filename}")
    
    return report_filename, comparison_filename

def main():
    """Main comparison function"""
    print("🎯 HATE SPEECH ANALYSIS COMPARISON")
    print("🦙 Llama-cpp vs Human Annotations")
    print("=" * 60)
    
    try:
        # Load data
        df_llama, df_processed, llama_file = load_data()
        
        # Merge datasets
        df_merged = merge_datasets(df_llama, df_processed)
        
        # Convert ratings
        df_merged = convert_ratings(df_merged)
        
        # Calculate metrics
        metrics = calculate_metrics(df_merged)
        
        # Create visualizations
        plot_file = create_visualizations(df_merged, metrics, llama_file)
        
        # Show comment-by-comment analysis
        show_comment_by_comment_analysis(df_merged)
        
        # Save results
        report_file, comparison_file = save_detailed_results(df_merged, metrics, llama_file)
        
        print(f"\n🎉 COMPARISON COMPLETE!")
        print("=" * 25)
        print(f"📊 Analyzed {len(df_merged)} comments")
        print(f"📈 Overall accuracy: {metrics['accuracy']:.1%}")
        print(f"🎨 Visualization: {plot_file}")
        print(f"📄 Report: {report_file}")
        print(f"📊 Data: {comparison_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()