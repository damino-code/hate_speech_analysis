#!/usr/bin/env python3
"""
Compare LLaMA-CPP analysis results with original processed dataset annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def load_and_compare_data():
    """Load both datasets and prepare for comparison"""
    
    print("📊 LOADING DATA FOR COMPARISON")
    print("=" * 50)
    
    # Load our analysis results
    print("🔍 Loading LLaMA-CPP analysis results...")
    analysis_df = pd.read_csv('llamacpp_hate_analysis_20251028_155930.csv')
    print(f"   • Analysis records: {len(analysis_df)}")
    
    # Load processed dataset (sample approach for large file)
    print("🔍 Loading processed dataset...")
    try:
        # For large files, we'll load only the needed columns and filter by comment IDs
        needed_columns = ['comment_id', 'hatespeech', 'hate_speech_score', 'text', 'sentiment', 
                         'respect', 'insult', 'humiliate', 'dehumanize', 'violence']
        
        # Get unique comment IDs from our analysis
        analysis_comment_ids = set(analysis_df['comment_id'].astype(str))
        print(f"   • Comment IDs to match: {len(analysis_comment_ids)}")
        
        # Load processed dataset in chunks to handle large size
        chunk_list = []
        chunk_size = 10000
        
        for chunk in pd.read_csv('processed_dataset.csv', chunksize=chunk_size, usecols=needed_columns):
            # Filter chunk for our comment IDs
            chunk['comment_id'] = chunk['comment_id'].astype(str)
            matching_chunk = chunk[chunk['comment_id'].isin(analysis_comment_ids)]
            if len(matching_chunk) > 0:
                chunk_list.append(matching_chunk)
        
        processed_df = pd.concat(chunk_list, ignore_index=True) if chunk_list else pd.DataFrame()
        print(f"   • Processed records found: {len(processed_df)}")
        
    except Exception as e:
        print(f"❌ Error loading processed dataset: {e}")
        return None, None
    
    return analysis_df, processed_df

def align_datasets(analysis_df, processed_df):
    """Align datasets for comparison and identify structural issues"""
    
    print("\n🔄 ALIGNING DATASETS")
    print("-" * 30)
    
    # Convert comment_id to string for consistent matching
    analysis_df['comment_id'] = analysis_df['comment_id'].astype(str)
    processed_df['comment_id'] = processed_df['comment_id'].astype(str)
    
    # Check for duplicates in processed dataset (multiple annotators per comment)
    processed_grouped = processed_df.groupby('comment_id').agg({
        'hatespeech': 'mean',  # Average hate speech rating
        'hate_speech_score': 'mean',  # Average hate speech score
        'text': 'first',  # Take first text (should be same)
        'sentiment': 'mean',
        'respect': 'mean',
        'insult': 'mean',
        'humiliate': 'mean',
        'dehumanize': 'mean',
        'violence': 'mean'
    }).reset_index()
    
    print(f"📋 Original processed records: {len(processed_df)}")
    print(f"📋 Unique comments after grouping: {len(processed_grouped)}")
    
    # Merge datasets
    merged_df = analysis_df.merge(
        processed_grouped, 
        on='comment_id', 
        how='inner',
        suffixes=('_llama', '_human')
    )
    
    print(f"📋 Successfully merged records: {len(merged_df)}")
    
    # Identify missing records
    analysis_ids = set(analysis_df['comment_id'])
    processed_ids = set(processed_grouped['comment_id'])
    
    missing_in_processed = analysis_ids - processed_ids
    missing_in_analysis = processed_ids - analysis_ids
    
    if missing_in_processed:
        print(f"⚠️  Comments in analysis but not in processed: {len(missing_in_processed)}")
    if missing_in_analysis:
        print(f"⚠️  Comments in processed but not in analysis: {len(missing_in_analysis)}")
    
    return merged_df

def analyze_rating_comparison(merged_df):
    """Compare LLaMA ratings with human annotations"""
    
    print(f"\n📊 RATING COMPARISON ANALYSIS")
    print("-" * 40)
    
    # Convert LLaMA ratings to match human scale if needed
    # Human: hatespeech (0-4), hate_speech_score (continuous)
    # LLaMA: rating (0.0-2.0)
    
    # Scale LLaMA ratings to 0-4 range for direct comparison with hatespeech
    merged_df['rating_scaled'] = merged_df['rating'] * 2  # 0-2 -> 0-4
    
    # Basic statistics
    print(f"📈 HUMAN ANNOTATIONS:")
    print(f"   • Hate speech (0-4): mean={merged_df['hatespeech'].mean():.2f}, std={merged_df['hatespeech'].std():.2f}")
    print(f"   • Hate score: mean={merged_df['hate_speech_score'].mean():.2f}, std={merged_df['hate_speech_score'].std():.2f}")
    
    print(f"\n🤖 LLAMA-CPP PREDICTIONS:")
    print(f"   • Rating (0-2): mean={merged_df['rating'].mean():.2f}, std={merged_df['rating'].std():.2f}")
    print(f"   • Rating scaled (0-4): mean={merged_df['rating_scaled'].mean():.2f}, std={merged_df['rating_scaled'].std():.2f}")
    print(f"   • Confidence: mean={merged_df['confidence'].mean():.2f}, std={merged_df['confidence'].std():.2f}")
    
    # Correlation analysis
    print(f"\n🔬 CORRELATION ANALYSIS:")
    
    # LLaMA rating vs Human hatespeech
    corr1, p1 = pearsonr(merged_df['rating'], merged_df['hatespeech'])
    print(f"   • LLaMA rating vs Human hatespeech: r={corr1:.3f}, p={p1:.3e}")
    
    # LLaMA rating vs Human hate_speech_score  
    corr2, p2 = pearsonr(merged_df['rating'], merged_df['hate_speech_score'])
    print(f"   • LLaMA rating vs Human hate_score: r={corr2:.3f}, p={p2:.3e}")
    
    # Scaled comparison
    corr3, p3 = pearsonr(merged_df['rating_scaled'], merged_df['hatespeech'])
    print(f"   • LLaMA scaled vs Human hatespeech: r={corr3:.3f}, p={p3:.3e}")
    
    return merged_df

def create_comparison_visualization(merged_df):
    """Create visualizations comparing predictions"""
    
    print(f"\n🎨 CREATING COMPARISON VISUALIZATIONS")
    print("-" * 40)
    
    # Set up the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LLaMA-CPP vs Human Annotations Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: LLaMA rating vs Human hatespeech
    ax1.scatter(merged_df['hatespeech'], merged_df['rating'], alpha=0.6, s=30)
    ax1.set_xlabel('Human Hate Speech Rating (0-4)')
    ax1.set_ylabel('LLaMA Rating (0-2)')
    ax1.set_title('LLaMA vs Human Ratings')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation line
    z = np.polyfit(merged_df['hatespeech'], merged_df['rating'], 1)
    p = np.poly1d(z)
    ax1.plot(merged_df['hatespeech'], p(merged_df['hatespeech']), "r--", alpha=0.8)
    
    # Add correlation text
    corr, _ = pearsonr(merged_df['hatespeech'], merged_df['rating'])
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # Plot 2: Distribution comparison
    ax2.hist(merged_df['hatespeech'], bins=20, alpha=0.5, label='Human', density=True)
    ax2.hist(merged_df['rating_scaled'], bins=20, alpha=0.5, label='LLaMA (scaled)', density=True)
    ax2.set_xlabel('Rating')
    ax2.set_ylabel('Density')
    ax2.set_title('Rating Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confidence vs Agreement
    # Calculate agreement as absolute difference
    merged_df['agreement'] = 1 - abs(merged_df['rating_scaled'] - merged_df['hatespeech']) / 4
    ax3.scatter(merged_df['confidence'], merged_df['agreement'], alpha=0.6, s=30)
    ax3.set_xlabel('LLaMA Confidence')
    ax3.set_ylabel('Agreement with Humans (0-1)')
    ax3.set_title('Confidence vs Agreement')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Category breakdown
    # Create category mapping for humans (0-4 -> categories)
    def human_to_category(rating):
        if rating <= 1:
            return 'No Hate'
        elif rating <= 2.5:
            return 'Unclear'
        else:
            return 'Hate Speech'
    
    merged_df['human_category'] = merged_df['hatespeech'].apply(human_to_category)
    merged_df['llama_category'] = merged_df['category'].map({
        'no': 'No Hate',
        'unclear': 'Unclear', 
        'yes': 'Hate Speech'
    })
    
    # Create confusion matrix
    confusion_data = pd.crosstab(merged_df['human_category'], merged_df['llama_category'])
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=ax4)
    ax4.set_title('Confusion Matrix: Human vs LLaMA Categories')
    ax4.set_xlabel('LLaMA Categories')
    ax4.set_ylabel('Human Categories')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'llama_vs_human_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {plot_file}")
    plt.show()
    
    return plot_file

def identify_structural_issues(analysis_df, processed_df):
    """Identify any structural issues that might need fixing"""
    
    print(f"\n🔍 STRUCTURAL ANALYSIS")
    print("-" * 30)
    
    issues = []
    
    # Check column names
    analysis_cols = set(analysis_df.columns)
    expected_cols = {'rating', 'category', 'confidence', 'method', 'response', 'text', 'comment_id'}
    
    missing_cols = expected_cols - analysis_cols
    extra_cols = analysis_cols - expected_cols
    
    if missing_cols:
        issues.append(f"Missing columns in analysis: {missing_cols}")
    if extra_cols:
        print(f"ℹ️  Extra columns in analysis: {extra_cols}")
    
    # Check data types
    if analysis_df['comment_id'].dtype != 'object':
        issues.append("comment_id should be string/object type for matching")
    
    # Check rating range
    rating_range = (analysis_df['rating'].min(), analysis_df['rating'].max())
    if rating_range[0] < 0 or rating_range[1] > 2:
        issues.append(f"Rating values outside expected range 0-2: {rating_range}")
    
    # Check for missing values
    missing_counts = analysis_df.isnull().sum()
    if missing_counts.sum() > 0:
        issues.append(f"Missing values found: {missing_counts.to_dict()}")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
        
        print(f"\n💡 SUGGESTED FIXES:")
        print(f"   • Ensure comment_id is string type: df['comment_id'] = df['comment_id'].astype(str)")
        print(f"   • Check rating bounds: df['rating'] = df['rating'].clip(0, 2)")
        print(f"   • Handle missing values appropriately")
        
        return False
    else:
        print("✅ No structural issues found!")
        return True

def main():
    """Main comparison function"""
    
    print("🔍 LLAMA-CPP vs HUMAN ANNOTATIONS COMPARISON")
    print("=" * 60)
    
    # Load data
    analysis_df, processed_df = load_and_compare_data()
    
    if analysis_df is None or processed_df is None:
        print("❌ Failed to load data for comparison")
        return
    
    if len(processed_df) == 0:
        print("❌ No matching records found between datasets")
        return
    
    # Check structural issues
    structural_ok = identify_structural_issues(analysis_df, processed_df)
    
    # Align datasets
    merged_df = align_datasets(analysis_df, processed_df)
    
    if len(merged_df) == 0:
        print("❌ No records could be merged for comparison")
        return
    
    # Perform comparison analysis
    merged_df = analyze_rating_comparison(merged_df)
    
    # Create visualizations
    plot_file = create_comparison_visualization(merged_df)
    
    # Save detailed comparison results
    comparison_file = 'llama_human_comparison_detailed.csv'
    merged_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Detailed comparison saved: {comparison_file}")
    
    # Summary
    print(f"\n📋 COMPARISON SUMMARY")
    print("=" * 30)
    print(f"• Total records compared: {len(merged_df)}")
    print(f"• LLaMA mean rating: {merged_df['rating'].mean():.2f}")
    print(f"• Human mean hate rating: {merged_df['hatespeech'].mean():.2f}")
    corr, _ = pearsonr(merged_df['rating'], merged_df['hatespeech'])
    print(f"• Correlation: {corr:.3f}")
    
    if not structural_ok:
        print(f"\n⚠️  RECOMMENDATION: Fix structural issues in analysis file for better comparison")
        return False
    
    print(f"\n🎉 Comparison completed successfully!")
    return True

if __name__ == "__main__":
    main()