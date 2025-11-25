#!/usr/bin/env python3
"""
LLaMA vs Human Annotations Evaluation - Rating Distribution and F1 Score
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.stats import pearsonr

def load_comparison_data():
    """Load and align LLaMA analysis with human annotations"""
    
    print("📊 LOADING DATA FOR EVALUATION")
    print("=" * 50)
    
    # Load our analysis results
    analysis_df = pd.read_csv('llamacpp_hate_analysis_20251028_155930.csv')
    print(f"   • LLaMA analysis records: {len(analysis_df)}")
    
    # Load processed dataset (only needed columns)
    needed_columns = ['comment_id', 'hatespeech', 'hate_speech_score', 'text']
    
    # Get comment IDs from our analysis
    analysis_comment_ids = set(analysis_df['comment_id'].astype(str))
    print(f"   • Comment IDs to match: {len(analysis_comment_ids)}")
    
    # Load processed dataset in chunks
    chunk_list = []
    chunk_size = 10000
    
    print("   • Loading processed dataset (chunked)...")
    for chunk in pd.read_csv('processed_dataset.csv', chunksize=chunk_size, usecols=needed_columns):
        chunk['comment_id'] = chunk['comment_id'].astype(str)
        matching_chunk = chunk[chunk['comment_id'].isin(analysis_comment_ids)]
        if len(matching_chunk) > 0:
            chunk_list.append(matching_chunk)
    
    processed_df = pd.concat(chunk_list, ignore_index=True) if chunk_list else pd.DataFrame()
    print(f"   • Processed records found: {len(processed_df)}")
    
    # Group by comment_id (multiple annotators per comment)
    processed_grouped = processed_df.groupby('comment_id').agg({
        'hatespeech': 'mean',  # Average human rating
        'hate_speech_score': 'mean',
        'text': 'first'
    }).reset_index()
    
    # Merge datasets
    analysis_df['comment_id'] = analysis_df['comment_id'].astype(str)
    merged_df = analysis_df.merge(processed_grouped, on='comment_id', how='inner', suffixes=('_llama', '_human'))
    
    print(f"   • Successfully merged: {len(merged_df)} records")
    return merged_df

def create_rating_distribution_comparison(merged_df):
    """Create rating distribution comparison visualization"""
    
    print("\n🎨 CREATING RATING DISTRIBUTION COMPARISON")
    print("-" * 50)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Rating Distribution Comparison: LLaMA vs Human Annotations', fontsize=16, fontweight='bold')
    
    # Plot 1: Distribution histograms
    # Both systems use 0-2 scale - no scaling needed!
    
    ax1.hist(merged_df['hatespeech'], bins=20, alpha=0.6, label='Human Annotations (0-2)', 
             density=True, color='blue', edgecolor='black')
    ax1.hist(merged_df['rating'], bins=20, alpha=0.6, label='LLaMA Predictions (0-2)', 
             density=True, color='red', edgecolor='black')
    
    ax1.set_xlabel('Rating (0-2 Scale)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Rating Distribution Comparison (Same Scale)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    human_mean = merged_df['hatespeech'].mean()
    human_std = merged_df['hatespeech'].std()
    llama_mean = merged_df['rating'].mean()
    llama_std = merged_df['rating'].std()
    
    stats_text = f'Human: μ={human_mean:.2f}, σ={human_std:.2f}\nLLaMA: μ={llama_mean:.2f}, σ={llama_std:.2f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Plot 2: Scatter plot with correlation (both on 0-2 scale)
    ax2.scatter(merged_df['hatespeech'], merged_df['rating'], alpha=0.6, s=30, color='green')
    ax2.set_xlabel('Human Hate Speech Rating (0-2)', fontsize=12)
    ax2.set_ylabel('LLaMA Rating (0-2)', fontsize=12)
    ax2.set_title('Direct Comparison: Human vs LLaMA (Same Scale)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add perfect agreement line (y=x)
    ax2.plot([0, 2], [0, 2], 'k--', alpha=0.5, linewidth=1, label='Perfect Agreement')
    
    # Add correlation line
    z = np.polyfit(merged_df['hatespeech'], merged_df['rating'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_df['hatespeech'].min(), merged_df['hatespeech'].max(), 100)
    ax2.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2, label='Best Fit Line')
    ax2.legend()
    
    # Calculate and display correlation
    corr, p_value = pearsonr(merged_df['hatespeech'], merged_df['rating'])
    corr_text = f'Pearson r = {corr:.3f}\np-value = {p_value:.2e}'
    ax2.text(0.05, 0.95, corr_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
             verticalalignment='top', fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'rating_distribution_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Distribution comparison saved: {plot_file}")
    plt.close()  # Close plot instead of showing it
    
    # Explain the correlation graph
    explain_correlation_graph(corr, p_value, merged_df)
    
    return corr

def explain_correlation_graph(correlation, p_value, merged_df):
    """Provide detailed explanation of the correlation graph"""
    
    print("\n" + "="*70)
    print("📈 CORRELATION GRAPH EXPLANATION (Same 0-2 Scale)")
    print("="*70)
    
    print(f"\n🔍 WHAT THE CORRELATION GRAPH SHOWS:")
    print(f"   • Each dot represents one comment")
    print(f"   • X-axis: Human annotator ratings (0-2 scale)")
    print(f"   • Y-axis: LLaMA model ratings (0-2 scale)")
    print(f"   • Black dashed line: Perfect agreement (y=x)")
    print(f"   • Red solid line: Best-fit trend line")
    
    print(f"\n📊 PERFECT SCALE ALIGNMENT:")
    print(f"   • Both systems use identical 0-2 scale")
    print(f"   • 0.0 = No hate speech")
    print(f"   • 1.0 = Unclear/borderline")
    print(f"   • 2.0 = Clear hate speech")
    print(f"   • Dots on black line = perfect agreement")
    
    print(f"\n📊 CORRELATION COEFFICIENT (r = {correlation:.4f}):")
    
    if abs(correlation) >= 0.8:
        strength = "Very Strong"
        emoji = "🟢"
    elif abs(correlation) >= 0.6:
        strength = "Strong"
        emoji = "🟢"
    elif abs(correlation) >= 0.4:
        strength = "Moderate"
        emoji = "🟡"
    elif abs(correlation) >= 0.2:
        strength = "Weak"
        emoji = "🟡"
    else:
        strength = "Very Weak"
        emoji = "🔴"
    
    direction = "positive" if correlation > 0 else "negative"
    
    print(f"   {emoji} Strength: {strength} {direction} correlation")
    print(f"   • Your value: {correlation:.4f}")
    
    print(f"\n🎯 WHAT THIS MEANS FOR SAME-SCALE COMPARISON:")
    if abs(correlation) >= 0.9:
        print(f"   ✅ Exceptional! LLaMA nearly matches human ratings")
    elif abs(correlation) >= 0.7:
        print(f"   ✅ Excellent agreement! Very similar rating patterns")
    elif abs(correlation) >= 0.5:
        print(f"   ✅ Good agreement. Generally consistent ratings")
    elif abs(correlation) >= 0.3:
        print(f"   ⚠️  Moderate agreement. Some systematic differences")
    else:
        print(f"   ❌ Poor agreement. Different rating patterns")
    
    print(f"\n� DETAILED SAME-SCALE ANALYSIS:")
    
    # Calculate exact matches and near matches
    exact_matches = (merged_df['hatespeech'] == merged_df['rating']).sum()
    near_matches = (abs(merged_df['hatespeech'] - merged_df['rating']) <= 0.5).sum()
    
    human_dist = merged_df['hatespeech'].value_counts().sort_index()
    llama_dist = merged_df['rating'].round().value_counts().sort_index()
    
    print(f"   • Exact matches: {exact_matches}/{len(merged_df)} ({exact_matches/len(merged_df)*100:.1f}%)")
    print(f"   • Near matches (±0.5): {near_matches}/{len(merged_df)} ({near_matches/len(merged_df)*100:.1f}%)")
    
    print(f"\n� RATING DISTRIBUTION COMPARISON:")
    print(f"   Rating  Human    LLaMA")
    print(f"   ------  -----    -----")
    for rating in [0.0, 1.0, 2.0]:
        human_count = human_dist.get(rating, 0)
        llama_count = llama_dist.get(rating, 0)
        print(f"   {rating:4.1f}   {human_count:5d}   {llama_count:5d}")
    
    print("="*70)

def calculate_f1_scores(merged_df):
    """Calculate F1 scores for direct comparison (both systems use 0-2 scale)"""
    
    print("\n📈 CALCULATING F1 SCORES (Direct 0-2 Scale Comparison)")
    print("-" * 60)
    
    # Method 1: Binary classification (Hate vs No Hate)
    # Both systems: rating > 1.0 = hate, <= 1.0 = no hate
    
    human_binary = (merged_df['hatespeech'] > 1.0).astype(int)
    llama_binary = (merged_df['rating'] > 1.0).astype(int)
    
    f1_binary = f1_score(human_binary, llama_binary)
    
    # Print F1 score prominently
    print("\n" + "="*60)
    print(f"🎯 F1 SCORE RESULT: {f1_binary:.4f}")
    print("="*60)
    print(f"🎯 Binary F1 Score (Hate vs No-Hate): {f1_binary:.4f}")
    
    # Method 2: Three-class classification (exact value matching)
    # Both systems use same scale: 0.0, 1.0, 2.0
    
    def to_3class(rating):
        if rating < 0.5:
            return 0  # No hate (0.0)
        elif rating < 1.5:
            return 1  # Unclear (1.0)
        else:
            return 2  # Hate (2.0)
    
    human_3class = merged_df['hatespeech'].apply(to_3class)
    llama_3class = merged_df['rating'].apply(to_3class)
    
    f1_3class_macro = f1_score(human_3class, llama_3class, average='macro')
    f1_3class_micro = f1_score(human_3class, llama_3class, average='micro')
    f1_3class_weighted = f1_score(human_3class, llama_3class, average='weighted')
    
    print(f"\n🎯 Three-class F1 Scores (0-2 Scale):")
    print(f"   • Macro average: {f1_3class_macro:.4f}")
    print(f"   • Micro average: {f1_3class_micro:.4f}")
    print(f"   • Weighted average: {f1_3class_weighted:.4f}")
    
    # Method 3: Exact agreement (since scales are identical)
    exact_agreement = (merged_df['hatespeech'] == merged_df['rating']).mean()
    print(f"\n🎯 Exact Agreement Rate: {exact_agreement:.4f} ({exact_agreement*100:.1f}%)")
    
    # Detailed classification report
    class_names = ['No Hate (0)', 'Unclear (1)', 'Hate Speech (2)']
    report = classification_report(human_3class, llama_3class, 
                                 target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(f"{'Class':<16} {'Precision':<10} {'Recall':<8} {'F1-Score':<8} {'Support':<8}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            support = int(report[str(i)]['support'])
            print(f"{class_name:<16} {precision:<10.3f} {recall:<8.3f} {f1:<8.3f} {support:<8d}")
    
    # Confusion matrix
    cm = confusion_matrix(human_3class, llama_3class)
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"{'Predicted →':<16} {'No Hate':<8} {'Unclear':<8} {'Hate':<8}")
    print("-" * 48)
    for i, true_label in enumerate(['No Hate', 'Unclear', 'Hate Speech']):
        row = f"{true_label:<16}"
        for j in range(len(class_names)):
            row += f" {cm[i,j]:<8d}"
        print(row)
    
    # Calculate accuracy
    accuracy = (human_3class == llama_3class).mean()
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f}")
    
    # Rating difference analysis
    rating_diff = merged_df['rating'] - merged_df['hatespeech']
    mean_abs_error = abs(rating_diff).mean()
    print(f"🎯 Mean Absolute Error: {mean_abs_error:.4f}")
    
    return {
        'f1_binary': f1_binary,
        'f1_macro': f1_3class_macro,
        'f1_micro': f1_3class_micro,
        'f1_weighted': f1_3class_weighted,
        'exact_agreement': exact_agreement,
        'accuracy': accuracy,
        'mean_abs_error': mean_abs_error,
        'correlation': None  # Will be filled later
    }

def main():
    """Main evaluation function"""
    
    print("🎯 LLAMA VS HUMAN ANNOTATIONS EVALUATION")
    print("Focus: Rating Distribution + F1 Scores")
    print("=" * 60)
    
    # Load data
    merged_df = load_comparison_data()
    
    if len(merged_df) == 0:
        print("❌ No data available for comparison")
        return
    
    # Create rating distribution comparison
    correlation = create_rating_distribution_comparison(merged_df)
    
    # Calculate F1 scores
    metrics = calculate_f1_scores(merged_df)
    metrics['correlation'] = correlation
    
    # Print F1 Score prominently on terminal
    print("\n" + "🔥"*30)
    print("🔥" + " "*10 + "F1 SCORE RESULTS" + " "*10 + "🔥")
    print("🔥"*30)
    print(f"🎯 PRIMARY F1 SCORE: {metrics['f1_binary']:.4f}")
    print(f"🎯 WEIGHTED F1 SCORE: {metrics['f1_weighted']:.4f}")
    print(f"🎯 CORRELATION: {correlation:.4f}")
    print("🔥"*30)
    
    # Summary
    print(f"\n" + "="*70)
    print(f"📊 EVALUATION RESULTS SUMMARY")
    print("="*70)
    print(f"📊 Records evaluated: {len(merged_df)}")
    print(f"🎯 Binary F1 Score (Hate vs No-Hate): {metrics['f1_binary']:.4f}")
    print(f"🎯 Three-class Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"📈 Pearson Correlation: {correlation:.4f}")
    print(f"📊 Overall Accuracy: {metrics['accuracy']:.4f}")
    print("="*70)
    
    # Save metrics
    import json
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved: evaluation_metrics.json")
    
    # Performance interpretation
    print(f"\n🔍 PERFORMANCE INTERPRETATION:")
    if metrics['f1_weighted'] >= 0.7:
        print("🟢 Good performance (F1 ≥ 0.7)")
    elif metrics['f1_weighted'] >= 0.5:
        print("🟡 Moderate performance (0.5 ≤ F1 < 0.7)")
    else:
        print("🔴 Poor performance (F1 < 0.5)")
    
    if abs(correlation) >= 0.7:
        print("🟢 Strong correlation with human annotations")
    elif abs(correlation) >= 0.4:
        print("🟡 Moderate correlation with human annotations")
    else:
        print("🔴 Weak correlation with human annotations")

if __name__ == "__main__":
    main()
