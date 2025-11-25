#!/usr/bin/env python3
"""
Correlation Analysis: Hate Rating vs Confidence
Analyzes how hate speech ratings correlate with model confidence
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from llamacpp_hate_analyzer import LlamaCppHateAnalyzer

def analyze_existing_data():
    """Analyze correlation from existing CSV data"""
    print("📊 ANALYZING EXISTING DATA")
    print("=" * 50)
    
    # Load existing data
    csv_files = [f for f in os.listdir('.') if f.startswith('llamacpp_hate_analysis_') and f.endswith('.csv')]
    if csv_files:
        latest_file = sorted(csv_files)[-1]
        df = pd.read_csv(latest_file)
        
        print(f"📁 Loaded: {latest_file}")
        print(f"📏 Records: {len(df)}")
        
        # Basic statistics
        print(f"\n📈 RATING STATISTICS")
        print(f"Mean rating: {df['rating'].mean():.3f}")
        print(f"Std rating: {df['rating'].std():.3f}")
        print(f"Min rating: {df['rating'].min():.3f}")
        print(f"Max rating: {df['rating'].max():.3f}")
        
        print(f"\n🎯 CONFIDENCE STATISTICS")
        print(f"Mean confidence: {df['confidence'].mean():.3f}")
        print(f"Std confidence: {df['confidence'].std():.3f}")
        print(f"Min confidence: {df['confidence'].min():.3f}")
        print(f"Max confidence: {df['confidence'].max():.3f}")
        
        # Correlation analysis
        correlation_analysis(df)
        plot_correlation(df, 'existing')
        
        return df
    else:
        print("❌ No existing CSV files found")
        return None

def run_fresh_analysis(n_samples=50):
    """Run fresh analysis with new confidence-based system"""
    print(f"\n🔄 RUNNING FRESH ANALYSIS ({n_samples} samples)")
    print("=" * 50)
    
    # Load selected comments
    df_comments = pd.read_csv('selected_comments.csv')
    sample_comments = df_comments.head(n_samples)
    
    # Initialize analyzer
    analyzer = LlamaCppHateAnalyzer()
    
    results = []
    for idx, row in sample_comments.iterrows():
        try:
            result = analyzer.analyze_text(row['text'])
            results.append({
                'comment_id': row['comment_id'],
                'rating': result['rating'],
                'confidence': result['confidence'],
                'category': result['category'],
                'text': row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            })
            
            if len(results) % 10 == 0:
                print(f"  Processed: {len(results)}/{n_samples}")
                
        except Exception as e:
            print(f"  Error processing comment {row['comment_id']}: {e}")
    
    df_fresh = pd.DataFrame(results)
    
    # Save fresh results
    fresh_file = f'correlation_analysis_fresh_{len(results)}.csv'
    df_fresh.to_csv(fresh_file, index=False)
    print(f"💾 Saved: {fresh_file}")
    
    # Analyze fresh data
    correlation_analysis(df_fresh)
    plot_correlation(df_fresh, 'fresh')
    
    return df_fresh

def correlation_analysis(df):
    """Perform statistical correlation analysis"""
    print(f"\n🔬 CORRELATION ANALYSIS")
    print("-" * 30)
    
    # Pearson correlation (linear relationship)
    pearson_r, pearson_p = pearsonr(df['rating'], df['confidence'])
    print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    
    # Spearman correlation (monotonic relationship)
    spearman_r, spearman_p = spearmanr(df['rating'], df['confidence'])
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
    
    # Interpretation
    if abs(pearson_r) < 0.1:
        strength = "negligible"
    elif abs(pearson_r) < 0.3:
        strength = "weak"
    elif abs(pearson_r) < 0.5:
        strength = "moderate"
    elif abs(pearson_r) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if pearson_r > 0 else "negative"
    
    print(f"\n📊 INTERPRETATION:")
    print(f"• Strength: {strength} {direction} correlation")
    print(f"• Statistical significance: {'Yes' if pearson_p < 0.05 else 'No'} (p < 0.05)")
    
    # Category breakdown
    print(f"\n📋 CATEGORY BREAKDOWN:")
    category_stats = df.groupby('category').agg({
        'rating': ['mean', 'std', 'count'],
        'confidence': ['mean', 'std']
    }).round(3)
    print(category_stats)

def plot_correlation(df, data_type):
    """Create correlation visualization"""
    print(f"\n🎨 Creating correlation plots for {data_type} data...")
    
    # Set up the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Hate Rating vs Confidence Analysis ({data_type.title()} Data)', fontsize=16)
    
    # 1. Scatter plot with regression line
    sns.scatterplot(data=df, x='confidence', y='rating', hue='category', ax=ax1, alpha=0.7)
    sns.regplot(data=df, x='confidence', y='rating', ax=ax1, scatter=False, color='red')
    ax1.set_title('Rating vs Confidence (with trend line)')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Hate Rating')
    
    # 2. Hexbin plot for density
    hb = ax2.hexbin(df['confidence'], df['rating'], gridsize=20, cmap='YlOrRd')
    ax2.set_title('Rating vs Confidence (density)')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Hate Rating')
    plt.colorbar(hb, ax=ax2, label='Count')
    
    # 3. Box plot by confidence bins
    df_binned = df.copy()
    df_binned['confidence_bin'] = pd.cut(df['confidence'], 
                                       bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0], 
                                       labels=['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'])
    sns.boxplot(data=df_binned, x='confidence_bin', y='rating', ax=ax3)
    ax3.set_title('Rating distribution by confidence bins')
    ax3.set_xlabel('Confidence Bins')
    ax3.set_ylabel('Hate Rating')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Correlation heatmap
    corr_matrix = df[['rating', 'confidence']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Correlation Matrix')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f'correlation_analysis_{data_type}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot: {plot_file}")
    plt.show()

def main():
    """Main analysis function"""
    import os
    
    print("🔍 HATE RATING vs CONFIDENCE CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Analyze existing data first
    df_existing = analyze_existing_data()
    
    # Run fresh analysis
    print("\n" + "="*60)
    df_fresh = run_fresh_analysis(30)  # Smaller sample for testing
    
    # Compare if both exist
    if df_existing is not None and df_fresh is not None:
        print(f"\n📊 COMPARISON: OLD vs NEW")
        print("-" * 30)
        print(f"Old system - Mean rating: {df_existing['rating'].mean():.3f}, Mean confidence: {df_existing['confidence'].mean():.3f}")
        print(f"New system - Mean rating: {df_fresh['rating'].mean():.3f}, Mean confidence: {df_fresh['confidence'].mean():.3f}")
        
        old_corr = df_existing['rating'].corr(df_existing['confidence'])
        new_corr = df_fresh['rating'].corr(df_fresh['confidence'])
        print(f"Old correlation: {old_corr:.4f}")
        print(f"New correlation: {new_corr:.4f}")

if __name__ == "__main__":
    main()