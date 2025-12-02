#!/usr/bin/env python3
"""
Multi-Attribute Evaluation Script
Evaluates how well LLaMA's sentiment and other attributes correlate with human hate speech ratings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
from sklearn.metrics import f1_score, accuracy_score, classification_report, mean_absolute_error
import json

def load_latest_analysis():
    """Find and load the latest multi-attribute analysis CSV"""
    files = glob.glob("annotator_role_christian_conservative_*.csv")
    if not files:
        print("❌ No multi-attribute analysis files found.")
        return None, None
    
    latest_file = max(files, key=os.path.getctime)
    print(f"📄 Loading latest analysis: {latest_file}")
    return pd.read_csv(latest_file), latest_file

def load_ground_truth():
    """Load the processed dataset with human ratings for all attributes"""
    print("📄 Loading ground truth from processed_dataset.csv...")
    try:
        # Load all attribute columns plus comment_id and text
        attribute_columns = [
            'comment_id', 'sentiment', 'respect', 'insult', 'humiliate', 
            'status', 'dehumanize', 'violence', 'genocide', 'attack_defend', 
            'hatespeech', 'text'
        ]
        
        df = pd.read_csv("processed_dataset.csv", usecols=attribute_columns)
        
        # Group by comment_id to get average human ratings (if multiple annotators)
        df_grouped = df.groupby('comment_id').agg({
            'sentiment': 'mean',
            'respect': 'mean', 
            'insult': 'mean',
            'humiliate': 'mean',
            'status': 'mean',
            'dehumanize': 'mean',
            'violence': 'mean',
            'genocide': 'mean',
            'attack_defend': 'mean',
            'hatespeech': 'mean',
            'text': 'first'
        }).reset_index()
        
        print(f"✅ Loaded {len(df_grouped)} unique comments with ground truth for all attributes")
        return df_grouped
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return None

def evaluate_correlations(merged_df):
    """Calculate and visualize correlations between LLM and human attributes"""
    print("\n📊 EVALUATING ATTRIBUTE-TO-ATTRIBUTE CORRELATIONS")
    print("=" * 60)
    
    # Attribute pairs (LLM vs Human)
    attribute_pairs = [
        ('sentiment', 'sentiment_human'),
        ('respect', 'respect_human'), 
        ('insult', 'insult_human'),
        ('humiliate', 'humiliate_human'),
        ('status', 'status_human'),
        ('dehumanize', 'dehumanize_human'),
        ('violence', 'violence_human'),
        ('genocide', 'genocide_human'),
        ('attack_defend', 'attack_defend_human')
    ]
    
    correlations = {}
    
    print(f"📈 LLM vs Human Attribute Correlations:")
    print(f"{'LLM Attribute':<15} {'Human Attribute':<18} {'Correlation':<12} {'Strength'}")
    print("-" * 70)
    
    for llm_attr, human_attr in attribute_pairs:
        if llm_attr in merged_df.columns and human_attr in merged_df.columns:
            corr = merged_df[llm_attr].corr(merged_df[human_attr])
            correlations[llm_attr] = corr
            
            # Determine correlation strength
            if abs(corr) >= 0.7:
                strength = "🟢 Strong"
            elif abs(corr) >= 0.4:
                strength = "🟡 Moderate" 
            elif abs(corr) >= 0.2:
                strength = "🟠 Weak"
            else:
                strength = "🔴 Very Weak"
                
            print(f"{llm_attr:<15} {human_attr:<18} {corr:+.4f}      {strength}")
        else:
            missing_col = llm_attr if llm_attr not in merged_df.columns else human_attr
            print(f"⚠️  Missing column: {missing_col}")
    
    return correlations

def calculate_classification_metrics(merged_df):
    """Calculate F1 scores, accuracy, and other metrics for each attribute"""
    print("\n🎯 CALCULATING CLASSIFICATION METRICS (LLM vs Human)")
    print("=" * 70)
    
    # Attribute pairs and their classification schemes
    attribute_configs = {
        'sentiment': {
            'llm_col': 'sentiment',
            'human_col': 'sentiment_human', 
            'scale': (1.0, 5.0),
            'classes': ['very_negative', 'negative', 'neutral', 'positive', 'very_positive'],
            'thresholds': [1.5, 2.5, 3.5, 4.5]
        },
        'respect': {
            'llm_col': 'respect',
            'human_col': 'respect_human',
            'scale': (1.0, 2.0),
            'classes': ['not_respectful', 'respectful'],
            'thresholds': [1.5]
        },
        'insult': {
            'llm_col': 'insult',
            'human_col': 'insult_human',
            'scale': (1.0, 4.0), 
            'classes': ['not_insulting', 'slightly_insulting', 'insulting', 'very_insulting'],
            'thresholds': [1.5, 2.5, 3.5]
        },
        'humiliate': {
            'llm_col': 'humiliate',
            'human_col': 'humiliate_human',
            'scale': (1.0, 3.0),
            'classes': ['not_humiliating', 'somewhat_humiliating', 'humiliating'],
            'thresholds': [1.5, 2.5]
        },
        'status': {
            'llm_col': 'status', 
            'human_col': 'status_human',
            'scale': (1.0, 2.0),
            'classes': ['inferior_status', 'equal_superior_status'],
            'thresholds': [1.5]
        },
        'dehumanize': {
            'llm_col': 'dehumanize',
            'human_col': 'dehumanize_human',
            'scale': (1.0, 2.0),
            'classes': ['not_dehumanizing', 'dehumanizing'],
            'thresholds': [1.5]
        },
        'violence': {
            'llm_col': 'violence',
            'human_col': 'violence_human',
            'scale': (1.0, 2.0),
            'classes': ['no_violence', 'calls_for_violence'],
            'thresholds': [1.5]
        },
        'genocide': {
            'llm_col': 'genocide',
            'human_col': 'genocide_human',
            'scale': (1.0, 2.0),
            'classes': ['no_genocide', 'calls_for_genocide'],
            'thresholds': [1.5]
        },
        'attack_defend': {
            'llm_col': 'attack_defend',
            'human_col': 'attack_defend_human',
            'scale': (1.0, 4.0),
            'classes': ['strongly_defending', 'somewhat_defending', 'somewhat_attacking', 'strongly_attacking'],
            'thresholds': [1.5, 2.5, 3.5]
        }
    }
    
    results = {}
    
    def classify_rating(rating, thresholds):
        """Convert continuous rating to discrete class"""
        for i, threshold in enumerate(thresholds):
            if rating <= threshold:
                return i
        return len(thresholds)
    
    print(f"{'Attribute':<15} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Weighted':<12} {'MAE':<8} {'Classes'}")
    print("-" * 85)
    
    for attr_name, config in attribute_configs.items():
        llm_col = config['llm_col']
        human_col = config['human_col']
        
        if llm_col in merged_df.columns and human_col in merged_df.columns:
            # Get valid data (no NaN)
            mask = merged_df[llm_col].notna() & merged_df[human_col].notna()
            llm_ratings = merged_df.loc[mask, llm_col]
            human_ratings = merged_df.loc[mask, human_col]
            
            if len(llm_ratings) == 0:
                print(f"⚠️  {attr_name:<15} No valid data")
                continue
            
            # Convert to classes
            llm_classes = [classify_rating(rating, config['thresholds']) for rating in llm_ratings]
            human_classes = [classify_rating(rating, config['thresholds']) for rating in human_ratings]
            
            # Calculate metrics
            accuracy = accuracy_score(human_classes, llm_classes)
            f1_macro = f1_score(human_classes, llm_classes, average='macro', zero_division=0)
            f1_weighted = f1_score(human_classes, llm_classes, average='weighted', zero_division=0)
            mae = mean_absolute_error(human_ratings, llm_ratings)
            
            results[attr_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'mae': mae,
                'n_samples': len(llm_ratings),
                'n_classes': len(config['classes']),
                'classes': config['classes']
            }
            
            print(f"{attr_name:<15} {accuracy:<10.3f} {f1_macro:<10.3f} {f1_weighted:<12.3f} {mae:<8.3f} {len(config['classes'])}")
            
        else:
            missing = llm_col if llm_col not in merged_df.columns else human_col
            print(f"⚠️  {attr_name:<15} Missing column: {missing}")
    
    # Overall summary
    if results:
        avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
        avg_f1_macro = np.mean([r['f1_macro'] for r in results.values()])
        avg_f1_weighted = np.mean([r['f1_weighted'] for r in results.values()])
        avg_mae = np.mean([r['mae'] for r in results.values()])
        
        print("-" * 85)
        print(f"{'AVERAGE':<15} {avg_accuracy:<10.3f} {avg_f1_macro:<10.3f} {avg_f1_weighted:<12.3f} {avg_mae:<8.3f}")
        print()
        
        # Performance interpretation
        print("🎯 PERFORMANCE SUMMARY:")
        print(f"   • Overall Accuracy: {avg_accuracy:.3f} ({'Good' if avg_accuracy >= 0.7 else 'Moderate' if avg_accuracy >= 0.5 else 'Poor'})")
        print(f"   • Overall F1-Macro: {avg_f1_macro:.3f} ({'Good' if avg_f1_macro >= 0.7 else 'Moderate' if avg_f1_macro >= 0.5 else 'Poor'})")
        print(f"   • Average MAE: {avg_mae:.3f} (lower is better)")
    
    return results

def save_detailed_metrics(metrics_results, analysis_file=None):
    """Save detailed metrics to JSON file with appropriate naming based on dataset type"""
    print("\n💾 Saving detailed metrics...")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine filename and metric prefix based on analysis type
    if analysis_file and 'likert' in analysis_file.lower():
        filename = f"likert_metrics_{timestamp}.json"
        metric_prefix = "likert"
    elif analysis_file and 'vanilla' in analysis_file.lower():
        filename = f"vanilla_metrics_{timestamp}.json"
        metric_prefix = "vanilla"
    elif analysis_file and 'annotator_role' in analysis_file.lower():
        # Extract profile name if available
        if '_male_basic_' in analysis_file:
            filename = f"annotator_role_male_basic_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_male_basic"
        elif '_female_basic_' in analysis_file:
            filename = f"annotator_role_female_basic_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_female_basic"
        elif '_young_white_male_' in analysis_file:
            filename = f"annotator_role_young_white_male_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_young_white_male"
        elif '_middle_black_female_' in analysis_file:
            filename = f"annotator_role_middle_black_female_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_middle_black_female"
        elif '_older_asian_male_' in analysis_file:
            filename = f"annotator_role_older_asian_male_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_older_asian_male"
        elif '_christian_conservative_' in analysis_file:
            filename = f"annotator_role_christian_conservative_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_christian_conservative"
        elif '_muslim_moderate_' in analysis_file:
            filename = f"annotator_role_muslim_moderate_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_muslim_moderate"
        elif '_jewish_liberal_' in analysis_file:
            filename = f"annotator_role_jewish_liberal_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_jewish_liberal"
        elif '_atheist_liberal_' in analysis_file:
            filename = f"annotator_role_atheist_liberal_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_atheist_liberal"
        elif '_comprehensive_diverse_' in analysis_file:
            filename = f"annotator_role_comprehensive_diverse_metrics_{timestamp}.json"
            metric_prefix = "annotator_role_comprehensive_diverse"
        else:
            filename = f"annotator_role_metrics_{timestamp}.json"
            metric_prefix = "annotator_role"
    else:
        filename = f"multiattribute_metrics_{timestamp}.json"
        metric_prefix = "standard"
    
    # Convert numpy types to Python types for JSON serialization
    json_data = {
        'timestamp': timestamp,
        'analysis_file': analysis_file if analysis_file else 'unknown',
        'analysis_type': metric_prefix,
        'metrics': {}
    }
    
    for attr_name, metrics in metrics_results.items():
        # Use the metric prefix in the keys
        json_data['metrics'][attr_name] = {
            f'{metric_prefix}_accuracy': float(metrics['accuracy']),
            f'{metric_prefix}_f1_macro': float(metrics['f1_macro']),
            f'{metric_prefix}_f1_weighted': float(metrics['f1_weighted']),
            f'{metric_prefix}_mae': float(metrics['mae']),
            'n_samples': int(metrics['n_samples']),
            'n_classes': int(metrics['n_classes'])
        }
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Detailed metrics saved: {filename}")
    print(f"📊 Metric names prefixed with: {metric_prefix}_")
    return filename, metric_prefix

def visualize_correlations(merged_df, correlations, metric_prefix="standard"):
    """Create visualization of correlations with appropriate naming"""
    print("\n🎨 Creating correlation visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    attributes = list(correlations.keys())
    
    # Calculate correlation matrix for LLM and human attributes only
    cols_to_corr = []
    for attr in attributes:
        if attr in merged_df.columns and f"{attr}_human" in merged_df.columns:
            cols_to_corr.extend([attr, f"{attr}_human"])
    
    corr_matrix = merged_df[cols_to_corr].corr()
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', 
                vmin=-1, vmax=1, center=0, mask=mask)
    
    plt.title('Correlation Matrix: LLaMA Attributes vs Human Hate Speech', fontsize=14)
    plt.tight_layout()
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{metric_prefix}_correlation_matrix_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"📊 Saved correlation matrix: {filename}")
    
    # Create scatter plots for top 3 correlated attributes with their human counterparts
    top_attrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Top 3 Correlated LLM vs Human Attribute Comparisons', fontsize=16)
    
    for i, (attr, corr) in enumerate(top_attrs):
        human_attr = f"{attr}_human"
        if human_attr in merged_df.columns:
            sns.regplot(data=merged_df, x=human_attr, y=attr, ax=axes[i], 
                        scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            axes[i].set_title(f'LLM vs Human {attr.title()}\nr = {corr:.3f}')
            axes[i].set_xlabel(f'Human {attr.title()}')
            axes[i].set_ylabel(f'LLM {attr.title()}')
    
    plt.tight_layout()
    scatter_filename = f"{metric_prefix}_scatter_plots_{timestamp}.png"
    plt.savefig(scatter_filename, dpi=300)
    print(f"📊 Saved scatter plots: {scatter_filename}")
    
    return filename, scatter_filename

def main():
    print("🎯 MULTI-ATTRIBUTE EVALUATION")
    print("=" * 60)
    
    # Load data
    analysis_df, analysis_file = load_latest_analysis()
    if analysis_df is None:
        return
        
    ground_truth_df = load_ground_truth()
    if ground_truth_df is None:
        return
    
    # Merge datasets and rename human columns for clarity
    print("\n🔄 Merging datasets...")
    # Ensure comment_id types match
    analysis_df['comment_id'] = analysis_df['comment_id'].astype(str)
    ground_truth_df['comment_id'] = ground_truth_df['comment_id'].astype(str)
    
    # Rename human columns to distinguish from LLM columns
    human_columns_rename = {
        'sentiment': 'sentiment_human',
        'respect': 'respect_human',
        'insult': 'insult_human', 
        'humiliate': 'humiliate_human',
        'status': 'status_human',
        'dehumanize': 'dehumanize_human',
        'violence': 'violence_human',
        'genocide': 'genocide_human',
        'attack_defend': 'attack_defend_human',
        'hatespeech': 'hatespeech_human'
    }
    
    ground_truth_df = ground_truth_df.rename(columns=human_columns_rename)
    
    merged_df = pd.merge(analysis_df, ground_truth_df, on='comment_id', how='inner')
    print(f"✅ Successfully merged {len(merged_df)} records")
    print(f"📊 Available LLM attributes: {[col for col in merged_df.columns if col in ['sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide', 'attack_defend']]}")
    print(f"📊 Available Human attributes: {[col for col in merged_df.columns if '_human' in col]}")
    
    if len(merged_df) == 0:
        print("❌ No matching records found. Check comment_ids.")
        return
    
    # Evaluate correlations
    correlations = evaluate_correlations(merged_df)
    
    # Calculate F1 scores and accuracy metrics
    metrics_results = calculate_classification_metrics(merged_df)
    
    # Save detailed metrics with appropriate filename and get metric_prefix
    metrics_file, metric_prefix = save_detailed_metrics(metrics_results, analysis_file)
    
    # Visualize with appropriate naming
    corr_matrix_file, scatter_plots_file = visualize_correlations(merged_df, correlations, metric_prefix)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Metrics saved: {metrics_file}")
    print(f"📁 Visualizations: {corr_matrix_file}, {scatter_plots_file}")

if __name__ == "__main__":
    main()
