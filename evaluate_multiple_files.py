#!/usr/bin/env python3
"""
Multi-File Evaluation Script
Evaluates multiple multi-attribute analysis CSV files and compares them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import json
from datetime import datetime

def find_all_analysis_files(pattern="annotator_role_female_basic_*.csv"):
    """Find all multi-attribute analysis CSV files"""
    files = glob.glob(pattern)
    if not files:
        print(f"❌ No files matching pattern: {pattern}")
        return []
    
    # Sort by creation time
    files_with_time = [(f, os.path.getctime(f)) for f in files]
    files_sorted = sorted(files_with_time, key=lambda x: x[1], reverse=True)
    
    print(f"📁 Found {len(files)} analysis file(s):")
    for i, (f, ctime) in enumerate(files_sorted, 1):
        size_kb = os.path.getsize(f) / 1024
        timestamp = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {i}. {f} ({size_kb:.1f} KB, created {timestamp})")
    
    return [f for f, _ in files_sorted]

def load_ground_truth():
    """Load the processed dataset with human ratings"""
    print("\n📄 Loading ground truth from processed_dataset.csv...")
    try:
        attribute_columns = [
            'comment_id', 'sentiment', 'respect', 'insult', 'humiliate', 
            'status', 'dehumanize', 'violence', 'genocide', 'attack_defend', 
            'hatespeech', 'text'
        ]
        
        df = pd.read_csv("processed_dataset.csv", usecols=attribute_columns)
        
        # Group by comment_id for average ratings
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
        
        print(f"✅ Loaded {len(df_grouped)} unique comments with ground truth")
        return df_grouped
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return None

def get_metric_prefix_from_filename(filename):
    """Extract metric prefix from analysis filename"""
    basename = os.path.basename(filename).replace('.csv', '')
    
    # Remove timestamp pattern (assumes format: _YYYYMMDD_HHMMSS at the end)
    import re
    basename_no_timestamp = re.sub(r'_\d{8}_\d{6}$', '', basename)
    
    return basename_no_timestamp

def save_visualizations(merged_df, attributes, metric_prefix, timestamp):
    """Save correlation matrix and scatter plots with proper naming"""
    print(f"\n🎨 Creating visualizations for {metric_prefix}...")
    
    viz_files = []
    
    # 1. Correlation Matrix
    plt.figure(figsize=(12, 10))
    
    # Build correlation matrix for LLM vs Human attributes
    cols_to_corr = []
    for attr in attributes:
        if attr in merged_df.columns and f"{attr}_human" in merged_df.columns:
            cols_to_corr.extend([attr, f"{attr}_human"])
    
    if cols_to_corr:
        corr_matrix = merged_df[cols_to_corr].corr()
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', 
                    vmin=-1, vmax=1, center=0, mask=mask)
        
        plt.title(f'Correlation Matrix: {metric_prefix}', fontsize=14, pad=20)
        plt.tight_layout()
        
        corr_filename = f"{metric_prefix}_correlation_matrix_{timestamp}.png"
        plt.savefig(corr_filename, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(corr_filename)
        print(f"📊 Saved correlation matrix: {corr_filename}")
    
    # 2. Scatter Plots for top 3 correlated attributes
    correlations = {}
    for attr in attributes:
        if attr in merged_df.columns and f"{attr}_human" in merged_df.columns:
            corr = merged_df[attr].corr(merged_df[f"{attr}_human"])
            if not np.isnan(corr):
                correlations[attr] = corr
    
    if correlations:
        top_attrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Top 3 Correlated Attributes: {metric_prefix}', fontsize=16)
        
        for i, (attr, corr) in enumerate(top_attrs):
            human_attr = f"{attr}_human"
            sns.regplot(data=merged_df, x=human_attr, y=attr, ax=axes[i], 
                        scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            axes[i].set_title(f'LLM vs Human {attr.title()}\nr = {corr:.3f}')
            axes[i].set_xlabel(f'Human {attr.title()}')
            axes[i].set_ylabel(f'LLM {attr.title()}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_filename = f"{metric_prefix}_scatter_plots_{timestamp}.png"
        plt.savefig(scatter_filename, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(scatter_filename)
        print(f"📊 Saved scatter plots: {scatter_filename}")
    
    return viz_files

def evaluate_single_file(analysis_file, ground_truth_df, save_individual_outputs=True):
    """Evaluate a single analysis file and optionally save individual metrics/visualizations"""
    print(f"\n{'='*70}")
    print(f"🎯 EVALUATING: {analysis_file}")
    print(f"{'='*70}")
    
    # Load analysis results
    analysis_df = pd.read_csv(analysis_file)
    print(f"📊 Analysis contains {len(analysis_df)} comments")
    
    # Check what type of analysis (vanilla, annotator_role, etc.)
    prompt_type = analysis_df.get('prompt_type', ['unknown'])[0] if 'prompt_type' in analysis_df.columns else 'standard'
    print(f"📝 Prompt type: {prompt_type}")
    
    # Merge with ground truth
    merged_df = analysis_df.merge(
        ground_truth_df,
        on='comment_id',
        suffixes=('', '_human')
    )
    
    print(f"✅ Merged {len(merged_df)} comments with ground truth")
    
    # Calculate metrics for each attribute
    attributes = ['sentiment', 'respect', 'insult', 'humiliate', 'status', 
                  'dehumanize', 'violence', 'genocide', 'attack_defend']
    
    results = {
        'file': analysis_file,
        'prompt_type': prompt_type,
        'n_comments': len(merged_df),
        'attributes': {}
    }
    
    print(f"\n📈 METRICS:")
    print("-" * 70)
    print(f"{'Attribute':<15} {'Correlation':<12} {'Accuracy':<12} {'F1-Macro':<12} {'MAE':<8}")
    print("-" * 70)
    
    for attr in attributes:
        llm_col = attr
        human_col = f"{attr}_human"
        
        if llm_col not in merged_df.columns or human_col not in merged_df.columns:
            continue
        
        # Correlation
        corr = merged_df[llm_col].corr(merged_df[human_col])
        
        # Classification metrics (convert to discrete classes)
        llm_discrete = np.round(merged_df[llm_col]).astype(int)
        human_discrete = np.round(merged_df[human_col]).astype(int)
        
        accuracy = accuracy_score(human_discrete, llm_discrete)
        f1 = f1_score(human_discrete, llm_discrete, average='macro', zero_division=0)
        mae = mean_absolute_error(merged_df[human_col], merged_df[llm_col])
        
        results['attributes'][attr] = {
            'correlation': float(corr) if not np.isnan(corr) else 0.0,
            'accuracy': float(accuracy),
            'f1_macro': float(f1),
            'mae': float(mae)
        }
        
        print(f"{attr:<15} {corr:>11.3f} {accuracy:>11.3f} {f1:>11.3f} {mae:>7.3f}")
    
    # Overall averages
    avg_corr = np.mean([m['correlation'] for m in results['attributes'].values()])
    avg_acc = np.mean([m['accuracy'] for m in results['attributes'].values()])
    avg_f1 = np.mean([m['f1_macro'] for m in results['attributes'].values()])
    avg_mae = np.mean([m['mae'] for m in results['attributes'].values()])
    
    print("-" * 70)
    print(f"{'AVERAGE':<15} {avg_corr:>11.3f} {avg_acc:>11.3f} {avg_f1:>11.3f} {avg_mae:>7.3f}")
    
    results['overall'] = {
        'avg_correlation': float(avg_corr),
        'avg_accuracy': float(avg_acc),
        'avg_f1_macro': float(avg_f1),
        'avg_mae': float(avg_mae)
    }
    
    # Save individual outputs if requested
    if save_individual_outputs:
        metric_prefix = get_metric_prefix_from_filename(analysis_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics JSON
        metrics_filename = f"{metric_prefix}_metrics_{timestamp}.json"
        with open(metrics_filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'analysis_file': analysis_file,
                'analysis_type': metric_prefix,
                'metrics': results
            }, f, indent=2)
        print(f"\n💾 Metrics saved: {metrics_filename}")
        
        # Save visualizations
        viz_files = save_visualizations(merged_df, attributes, metric_prefix, timestamp)
        results['metrics_file'] = metrics_filename
        results['visualization_files'] = viz_files
    
    return results

def compare_results(all_results):
    """Compare results across multiple files"""
    if len(all_results) < 2:
        print("\n⚠️  Only one file evaluated, no comparison available")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON ACROSS {len(all_results)} FILES")
    print(f"{'='*70}")
    
    # Create comparison table
    comparison_df = pd.DataFrame([
        {
            'File': os.path.basename(r['file']),
            'Type': r['prompt_type'],
            'N': r['n_comments'],
            'Avg Corr': r['overall']['avg_correlation'],
            'Avg Acc': r['overall']['avg_accuracy'],
            'Avg F1': r['overall']['avg_f1_macro'],
            'Avg MAE': r['overall']['avg_mae']
        }
        for r in all_results
    ])
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Identify best performing file
    best_corr = comparison_df.loc[comparison_df['Avg Corr'].idxmax()]
    best_acc = comparison_df.loc[comparison_df['Avg Acc'].idxmax()]
    best_f1 = comparison_df.loc[comparison_df['Avg F1'].idxmax()]
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"  • Best Correlation: {best_corr['File']} ({best_corr['Avg Corr']:.3f})")
    print(f"  • Best Accuracy: {best_acc['File']} ({best_acc['Avg Acc']:.3f})")
    print(f"  • Best F1-Score: {best_f1['File']} ({best_f1['Avg F1']:.3f})")
    
    return comparison_df

def main():
    print("🎯 MULTI-FILE EVALUATION TOOL")
    print("=" * 70)
    
    # Find all analysis files
    analysis_files = find_all_analysis_files()
    
    if not analysis_files:
        print("❌ No analysis files found. Run an analyzer first!")
        return
    
    # Ask user which files to evaluate
    print("\n❓ Evaluation options:")
    print("  1. Evaluate latest file only (default)")
    print("  2. Evaluate all files")
    print("  3. Select specific file")
    
    choice = input("\nEnter choice (1-3) or press Enter for default: ").strip()
    
    # Load ground truth
    ground_truth = load_ground_truth()
    if ground_truth is None:
        return
    
    all_results = []
    
    if choice == '2':
        # Evaluate all files
        print(f"\n📊 Evaluating all {len(analysis_files)} files...")
        for f in analysis_files:
            results = evaluate_single_file(f, ground_truth)
            all_results.append(results)
    
    elif choice == '3':
        # Select specific file
        print("\n📁 Select file number:")
        for i, f in enumerate(analysis_files, 1):
            print(f"  {i}. {f}")
        
        selection = input("\nEnter number: ").strip()
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(analysis_files):
                results = evaluate_single_file(analysis_files[idx], ground_truth)
                all_results.append(results)
            else:
                print("❌ Invalid selection")
                return
        except ValueError:
            print("❌ Invalid input")
            return
    
    else:
        # Default: evaluate latest only
        print(f"\n📊 Evaluating latest file: {analysis_files[0]}")
        results = evaluate_single_file(analysis_files[0], ground_truth)
        all_results.append(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"multi_file_evaluation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved: {output_file}")
    
    # Compare if multiple files
    if len(all_results) > 1:
        comparison_df = compare_results(all_results)
        
        # Save comparison
        comparison_file = f"comparison_summary_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"💾 Comparison saved: {comparison_file}")
    
    print(f"\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
