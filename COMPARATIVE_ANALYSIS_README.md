# Multi-Attribute Hate Speech Analysis - Comparative Study

## 📁 File Overview

This directory contains two different approaches to multi-attribute hate speech analysis for comparative research:

### 1. **Vanilla Prompting** (`llamacpp_vanilla_analyzer.py`)
- **Purpose:** Baseline analysis without any annotator demographic information
- **Approach:** Simple, neutral prompts asking for objective ratings
- **Use case:** Control group to measure base model performance

### 2. **Annotator Role Prompting** (`llamacpp_annotator_role_analyzer.py`)
- **Purpose:** Test if specifying annotator demographics affects LLM ratings
- **Approach:** Prompts include specific annotator profiles (gender, race, religion, ideology)
- **Use case:** Experimental group to measure demographic bias effects

---

## 🎯 Research Question

**Does specifying annotator demographic characteristics in the prompt influence LLM hate speech ratings?**

This addresses:
- **Demographic bias**: Do different "annotator personas" rate content differently?
- **Perspective awareness**: Can LLMs simulate diverse viewpoints?
- **Fairness**: Should content moderation systems account for annotator diversity?

---

## 🚀 Quick Start

### Run Vanilla Analysis (Baseline)
```bash
python llamacpp_vanilla_analyzer.py selected_comments.csv
```

**Output:** `vanilla_analysis_TIMESTAMP.csv`
- 200 comments analyzed
- No demographic information in prompts
- Pure "objective" ratings

### Run Annotator Role Analysis (Experimental)
```bash
python llamacpp_annotator_role_analyzer.py selected_comments.csv
```

**Output:** `annotator_role_analysis_TIMESTAMP.csv`
- 50 comments × 10 profiles = 500 analyses
- Each comment rated from 10 different demographic perspectives
- Includes annotator profile columns

---

## 👥 Annotator Profiles Tested

### **Gender Profiles**
1. **male_basic**: Male annotator
2. **female_basic**: Female annotator

### **Demographic Profiles**
3. **young_white_male**: 25yo White male, college educated
4. **middle_black_female**: 45yo Black female, graduate degree
5. **older_asian_male**: 60yo Asian male, high school education

### **Religion + Ideology Profiles**
6. **christian_conservative**: Christian male, conservative
7. **muslim_moderate**: Muslim female, moderate
8. **jewish_liberal**: Jewish male, liberal
9. **atheist_liberal**: Atheist female, liberal

### **Comprehensive Profile**
10. **comprehensive_diverse**: Hispanic Catholic female, 35, graduate degree, moderate, middle income

---

## 📊 Output Format

### Vanilla Analysis Output
```csv
sentiment,respect,insult,humiliate,status,dehumanize,violence,genocide,attack_defend,confidence,prompt_type,raw_output,text,comment_id,index
3.0,2.0,1.0,1.0,2.0,1.0,1.0,1.0,2.0,0.8,vanilla,"...",Afghanistan will rise...,39073,1
```

### Annotator Role Analysis Output
```csv
sentiment,respect,insult,humiliate,status,dehumanize,violence,genocide,attack_defend,confidence,prompt_type,annotator_gender,annotator_age,annotator_race,annotator_religion,annotator_ideology,profile_id,raw_output,text,comment_id,index
3.0,2.0,1.0,1.0,2.0,1.0,1.0,1.0,2.0,0.8,annotator_role,male,25 years old,White/Caucasian,,,young_white_male,"...",Afghanistan will rise...,39073,1
3.2,1.8,1.5,1.0,1.8,1.0,1.0,1.0,2.5,0.7,annotator_role,female,45 years old,Black/African American,,,middle_black_female,"...",Afghanistan will rise...,39073,1
```

---

## 📈 Analysis Workflow

### Step 1: Run Both Analyzers
```bash
# Baseline (200 comments, ~15-20 minutes)
python llamacpp_vanilla_analyzer.py selected_comments.csv

# Experimental (50 comments × 10 profiles, ~30-40 minutes)
python llamacpp_annotator_role_analyzer.py selected_comments.csv
```

### Step 2: Compare Results

**Key Metrics to Compare:**

1. **Inter-profile variance**: Do different profiles rate the same comment differently?
2. **Attribute sensitivity**: Which attributes show most variation across profiles?
3. **Demographic patterns**: Do gender, race, or religion correlate with rating differences?
4. **Baseline comparison**: How much do role-prompted ratings differ from vanilla?

### Step 3: Statistical Analysis

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load results
vanilla_df = pd.read_csv('vanilla_analysis_TIMESTAMP.csv')
role_df = pd.read_csv('annotator_role_analysis_TIMESTAMP.csv')

# Calculate variance per comment across profiles
role_grouped = role_df.groupby('comment_id')['sentiment'].agg(['mean', 'std', 'min', 'max'])

print("Average sentiment variance across profiles:")
print(f"Mean std: {role_grouped['std'].mean():.3f}")
print(f"Max range: {(role_grouped['max'] - role_grouped['min']).max():.3f}")

# Compare vanilla vs role-prompted averages
role_avg = role_df.groupby('comment_id')['sentiment'].mean()
vanilla_vals = vanilla_df.set_index('comment_id')['sentiment']

# Merge on common comment_ids
merged = pd.DataFrame({
    'vanilla': vanilla_vals,
    'role_avg': role_avg
}).dropna()

# Calculate correlation
corr, p_value = stats.pearsonr(merged['vanilla'], merged['role_avg'])
print(f"\nVanilla vs Role-Prompted Correlation: r={corr:.3f}, p={p_value:.4f}")

# Test if specific demographics show systematic differences
gender_comparison = role_df.groupby(['comment_id', 'annotator_gender'])['sentiment'].mean().unstack()
print("\nAverage sentiment by annotator gender:")
print(gender_comparison.mean())
```

---

## 🔬 Expected Outcomes

### Hypothesis 1: Minimal Variation (Null)
- **Prediction**: All profiles rate similarly, variance is low
- **Interpretation**: LLM is not influenced by demographic priming
- **Implication**: Demographic prompting is ineffective

### Hypothesis 2: Systematic Variation (Alternative)
- **Prediction**: Different profiles show consistent rating patterns
- **Interpretation**: LLM simulates perspective-taking based on demographics
- **Implication**: Demographic diversity in prompting matters

### Hypothesis 3: Random Variation (Noise)
- **Prediction**: High variance but no systematic patterns
- **Interpretation**: LLM responses are inconsistent/random
- **Implication**: Model quality issue, not perspective simulation

---

## 📋 Comparative Analysis Checklist

- [ ] Run vanilla analysis (baseline)
- [ ] Run annotator role analysis (experimental)
- [ ] Calculate inter-profile variance per comment
- [ ] Test for systematic demographic effects (gender, race, religion)
- [ ] Compare vanilla vs role-prompted average ratings
- [ ] Analyze which attributes show most variation
- [ ] Test statistical significance of differences
- [ ] Visualize rating distributions by profile
- [ ] Document findings in research report

---

## 📊 Visualization Examples

### Plot 1: Sentiment Distribution by Profile
```python
import matplotlib.pyplot as plt
import seaborn as sns

role_df = pd.read_csv('annotator_role_analysis_TIMESTAMP.csv')

plt.figure(figsize=(12, 6))
sns.boxplot(data=role_df, x='profile_id', y='sentiment')
plt.xticks(rotation=45, ha='right')
plt.title('Sentiment Ratings by Annotator Profile')
plt.ylabel('Sentiment (1-5)')
plt.xlabel('Annotator Profile')
plt.tight_layout()
plt.savefig('sentiment_by_profile.png', dpi=300)
```

### Plot 2: Vanilla vs Role-Prompted Comparison
```python
# Calculate average role-prompted rating per comment
role_avg = role_df.groupby('comment_id')[['sentiment', 'insult', 'violence']].mean()
vanilla_vals = vanilla_df.set_index('comment_id')[['sentiment', 'insult', 'violence']]

merged = vanilla_vals.join(role_avg, lsuffix='_vanilla', rsuffix='_role').dropna()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
attrs = ['sentiment', 'insult', 'violence']

for i, attr in enumerate(attrs):
    axes[i].scatter(merged[f'{attr}_vanilla'], merged[f'{attr}_role'], alpha=0.5)
    axes[i].plot([1, 5], [1, 5], 'r--', label='Perfect Agreement')
    axes[i].set_xlabel(f'Vanilla {attr.title()}')
    axes[i].set_ylabel(f'Role-Prompted Avg {attr.title()}')
    axes[i].set_title(f'{attr.title()} Comparison')
    axes[i].legend()

plt.tight_layout()
plt.savefig('vanilla_vs_role_comparison.png', dpi=300)
```

### Plot 3: Heatmap of Profile Differences
```python
# Create matrix of average ratings per profile
profile_matrix = role_df.groupby('profile_id')[
    ['sentiment', 'respect', 'insult', 'humiliate', 'violence']
].mean()

plt.figure(figsize=(10, 8))
sns.heatmap(profile_matrix.T, annot=True, fmt='.2f', cmap='RdYlGn_r')
plt.title('Average Attribute Ratings by Annotator Profile')
plt.ylabel('Attribute')
plt.xlabel('Annotator Profile')
plt.tight_layout()
plt.savefig('profile_attribute_heatmap.png', dpi=300)
```

---

## ⚠️ Important Notes

### Sample Sizes
- **Vanilla**: 200 comments (larger sample, baseline)
- **Role**: 50 comments × 10 profiles = 500 analyses (intensive)
- **Rationale**: Role analysis is 10× more computationally expensive

### Runtime Estimates
- **Vanilla**: ~15-20 minutes (200 comments)
- **Role**: ~30-40 minutes (500 analyses)
- **Per comment**: ~5-6 seconds with 3B model

### Model Requirements
- Works with any GGUF model (Llama-3.2-3B, Llama-3.1-8B, Phi-3)
- Larger models (8B) may show stronger demographic effects
- Quantized models (Q4) sufficient for this task

---

## 🎓 Research Applications

### Use Cases:
1. **Bias Detection**: Measure if LLMs have built-in demographic biases
2. **Perspective Simulation**: Test if LLMs can adopt different viewpoints
3. **Fairness Research**: Study how annotator diversity affects content moderation
4. **Prompt Engineering**: Understand effects of demographic priming
5. **Comparative ML**: Benchmark against human annotator variance

### Related Work:
- Annotator demographic effects on hate speech labeling
- Perspective API and fairness across demographics
- Social stereotypes in language models
- Content moderation bias studies

---

## 📚 References

- Original analysis: `llamacpp_sentiment_analyzer.py`
- Evaluation framework: `evaluation_multiattribute.py`
- Dataset: `processed_dataset.csv` (39,565 human-annotated comments)
- Project report: `PROJECT_REPORT.md`

---

## 🆘 Troubleshooting

**Issue**: Model not found
```bash
# Check for GGUF files
find . -name "*.gguf"
```

**Issue**: Out of memory
```bash
# Use smaller model or reduce batch size
# Edit script: sample_size=20 (instead of 50)
```

**Issue**: Slow performance
```bash
# Enable GPU acceleration (if available)
# Edit script: n_gpu_layers=35
```

---

## 📬 Next Steps

1. Run both analyses
2. Compare vanilla vs annotator role results
3. Calculate statistical significance of differences
4. Create visualizations
5. Update `PROJECT_REPORT.md` with findings
6. Consider testing with larger model (8B) for stronger effects
