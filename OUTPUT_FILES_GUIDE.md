# Output Files Guide - Multi-Attribute Analysis

## 📁 File Structure After Running Analyzers

### **1. Vanilla Analysis Output** (Baseline/Control)

**Command:**
```bash
python llamacpp_vanilla_analyzer.py selected_comments.csv
```

**Output:** `vanilla_analysis_YYYYMMDD_HHMMSS.csv`

**Structure:**
- **1 file** with 200 rows (200 comments)
- Pure baseline without demographic information

**Example filename:**
```
vanilla_analysis_20251202_143000.csv
```

---

### **2. Annotator Role Analysis Output** (Experimental)

**Command:**
```bash
python llamacpp_annotator_role_analyzer.py selected_comments.csv
```

**Output:** **11 files total**

#### **Individual Profile Files (10 files):**
Each profile gets its own CSV with 50 rows (50 comments):

```
annotator_role_male_basic_20251202_150000.csv              (50 rows)
annotator_role_female_basic_20251202_150000.csv            (50 rows)
annotator_role_young_white_male_20251202_150000.csv        (50 rows)
annotator_role_middle_black_female_20251202_150000.csv     (50 rows)
annotator_role_older_asian_male_20251202_150000.csv        (50 rows)
annotator_role_christian_conservative_20251202_150000.csv  (50 rows)
annotator_role_muslim_moderate_20251202_150000.csv         (50 rows)
annotator_role_jewish_liberal_20251202_150000.csv          (50 rows)
annotator_role_atheist_liberal_20251202_150000.csv         (50 rows)
annotator_role_comprehensive_diverse_20251202_150000.csv   (50 rows)
```

#### **Combined File (1 file):**
All profiles merged into one file:

```
annotator_role_combined_20251202_150000.csv                (500 rows)
```

---

## 🎯 How to Evaluate Each Profile

### **Option 1: Evaluate Individual Profile Files**

```bash
# Evaluate each profile separately
python evaluation_multiattribute.py
# Will pick the LATEST file (most recent profile analyzed)

# Or use the multi-file evaluator
python evaluate_multiple_files.py
# Choose option 2 "Evaluate all files"
# This will evaluate ALL 10 profile files individually
```

### **Option 2: Use Multi-File Evaluator (Recommended)**

```bash
python evaluate_multiple_files.py
```

**Interactive menu will show:**
```
📁 Found 11 analysis file(s):
  1. annotator_role_male_basic_20251202_150000.csv (26.0 KB)
  2. annotator_role_female_basic_20251202_150000.csv (26.0 KB)
  3. annotator_role_young_white_male_20251202_150000.csv (26.0 KB)
  ...
  11. annotator_role_combined_20251202_150000.csv (260.0 KB)

❓ Evaluation options:
  1. Evaluate latest file only
  2. Evaluate all files  ← Choose this!
  3. Select specific file
```

**This will generate:**
- Individual metrics for each profile
- Side-by-side comparison table
- Best/worst performing profiles
- `multi_file_evaluation_TIMESTAMP.json`
- `comparison_summary_TIMESTAMP.csv`

---

## 📊 File Comparison

| Analyzer | Files Created | Rows per File | Total Rows | Comments Analyzed |
|----------|---------------|---------------|------------|-------------------|
| **Vanilla** | 1 file | 200 | 200 | 200 |
| **Annotator Role** | 10 individual + 1 combined = 11 files | 50 (individual)<br>500 (combined) | 500 | 50 |

---

## 🔍 Quick Analysis Examples

### **Compare Two Specific Profiles:**

```python
import pandas as pd

# Load two profiles
male_df = pd.read_csv('annotator_role_male_basic_20251202_150000.csv')
female_df = pd.read_csv('annotator_role_female_basic_20251202_150000.csv')

# Merge on comment_id
merged = male_df.merge(female_df, on='comment_id', suffixes=('_male', '_female'))

# Compare sentiment ratings
print("Male vs Female Sentiment:")
print(merged[['comment_id', 'sentiment_male', 'sentiment_female']].head())

# Calculate correlation
corr = merged['sentiment_male'].corr(merged['sentiment_female'])
print(f"Correlation: {corr:.3f}")
```

### **Analyze All Profiles from Combined File:**

```python
import pandas as pd

# Load combined file
df = pd.read_csv('annotator_role_combined_20251202_150000.csv')

# Group by profile and calculate average sentiment
profile_sentiment = df.groupby('profile_id')['sentiment'].mean().sort_values()

print("Average Sentiment by Profile:")
print(profile_sentiment)

# Find comments with highest variance across profiles
variance_by_comment = df.groupby('comment_id')['sentiment'].var().sort_values(ascending=False)
print("\nComments with most disagreement:")
print(variance_by_comment.head(10))
```

### **Evaluate Specific Profile:**

```bash
# Method 1: Using original evaluation script (rename file first)
cp annotator_role_male_basic_20251202_150000.csv multi_attribute_analysis_temp.csv
python evaluation_multiattribute.py

# Method 2: Using multi-file evaluator
python evaluate_multiple_files.py
# Choose option 3: "Select specific file"
# Enter the number for the profile you want
```

---

## 🎨 Visualization Examples

### **Compare All Profiles on One Attribute:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load combined file
df = pd.read_csv('annotator_role_combined_20251202_150000.csv')

# Box plot of sentiment by profile
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='profile_id', y='sentiment')
plt.xticks(rotation=45, ha='right')
plt.title('Sentiment Distribution by Annotator Profile')
plt.ylabel('Sentiment (1-5)')
plt.tight_layout()
plt.savefig('sentiment_by_profile.png', dpi=300)
```

### **Heatmap of All Profiles:**

```python
# Create matrix of average ratings
profile_matrix = df.groupby('profile_id')[
    ['sentiment', 'respect', 'insult', 'violence', 'dehumanize']
].mean()

plt.figure(figsize=(10, 8))
sns.heatmap(profile_matrix.T, annot=True, fmt='.2f', cmap='RdYlGn_r')
plt.title('Average Attribute Ratings by Profile')
plt.ylabel('Attribute')
plt.xlabel('Profile')
plt.tight_layout()
plt.savefig('profile_heatmap.png', dpi=300)
```

---

## 📋 Workflow for Comparative Analysis

### **Step 1: Run Both Analyzers**

```bash
# Baseline (200 comments)
python llamacpp_vanilla_analyzer.py selected_comments.csv
# Output: vanilla_analysis_20251202_143000.csv

# Experimental (50 comments × 10 profiles)
python llamacpp_annotator_role_analyzer.py selected_comments.csv
# Output: 11 files (10 individual + 1 combined)
```

### **Step 2: Evaluate Each Profile**

```bash
# Evaluate all profile files
python evaluate_multiple_files.py
# Choose option 2: "Evaluate all files"
```

This produces:
- `multi_file_evaluation_TIMESTAMP.json` (detailed metrics)
- `comparison_summary_TIMESTAMP.csv` (side-by-side comparison)

### **Step 3: Compare with Vanilla Baseline**

```python
import pandas as pd

# Load vanilla results
vanilla_df = pd.read_csv('vanilla_analysis_20251202_143000.csv')

# Load best-performing profile
best_profile_df = pd.read_csv('annotator_role_male_basic_20251202_150000.csv')

# Merge on common comments
merged = vanilla_df.merge(best_profile_df, on='comment_id', suffixes=('_vanilla', '_profile'))

# Compare
print("Vanilla vs Profile Correlation:")
print(f"Sentiment: {merged['sentiment_vanilla'].corr(merged['sentiment_profile']):.3f}")
print(f"Insult: {merged['insult_vanilla'].corr(merged['insult_profile']):.3f}")
```

---

## ⚠️ Important Notes

### **File Naming Convention:**
- **Vanilla**: `vanilla_analysis_TIMESTAMP.csv`
- **Individual profiles**: `annotator_role_PROFILE-ID_TIMESTAMP.csv`
- **Combined**: `annotator_role_combined_TIMESTAMP.csv`
- **Timestamp format**: `YYYYMMDD_HHMMSS` (e.g., `20251202_150000`)

### **Evaluation Script Behavior:**
The original `evaluation_multiattribute.py` looks for files matching `multi_attribute_analysis_*.csv`. 

Your new annotator role files will NOT be picked up automatically because they use a different naming pattern.

**Solutions:**
1. Use `evaluate_multiple_files.py` (recommended)
2. Rename files to match pattern: `multi_attribute_analysis_PROFILE_TIMESTAMP.csv`
3. Manually specify file in evaluation script

---

## 🎯 Summary

**Vanilla Analysis:**
- ✅ 1 file: `vanilla_analysis_TIMESTAMP.csv` (200 comments)
- ✅ Easy to evaluate with standard `evaluation_multiattribute.py`

**Annotator Role Analysis:**
- ✅ 10 separate profile files (50 comments each)
- ✅ 1 combined file (500 rows total)
- ✅ Each profile can be evaluated individually
- ✅ Use `evaluate_multiple_files.py` for batch evaluation

**Benefits of Separate Files:**
- ✅ Can evaluate each profile independently
- ✅ Easy to focus on specific demographics
- ✅ Cleaner organization
- ✅ Still have combined file for aggregate analysis

**Next Steps:**
1. Run both analyzers
2. Use `evaluate_multiple_files.py` to evaluate all profiles
3. Compare results across profiles
4. Identify demographic effects
