# Hate Speech Multi-Attribute Analysis Project Report

**Author:** Amine  
**Date:** December 2, 2025  
**Repository:** damino-code/hate_speech_analysis

---

## 📋 Executive Summary

This project developed and evaluated a **multi-attribute hate speech analysis system** using a locally-run LLaMA language model to analyze 9 different dimensions of online comments. The system was evaluated against human-annotated ground truth data, revealing significant challenges in automated hate speech detection across nuanced attributes.

**Key Finding:** Binary classification tasks (status, violence, genocide) achieved 77-98% accuracy, while multi-class attributes (sentiment, insult, humiliate) struggled with only 21-32% accuracy. Overall correlation between LLM and human judgments was weak (r < 0.35 for all attributes).

---

## 🎯 Project Objectives

1. **Develop a multi-attribute hate speech analyzer** beyond simple binary hate/not-hate classification
2. **Evaluate LLM performance** on 9 distinct attributes across different scales
3. **Compare automated predictions** with human-annotated ground truth (39,565 comments)
4. **Identify strengths and weaknesses** of current LLM-based approaches

---

## 🗂️ Dataset

- **Source:** Processed hate speech dataset with human annotations
- **Total Comments:** 39,565 unique comments with ground truth
- **Analyzed Subset:** 200 comments with LLM predictions
- **Annotation Coverage:** Multiple annotators per comment with demographic information

### **9 Attributes Analyzed:**

| Attribute | Scale | Type | Description |
|-----------|-------|------|-------------|
| **Sentiment** | 1-5 | Multi-class | Very negative to very positive |
| **Respect** | 1-2 | Binary | Not respectful vs respectful |
| **Insult** | 1-4 | Multi-class | Not insulting to very insulting |
| **Humiliate** | 1-3 | Multi-class | Not humiliating to humiliating |
| **Status** | 1-2 | Binary | Inferior status vs equal/superior |
| **Dehumanize** | 1-2 | Binary | Not dehumanizing vs dehumanizing |
| **Violence** | 1-2 | Binary | No violence vs calls for violence |
| **Genocide** | 1-2 | Binary | No genocide vs calls for genocide |
| **Attack/Defend** | 1-4 | Multi-class | Strongly defending to strongly attacking |

---

## 🛠️ Technical Implementation

### **Model Architecture**
- **Model:** Llama-3.2-3B-Instruct-Q4_K_M (GGUF format)
- **Parameters:** ~3 billion
- **Quantization:** 4-bit (Q4_K_M) for efficient inference
- **Framework:** llama-cpp-python for local execution
- **Size:** ~2-3GB on disk

### **System Components**

1. **`llamacpp_sentiment_analyzer.py`**
   - Multi-attribute comment analyzer
   - Structured prompt engineering for 9 attributes
   - Confidence scoring and fallback analysis
   - Pattern matching with regex for output parsing

2. **`evaluation_multiattribute.py`**
   - Comprehensive evaluation framework
   - Attribute-to-attribute matching (LLM vs Human)
   - Classification metrics (F1, accuracy, MAE)
   - Correlation analysis and visualization

3. **Supporting Scripts**
   - `download_dataset.py` - Dataset acquisition
   - `save_processed_dataset.py` - Data preprocessing
   - `explore_dataset.py` - Exploratory data analysis
   - `select_random.py` - Sampling for analysis

---

## 📊 Results & Evaluation

### **Overall Performance Metrics**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average Accuracy** | 0.539 | Moderate (54% correct) |
| **Average F1-Macro** | 0.330 | Poor (33% balanced performance) |
| **Average F1-Weighted** | 0.469 | Moderate (47% weighted by class) |
| **Average MAE** | 1.137 | Average error of ~1.1 scale points |

### **Attribute-by-Attribute Performance**

#### **🏆 Best Performing (Binary Attributes)**

| Attribute | Accuracy | F1-Macro | F1-Weighted | MAE | Correlation |
|-----------|----------|----------|-------------|-----|-------------|
| **Status** | 97.5% | 0.494 | 0.973 | 0.736 | -0.089 |
| **Genocide** | 89.0% | 0.471 | 0.838 | 0.848 | NaN |
| **Violence** | 77.0% | 0.536 | 0.717 | 0.852 | +0.177 |

**Analysis:** Binary classification tasks work relatively well, achieving high accuracy. However, low correlations suggest the model may be achieving accuracy through class imbalance (most comments don't contain violence/genocide) rather than true understanding.

#### **⚠️ Worst Performing (Multi-class Attributes)**

| Attribute | Accuracy | F1-Macro | F1-Weighted | MAE | Correlation |
|-----------|----------|----------|-------------|-----|-------------|
| **Sentiment** | 20.5% | 0.079 | 0.108 | 1.310 | -0.333 |
| **Humiliate** | 24.5% | 0.147 | 0.110 | 1.524 | +0.111 |
| **Insult** | 27.5% | 0.189 | 0.197 | 1.525 | +0.219 |

**Analysis:** Multi-class attributes show poor performance. The negative sentiment correlation (-0.333) indicates the model may be interpreting the scale backwards. High MAE values (>1.5) mean predictions are often off by 1-2 scale points.

### **Correlation Analysis**

All LLM vs Human correlations are **weak** (|r| < 0.35):

| Rank | Attribute | Correlation | Strength |
|------|-----------|-------------|----------|
| 1 | Sentiment | -0.333 | Weak (Negative) ⚠️ |
| 2 | Attack/Defend | +0.236 | Weak |
| 3 | Insult | +0.219 | Weak |
| 4 | Violence | +0.177 | Very Weak |
| 5 | Dehumanize | +0.125 | Very Weak |
| 6 | Humiliate | +0.111 | Very Weak |
| 7 | Status | -0.089 | Very Weak |
| 8 | Respect | -0.205 | Weak (Negative) |
| 9 | Genocide | NaN | N/A (no variance) |

**Critical Issue:** The negative correlations for sentiment and respect suggest systematic scale interpretation problems.

---

## 📈 Visualizations Generated

### **1. Correlation Matrix Heatmap**
- File: `multi_attribute_correlation_matrix.png`
- Shows: Pairwise correlations between all LLM and human attributes
- Insight: Weak diagonal correlations indicate poor LLM-human alignment

### **2. Scatter Plots (Top 3 Correlations)**
- File: `multi_attribute_scatter_plots.png`
- Shows: LLM vs Human ratings for sentiment, attack_defend, and insult
- Components:
  - **Dots:** Individual comment ratings (200 total)
  - **Red line:** Linear regression showing overall trend
  - **r value:** Correlation coefficient
- Insight: High scatter indicates inconsistent predictions

### **3. Detailed Metrics JSON**
- File: `multiattribute_metrics_20251125_160039.json`
- Contains: Complete numerical results for all 9 attributes
- Format: Structured JSON for programmatic analysis

---

## 🔍 Key Findings

### **Strengths**
1. ✅ **Binary tasks perform adequately** - Status (97.5%), Genocide (89%), Violence (77%)
2. ✅ **System architecture is scalable** - Can analyze multiple attributes simultaneously
3. ✅ **Local execution** - Privacy-preserving, no external API costs
4. ✅ **Comprehensive evaluation framework** - Multiple metrics provide full picture

### **Weaknesses**
1. ❌ **Poor multi-class performance** - Sentiment (20.5%), Humiliate (24.5%), Insult (27.5%)
2. ❌ **Weak correlations across all attributes** - None exceed |r| = 0.35
3. ❌ **Scale interpretation issues** - Negative correlations suggest backwards understanding
4. ❌ **High prediction variance** - Large scatter in visualizations
5. ❌ **Small model limitations** - 3B parameters may be insufficient for nuanced judgments

### **Root Cause Analysis**

| Issue | Likely Cause | Evidence |
|-------|--------------|----------|
| Negative sentiment correlation | Scale interpretation backwards | r = -0.333 |
| Poor multi-class accuracy | Insufficient model capacity | 3B params, 20-32% accuracy |
| High binary accuracy but low correlation | Class imbalance bias | 97.5% accuracy but r = -0.089 |
| Inconsistent predictions | Weak prompt engineering | High scatter in visualizations |

---

## 💡 Recommendations

### **Immediate Improvements (Quick Wins)**

1. **Fix Scale Interpretation**
   - Reverse sentiment scale in prompt or post-processing
   - Add explicit scale anchors with examples
   - Test with clarified prompts

2. **Upgrade Model**
   - Switch to Llama-3.1-8B-Instruct (8 billion parameters)
   - Larger model may better capture nuances
   - Available in current codebase

3. **Simplify Multi-class Tasks**
   - Convert 4-class insult to binary (insulting/not)
   - Convert 5-class sentiment to 3-class (negative/neutral/positive)
   - Reduce cognitive load on model

### **Long-term Improvements**

1. **Fine-tuning**
   - Fine-tune model on hate speech dataset
   - Use human annotations as training data
   - Expect significant accuracy improvements

2. **Prompt Engineering**
   - Add few-shot examples for each attribute
   - Provide explicit rating criteria
   - Use chain-of-thought reasoning

3. **Ensemble Methods**
   - Combine multiple models (LLaMA, Phi-3, etc.)
   - Weighted voting based on confidence
   - Reduce variance through averaging

4. **Active Learning**
   - Identify low-confidence predictions
   - Get human annotation for uncertain cases
   - Iteratively improve model

---

## 📁 Project Files

### **Core Analysis Files**
- `llamacpp_sentiment_analyzer.py` - Multi-attribute analyzer (877 lines)
- `evaluation_multiattribute.py` - Evaluation framework (417 lines)
- `processed_dataset.csv` - Ground truth data (39,565 comments)
- `multi_attribute_analysis_20251125_155100.csv` - LLM predictions (200 comments)

### **Results & Metrics**
- `multiattribute_metrics_20251125_160039.json` - Detailed metrics
- `multi_attribute_correlation_matrix.png` - Correlation heatmap
- `multi_attribute_scatter_plots.png` - Top 3 attribute comparisons

### **Supporting Files**
- `download_dataset.py` - Dataset acquisition
- `save_processed_dataset.py` - Data preprocessing
- `explore_dataset.py` - EDA scripts
- `compare_attributes.py` - Attribute comparison utilities
- `requirements.txt` - Python dependencies

---

## 🎓 Lessons Learned

1. **Small models struggle with nuance** - 3B parameters insufficient for complex multi-attribute classification
2. **Binary tasks are easier** - Clear yes/no decisions work better than scaled ratings
3. **Correlation matters more than accuracy** - High accuracy with low correlation suggests bias/imbalance
4. **Prompt clarity is critical** - Negative correlations indicate communication failures
5. **Evaluation must be comprehensive** - Multiple metrics reveal different aspects of performance

---

## 🚀 Next Steps

### **Phase 1: Quick Fixes (1-2 days)**
- [ ] Fix sentiment scale interpretation
- [ ] Test with Llama-3.1-8B model
- [ ] Re-run evaluation on full 200 comments

### **Phase 2: Prompt Optimization (1 week)**
- [ ] Add few-shot examples
- [ ] Implement chain-of-thought prompting
- [ ] A/B test different prompt formats

### **Phase 3: Model Enhancement (2-4 weeks)**
- [ ] Fine-tune on hate speech dataset
- [ ] Implement ensemble methods
- [ ] Expand to full 39,565 comment evaluation

### **Phase 4: Production (1-2 months)**
- [ ] Build API for real-time analysis
- [ ] Create web interface for annotation
- [ ] Deploy active learning pipeline

---

## 📚 References & Resources

- **Dataset:** Human-annotated hate speech corpus (39,565 comments)
- **Model:** Llama-3.2-3B-Instruct-Q4_K_M (Hugging Face)
- **Framework:** llama-cpp-python
- **Evaluation Metrics:** sklearn.metrics (F1, accuracy, MAE, correlation)
- **Visualization:** matplotlib, seaborn

---

## 🎯 Conclusion

This project successfully implemented and evaluated a multi-attribute hate speech analysis system, revealing significant challenges in automated content moderation. While binary classification tasks achieve reasonable accuracy (77-98%), multi-class attributes struggle (21-32%), and overall correlations with human judgment remain weak (|r| < 0.35). The negative sentiment correlation suggests fundamental prompt engineering issues, while the low multi-class performance points to model capacity limitations.

**Key Takeaway:** Current 3B parameter LLMs are insufficient for nuanced hate speech analysis across multiple dimensions. Upgrading to larger models (8B+), fixing scale interpretations, and implementing fine-tuning are critical next steps before production deployment.

---

**Project Status:** ✅ Phase 1 Complete (Analysis & Evaluation)  
**Recommendation:** Proceed to Phase 2 (Optimization) with model upgrade and prompt engineering  
**Estimated Impact:** 20-40% accuracy improvement with recommended changes
