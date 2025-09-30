# Hate Speech Analysis Project

## Setup Instructions

### 1. Install Python (Recommended: Python 3.10+)
Make sure Python is installed on your system. You can check with:
```bash
python3 --version
```

### 2. Install Required Packages
If you do not use a virtual environment, install the required packages globally:
```bash
pip install pandas matplotlib datasets
```

### 3. Running the Python Scripts
To run any script, use:
```bash
python3 <script_name>.py
```
For example:
```bash
python3 select_random.py
python3 explore_classification.py
python3 Comment_annotator.py
```

### 4. (Optional) Using a Virtual Environment
To avoid conflicts with global packages, you can use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib datasets
```

### 5. Output Files
Some scripts will save results as CSV or PNG files in the project directory.

---

**Note:** If you encounter missing package errors, install them using `pip install <package_name>`.

## Script Descriptions

- `download_dataset.py`: Downloads the hate speech dataset from the source.
- `classifcation_changes.py`: Processes the dataset, merging binary columns into multi-class columns for easier analysis.
- `explore_classification.py`: Analyzes and visualizes the distribution, mean, and variance of target and annotator columns, saving plots.
- `Comment_annotator.py`: Explores the relationship between comments and annotators, showing how many annotators per comment and vice versa, with visualizations.
- `select_random.py`: Selects random comments with specified demographic and annotation features, saving the results to a CSV file.

---
