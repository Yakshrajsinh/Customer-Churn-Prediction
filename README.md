# Telco Customer Churn — Supervised Learning

## Project overview
This project compares K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Support Vector Machine (SVM), and Decision Tree classifiers to predict customer churn using the IBM Telco Customer Churn benchmark dataset.

## Dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Place the CSV in the repository root or in `data/`. The dataset is not bundled in this starter package.

## Files
- `CustomerChurn_SupervisedLearning.ipynb` — EDA, preprocessing, model training/tuning, evaluation, error analysis, and final pipeline export.
- `churn_model.pkl` — generated when the notebook reaches Step 8.
- `summary_report.md` — report template; fill in measured results after running the notebook.
- `requirements.txt` — Python dependencies.

## Setup and run
Recommended: Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook CustomerChurn_SupervisedLearning.ipynb
```

Run every cell from top to bottom. Check that all outputs and plots are visible before submitting. The notebook creates `churn_model.pkl` in the current working directory.

## Evaluation notes
Recall for the churn class is a key metric because false negatives represent customers who leave without being identified for retention. Precision also matters because the team has limited call capacity. Compare precision, recall, F1 and ROC-AUC on the untouched stratified test set; do not select a model based on accuracy alone.


## GitHub submission checklist
- [ ] Run the notebook from top to bottom and save all outputs.
- [ ] Ensure `churn_model.pkl` exists and reloads successfully.
- [ ] Complete `summary_report.md` using the measured results, not assumed scores.
- [ ] Add project screenshots and the video link.
- [ ] Commit the notebook, model, report, requirements and README.
