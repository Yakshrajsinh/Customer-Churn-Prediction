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


## Screenshots
<img width="1242" height="345" alt="Screenshot 2026-10-29 173747" src="https://github.com/user-attachments/assets/52e14061-dbc3-46eb-9c1a-01f51acb70f2" />
<img width="1247" height="753" alt="Screenshot 2026-10-29 173731" src="https://github.com/user-attachments/assets/f68acfd3-5bba-4b92-9615-b78270ff4f9f" />
<img width="620" height="768" alt="Screenshot 2026-10-29 173916" src="https://github.com/user-attachments/assets/3dc55f71-377a-42a6-a485-797bea3c2269" />
<img width="515" height="840" alt="Screenshot 2026-10-29 173829" src="https://github.com/user-attachments/assets/5ffeb1b9-2f8c-4560-8cab-0cd08e33da1d" />
<img width="487" height="665" alt="Screenshot 2026-10-29 174109" src="https://github.com/user-attachments/assets/ed63e0a7-8ed5-4e79-8673-7d176d96a9a6" />
