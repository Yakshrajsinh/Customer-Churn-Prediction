# Summary Report — Telco Customer Churn (Supervised Learning)

## 1. Business Problem & Dataset
A telecom retention team (Jio/Airtel-style) is losing subscribers every month and cannot tell who is about to leave. Winning back a lost customer costs far more (CAC) than keeping one (CLV), so the goal was a classifier that flags likely churners early. I used the IBM Telco Customer Churn dataset: 7,043 customers, 21 columns, binary target `Churn`. Overall, **26.5% of customers churned** (1,869 vs 5,174), so the data is moderately imbalanced.

## 2. Preprocessing & Imbalance Strategy
- Dropped `customerID`. The 11 blank `TotalCharges` values were all customers with `tenure = 0` (new signups); they were converted to numeric and imputed with the median.
- Engineered `tenure_group` (New/Mid/Senior/Loyal), `num_services` (count of six add-ons) and `AutoPay`.
- Binary columns mapped to 0/1, categorical columns one-hot encoded, numeric columns standardised.
- 80/20 stratified split (5,634 train / 1,409 test, `random_state=42`). **SMOTE was applied to the training set only**, balancing it from 4,139 / 1,495 to 4,139 / 4,139. The test set kept its original distribution.

## 3. Model Comparison (test set, Churn class)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Gaussian Naive Bayes** | 0.727 | 0.491 | **0.805** | 0.610 | 0.820 |
| KNN (k=11) | 0.702 | 0.464 | 0.789 | 0.584 | 0.804 |
| Decision Tree (depth=3) | 0.746 | 0.514 | 0.783 | 0.621 | 0.816 |
| SVM (C=0.1) | 0.746 | 0.514 | 0.778 | 0.619 | 0.828 |

SMOTE and `class_weight='balanced'` gave identical churn recall on the Decision Tree (0.783).

## 4. Recommendation
I saved **Gaussian Naive Bayes** as the final pipeline (`churn_model.pkl`) because it has the highest churn recall (0.805), catching 301 of the 374 churners in the test set, and it scores customers instantly. The trade-off is precision: only about 49% of flagged customers truly churn, still roughly twice the 26.5% base rate. The tuned Decision Tree is a strong alternative: its recall is only 2 points lower, its precision and F1 are higher, and it is fully explainable, so it suits teams that need to justify why a customer was flagged. Note that Naive Bayes' learned class priors are 0.5/0.5 because it trained on SMOTE-balanced data, so its probabilities are not calibrated to the real churn rate. Rank customers by score and cap outreach at the ~500 calls/day capacity.

## 5. Top Churn Signals & Actions
1. **Month-to-month contracts** churn at 42.7% vs 11.3% (one-year) and 2.8% (two-year), and this is the Decision Tree's root split (importance ≈ 0.70). *Action:* offer discounted annual or two-year upgrades.
2. **Short tenure**: customers in their first 12 months churn at 47.4%, falling to 9.5% after 49 months. *Action:* build onboarding and first-year check-in programmes.
3. **Fiber-optic internet and higher monthly charges** rank next in importance (≈ 0.12 and 0.04); test-set churners pay 72.8 per month on average (vs. 64.8 across all customers). *Action:* run a service-quality and value review for these customers.

**Error analysis:** the 73 missed churners (false negatives) look different from typical churners. Their average tenure is 34 months (vs 16 for all churners) and only 41% are month-to-month (vs 88%). The model misses long-tenure, contract-holding customers who leave anyway, so a separate watch-list for them is worth trying.

## 6. Next Steps
- Collect more behavioural data (usage trends, complaints, support calls) to catch the profile the model currently misses.
- Tune the decision threshold against the real cost of a retention offer, and calibrate probabilities.
- Try gradient boosting (XGBoost/LightGBM) and compare against these baselines.
- Deploy as a real-time scoring API, then monitor drift and campaign outcomes before scaling.
