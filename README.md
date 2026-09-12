# IncomeClass: Adult Income Classification

**Objective:** Develop a Machine Learning model to predict whether an individual's annual income exceeds $50,000, based on demographic and employment data from the "Adult Income" census dataset, prioritizing **recall maximization** of the positive class (>50K) to minimize false negatives.

### 📊 Data & Feature Engineering
* **Dataset:** Adult Income (census), 48,842 records and 15 columns, with imbalanced classes (≈76% `<=50K` vs. ≈24% `>50K`).
* **Cleaning:** Removal of duplicates; imputation of missing values in `workclass` (`unemployed`), `occupation` and `native-country` (`missing`), with creation of indicator flags.
* **Feature engineering:** Creation of binary variables (`is_capital_gain`, `is_capital_loss`) for columns with IQR≈0, and domain-knowledge ordinal encoding for `marital-status` and `relationship`.
* **Validation:** Stratified train/validation/test split (60%/20%/20%), with Grid Search (5-fold CV), threshold tuning, and Nested Cross-Validation to assess model robustness.

### 🤖 Model Performance
* **Models evaluated:** Logistic Regression, Random Forest.
* **Winning model:** Random Forest.
* **Key metrics (test set):**
  * **Recall (>50K):** 0.86 *(up from an initial recall of just 0.40)*
  * **ROC-AUC:** 0.856
  * **Precision:** 0.54 (threshold = 0.50)
  * *Comparison:* Logistic Regression achieved a recall of 0.84 and AUC of 0.882, with greater stability in Nested CV.

### 💡 Key Takeaways
* **Feature engineering was decisive:** Introducing `hours-per-week`, `is_capital_loss`, `marital_status_ord`, and `relationship_ord` raised recall from 0.74/0.79 to 0.84/0.86 and ROC-AUC from 0.82/0.84 to 0.88/0.89.
* **Production decision:** Random Forest was selected for maximizing recall on the class of interest — the central goal of the problem — despite Logistic Regression being slightly more stable under nested cross-validation.
* **Sensitive variables:** `gender` and `race` were analyzed but excluded from the final model due to low feature importance and the ethical considerations they raise in justifying income-related decisions.
