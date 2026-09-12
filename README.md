# IncomeClass: Adult Income Classification

**Objective:** Develop a Machine Learning model to predict whether an individual's annual income exceeds $50,000, based on demographic and employment data from the "Adult Income" dataset, using automated feature-combination experiments (tracked in MLflow) to maximize the **F2-Score** — balancing Recall and Precision, with emphasis on Recall.

### 📊 Data & Feature Engineering
* **Dataset:** Adult Income, 48,842 records, split into train (29,305 / 60%), validation (9,768 / 20%), and test (9,769 / 20%) sets, stratified by target class.
* **Automated experimentation:** Systematic search across thousands of feature combinations (3–11 features per experiment), each trained via a Scikit-learn Pipeline with an XGBoost classifier, logged and compared through MLflow.
* **Feature engineering:** Binary flags (`is_capital_gain`, `is_wife`, `is_native-country_missing`), ordinal encodings (`marital_status_ord`, `relationship_ord`, `workclass_ord`), and interaction terms (`edu_x_hours`), selected via cross-validated F2-Score.
* **Validation:** 5-fold Stratified Cross-Validation (F2-Score scorer) for model selection, followed by Train vs. Validation vs. Test consistency checks.

### 🤖 Model Performance
* **Model:** XGBoost Classifier (best of automated feature-combination search).
* **Winning feature set (11 features):** `age`, `education-num`, `hours-per-week`, `relationship_ord`, `marital_status_ord`, `is_capital_gain`, `capital-loss`, `is_wife`, `workclass_ord`, `fnlwgt`, `is_native-country_missing`.
* **Key metrics (test set):**
  * **F2-Score:** 0.78 (CV) / 0.784 (test)
  * **Recall (>50K):** 0.87–0.88
  * **Precision:** 0.54
  * **AUC-ROC:** 0.904
  * **AUC-PR:** 0.757

### 💡 Key Takeaways
* **Automated feature selection at scale:** Testing thousands of feature combinations via MLflow identified an 11-feature set that maximizes F2-Score, moving beyond manual feature curation.
* **Strong generalization:** Train/Validation/Test performance gap averaged ~0.005–0.01 in F2-Score, indicating minimal overfitting and high reliability across splits.
* **Feature importance concentration:** `marital_status_ord` alone accounts for 36% of feature importance; the top 3 features explain 81.5%, and the top 5 explain 90.3% of the model's predictive power.
* **Production decision:** XGBoost was selected over simpler models for its superior discrimination (AUC-ROC > 0.90) and stronger recall on the positive class, at the cost of moderate precision — an acceptable trade-off given the goal of minimizing missed high-income cases.
