# IncomeClass: Adult Income Classification

**Objective:** Develop a Machine Learning model to predict whether an individual's annual income exceeds $50,000, based on demographic and employment data from the "Adult Income" dataset. Four gradient-boosting variants (XGBoost optimized for F2, XGBoost optimized for F1, CatBoost, LightGBM) were tuned through automated feature-combination experiments tracked in MLflow, then compared in a final analysis notebook on the same held-out test set, using **F2-Score** (recall-weighted) as the reference metric.

**Tools & Techniques:** Databricks (PySpark ETL pipeline + Jobs orchestration), Python, Scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler, SimpleImputer, RandomizedSearchCV, StratifiedKFold), XGBoost, LightGBM, CatBoost, Logistic Regression, VotingClassifier, StackingClassifier, MLflow (experiment tracking), pandas, Matplotlib.

**Keywords:** Machine Learning, Binary Classification, Feature Engineering, One-Hot Encoding, Gradient Boosting, Model Comparison, Bootstrap Confidence Intervals, Hyperparameter Tuning, Threshold Tuning, Imbalanced Classification, F1-Score, F2-Score, MLflow, Databricks, Income Prediction.

### 📊 Data & Feature Engineering
* **Dataset:** Adult Income, 48,842 records, split into train (29,305 / 60%), validation (9,768 / 20%), and test (9,769 / 20%) sets, stratified by target class (23.9% positive class).
* **Automated experimentation:** Random search over feature combinations (3 fixed core features + 3–8 candidates, i.e. 6–11 features per experiment) from a candidate space of 106,590 combinations. Each combination is trained through a Scikit-learn Pipeline and logged in MLflow; thousands of runs were evaluated across the four model families, each with its own `RandomizedSearchCV` hyperparameters.
* **Feature engineering:** Binary flags (`is_married`, `is_capital_gain`, `is_wife`, `is_native-country_missing`), interaction terms (`edu_x_hours`), and nominal categoricals (`marital-status`, `occupation`, `workclass`, `relationship`, `native-country`, `race`) encoded via **OneHotEncoder** (native `cat_features` for CatBoost). Ordinal encoding was deliberately dropped in favor of OHE, since these variables have no natural order — an ordinal scheme was found to impose false distance assumptions between categories and to distort feature-importance rankings.
* **Validation:** 5-fold Stratified Cross-Validation for model selection inside each experiment; decision thresholds tuned on the validation set (never on test).

### 🏁 Final Model Comparison
A final notebook, run as the last task of the Databricks pipeline, loads the best run of each experiment from MLflow, rebuilds the same 60/20/20 split, and re-scores all models on the **same test set and the same metric (F2)**, since the F1-objective experiment logs F1 in its CV metric. It also measures overfitting, ROC/PR curves, confusion matrices, feature importances, and a paired bootstrap (1,000 resamples) on the test set.

| Model | Threshold | **F2 (test)** | Recall | Precision | AUC-ROC | AUC-PR | Train–test F2 gap |
|---|---|---|---|---|---|---|---|
| **XGBoost** (F1-objective) | 0.39 | **0.792** | 0.905 | **0.527** | **0.912** | **0.780** | 0.021 |
| **LightGBM** (F2-objective) | 0.37 | 0.790 | 0.928 | 0.496 | 0.910 | 0.770 | 0.010 |
| **CatBoost** (F2-objective) | 0.36 | 0.788 | 0.932 | 0.488 | 0.909 | 0.768 | 0.010 |
| **XGBoost** (F2-objective) | 0.35 | 0.786 | **0.935** | 0.481 | 0.907 | 0.766 | 0.014 |

95% bootstrap CIs for F2 are ~±0.011 and overlap almost entirely (e.g. XGBoost F1: 0.781–0.803). The probability that the top model beats the runner-up is 0.67, and 0.95 against the last, so **the four models are statistically tied on F2**. All models stay below a 0.03 train–test gap, and the validation–test gap is ≤ 0.006, so the tuned thresholds generalize.

Both XGBoost variants were also benchmarked against `VotingClassifier` and `StackingClassifier` ensembles (with Logistic Regression); the standalone XGBoost matched or slightly outperformed the ensembles (Δ ≤ 0.005), so the simpler single-model pipeline was kept.

### 💡 Key Takeaways
* **The F2 gap between models is within noise; the real choice is the operating point.** XGBoost (F1) leads on threshold-independent ranking quality (AUC-ROC 0.912, AUC-PR 0.780) and precision (0.527), while LightGBM, CatBoost and XGBoost (F2) capture more positives (recall 92.8–93.5%) at the cost of ~300 additional false positives.
* **The training objective mattered less than the threshold.** An XGBoost trained for F1 and thresholded on validation for F2 matches the models trained directly for F2. The F2 curve is nearly flat between thresholds 0.30 and 0.50, so the operating point can be shifted toward recall or precision with almost no F2 loss.
* **No selection optimism:** test F2 is at or above the cross-validation score of the winning run, even though each best run was picked from thousands of combinations.
* **Feature importance concentration:** `is_married` accounts for 47.5% of importance in the F2-objective XGBoost, and `relationship` for 53.3% in the F1-objective XGBoost; the top 3 features explain 79.6% (F2) and 79.2% (F1), and the top 5 explain 90.2% and 89.5%. These marital-status variables are strongly collinear, so importance shifts between them across models. `education-num`, `occupation` and `is_capital_gain` are consistently relevant across all models.
* **Automated feature selection at scale:** Testing thousands of feature combinations via MLflow surfaced different but overlapping optimal feature sets per model, all centered on marital-status signals plus occupation, education and capital-gain features.

### 🎯 Recommended Use
* **Marketing / campaign targeting** (each false positive is a wasted contact): favor the **XGBoost (F1-objective)** model, with a higher threshold than the F2-optimal one to push precision further.
* **Screening with a human-in-the-loop, or eligibility checks where missing a case is costly:** favor **LightGBM / CatBoost / XGBoost (F2)** (recall ≥ 92.8%), where a false negative costs more than a false positive filtered by a reviewer.

### ⚠️ Limitations
* Differences of ~0.005 F2 between models are not statistically significant on a test set of ~9.8k rows.
* Best models use demographic attributes (`sex`, `native-country`, `is_wife`); a per-group performance and fairness review is recommended before any production use.

Update date: 12/10/2026
