# IncomeClass: Adult Income Classification

**Objective:** Develop a Machine Learning model to predict whether an individual's annual income exceeds $50,000, based on demographic and employment data from the "Adult Income" dataset, using automated feature-combination experiments (tracked in MLflow) to compare two optimization objectives — **F2-Score** (recall-weighted) and **F1-Score** (balanced) — and select the trade-off best suited to the business use case.

**Tools & Techniques:** Databricks (PySpark ETL pipeline), Python, Scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler, SimpleImputer, RandomizedSearchCV, StratifiedKFold), XGBoost, Logistic Regression, VotingClassifier, StackingClassifier, MLflow (experiment tracking), pandas, Matplotlib.

**Keywords:** Machine Learning, Binary Classification, Feature Engineering, One-Hot Encoding, Gradient Boosting, Ensemble Learning, Hyperparameter Tuning, Imbalanced Classification, F1-Score, F2-Score, MLflow, Databricks, Income Prediction.

### 📊 Data & Feature Engineering
* **Dataset:** Adult Income, 48,842 records, split into train (29,305 / 60%), validation (9,768 / 20%), and test (9,769 / 20%) sets, stratified by target class.
* **Automated experimentation:** Systematic random search across 1,500 feature combinations (3–11 features per experiment) out of a candidate space of 106,590 possible combinations, each trained via a Scikit-learn Pipeline with an XGBoost classifier, logged and compared through MLflow — run independently for each optimization objective.
* **Feature engineering:** Binary flags (`is_married`, `is_capital_gain`, `is_wife`, `is_native-country_missing`), interaction terms (`edu_x_hours`), and nominal categoricals (`marital-status`, `occupation`, `workclass`, `relationship`, `native-country`, `race`) encoded via **OneHotEncoder**. Ordinal encoding was deliberately dropped in favor of OHE, since these variables have no natural order — an ordinal scheme was found to impose false distance assumptions between categories and to distort feature-importance rankings.
* **Validation:** 5-fold Stratified Cross-Validation (objective-specific scorer) for model selection, followed by Train vs. Validation vs. Test consistency checks and per-model threshold tuning on the validation set.

### 🤖 Model Performance

Two XGBoost models were tuned and evaluated independently, each with its own `RandomizedSearchCV` hyperparameters, feature-combination search, and decision threshold — optimized for a different objective.

| | **F2-Score model** (recall-weighted) | **F1-Score model** (balanced) |
|---|---|---|
| **Winning feature set (11)** | `age`, `education-num`, `hours-per-week`, `is_married`, `occupation`, `relationship`, `is_capital_gain`, `native-country`, `capital-loss`, `is_wife`, `sex` | `age`, `education-num`, `hours-per-week`, `is_married`, `occupation`, `workclass`, `is_capital_gain`, `capital-loss`, `is_male`, `sex`, `is_native-country_missing` |
| **Decision threshold** | 0.35 | 0.66 |
| **Recall (>50K)** | 0.935 | 0.732 |
| **Precision** | 0.481 | 0.670 |
| **F2 / F1 (test)** | 0.786 | 0.699 |
| **AUC-ROC** | 0.907 | 0.912 |
| **AUC-PR** | 0.766 | 0.780 |
| Train/Val/Test gap | ~0.011 avg. | ~0.028 avg. |

Both variants were also benchmarked against `VotingClassifier` and `StackingClassifier` ensembles; in both cases the standalone XGBoost model matched or slightly outperformed the ensembles (Δ ≤ 0.005), so the simpler single-model pipeline was kept in production for each objective.

### 💡 Key Takeaways
* **Objective choice drives the precision/recall trade-off directly, not model quality.** Both models reach comparable, strong discrimination (AUC-ROC 0.90–0.91), but the F2 model trades precision for near-total recall (93.5%), while the F1 model roughly balances the two (73.2% recall / 67.0% precision). Which to deploy depends on whether missed high-income cases (false negatives) or wasted downstream effort on false positives is more costly for the use case.
* **Automated feature selection at scale:** Testing 1,500 feature combinations via MLflow per objective surfaced different — but overlapping — optimal feature sets for each metric, both centered on marital-status signals (`is_married`) plus occupation and income-adjacent numeric features.
* **Feature importance concentration:** `is_married` alone accounts for 47.5% of feature importance in the F2 model and 40.2% in the F1 model; the top 3 features explain 79.6% (F2) and 75.2% (F1), and the top 5 explain 90.2% (F2) and 88.0% (F1) of predictive power. This concentration is consistent across both objectives, confirming it reflects a genuine signal in the data rather than an artifact of the encoding or optimization metric.
* **Both models generalize well:** Train/Validation/Test performance gaps stayed under 0.03 for both objectives, indicating no significant overfitting in either configuration.
* **Kept as parallel experimental branches:** Both notebooks are maintained side by side rather than converged into a single "production" choice, since the right operating point depends on downstream business cost of false positives vs. false negatives — a decision outside the scope of the modeling pipeline itself.

* ### 🎯 Recommended Use by Objective

* **Marketing / campaign targeting** (contacting customers with likely income >$50K): favor the **F1 model**. Each false positive has a real cost (a wasted contact), and 67% precision is far more commercially defensible than 48% — you spend outreach budget on leads that are actually worth pursuing.
* **Screening with a human-in-the-loop, or eligibility checks where missing a case is costly** (e.g., benefit eligibility, triage followed by manual review): favor the **F2 model**. Here the cost of a false negative (missing a qualifying case) outweighs the cost of a false positive that a human reviewer will filter out downstream, so maximizing recall (93.5%) is the right trade-off.

Update date: 22/09/2026
