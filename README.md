# Donor Propensity Prediction Model | Advance U Internship

An end-to-end machine learning project that predicts **which constituents are most likely to give** and **which are most likely to respond to fundraising appeals**, built during my internship with Advance U. The models compare Random Forest and XGBoost classifiers and identify the factors that most strongly drive giving behavior, so fundraising teams can prioritize outreach.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)

---

## Table of Contents
- [Overview](#overview)
- [Data and Privacy Notice](#data-and-privacy-notice)
- [Approach](#approach)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Skills Demonstrated](#skills-demonstrated)
- [Contact](#contact)

---

## Overview

Advancement and fundraising teams have limited time and budget, so knowing *who* to contact and *how* matters. This project builds propensity models on constituent records (demographics, relationship types, education, engagement, and appeal history) to answer three questions:

1. **Who is likely to give?** A binary classifier predicting giving propensity.
2. **Who is likely to respond to appeals?** A classifier built on appeal history and solicitation channels (telemarketing, direct mail, email, web, crowdfunding).
3. **How much do results depend on age-related features?** A "seed" model run without acquisition age to test how well the model holds up without the strongest predictor.

## Data and Privacy Notice

**The dataset is not included in this repository.** It contains sensitive, non-public information about real individuals, and it has been withheld to protect their privacy.

This repo shares the notebook to demonstrate my methodology, code, and analytical process. Because the data is not provided, the notebook cannot be re-run end to end as-is. The saved cell outputs (charts, confusion matrices, and metrics) show the results from the original run.

## Approach

1. **Data preparation:** cleaned and merged constituent and giving records; encoded categorical variables and constituent flags (e.g., alumni, parent, friend, faculty/staff, degree types); aggregated giving amounts and responded-appeal counts per individual.
2. **Exploratory analysis:** examined age and acquisition-age distributions with pie charts and used a radar chart to compare average prevalence of model features.
3. **Giving propensity models:** trained a Random Forest (1,000 trees) and an XGBoost classifier, then tuned XGBoost with `GridSearchCV` using recall as the scoring metric.
4. **Appeal response models:** repeated the Random Forest and tuned XGBoost workflow with appeal channels added as features (3-fold CV across 27 parameter combinations, 81 fits; best parameters: learning rate 0.01, max depth 3, 50 estimators).
5. **Seed model:** retrained without acquisition age to measure how much performance depends on it.
6. **Evaluation and interpretation:** compared accuracy, precision, recall, and F1 using confusion matrices, and examined feature importance for every model.

## Results

### Giving propensity

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest (1,000 trees) | 91.9% | 90.0% | 92.5% | 91.2% |
| XGBoost (GridSearch, recall-optimized) | 90.6% | 85.0% | 96.3% | 90.3% |

Random Forest delivered the best overall balance (highest accuracy, precision, and F1). The tuned XGBoost model caught the most true givers (highest recall) at the cost of more false positives, which may be preferable if missing a likely donor is costlier than an extra contact.

### Appeal response

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest (1,000 trees) | 99.9% | 99.9% | 99.97% | 99.94% |
| XGBoost (GridSearch) | 99.9% | 99.9% | 100% | 99.93% |

These near-perfect scores reflect **severe class imbalance** (roughly 99.8% of records are in the positive class), so accuracy is misleading here. See [Limitations](#limitations-and-next-steps).

### Seed model (without acquisition age)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 75.6% | 73.0% | 73.3% | 73.2% |

Removing acquisition age dropped performance by roughly 16 points of accuracy, showing how heavily the models lean on that single feature.

## Key Findings

- **Age features dominate.** Acquisition age and current age were the top predictors across nearly every Random Forest and XGBoost model.
- **Relationship type matters.** Flags such as *Friend*, *Current Parent*, and *Past Parent* ranked next in the giving models.
- **Channel matters for appeals.** Telemarketing was the strongest channel predictor of appeal response, followed by direct mail; being a current employee and attending events also contributed.
- **Constituents were most often acquired in their 20s and 30s.** About 36% were acquired at ages 21-30 and 28% at 31-40, which helps explain why acquisition age is so informative.

## Limitations and Next Steps

- **Class imbalance in the appeal model.** With so few negative cases, the tuned XGBoost model predicted nearly everything as positive. A stronger approach would use class weighting or resampling, evaluate with precision-recall AUC, and compare against a baseline.
- **Possible feature leakage.** Acquisition age is the dominant feature and may encode information tied closely to the outcome. The seed model explores this, and I would investigate it further before deployment.
- **Next steps:** probability calibration, decile/lift analysis to quantify business impact, SHAP-based explanations, and a held-out time-based validation split.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Environment | Jupyter Notebook |
| Data Handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | scikit-learn (Random Forest, GridSearchCV), XGBoost |

## Repository Structure

```
.
├── Cleaned up Project code.ipynb   # Full analysis: prep, EDA, modeling, evaluation
└── README.md                       # Project documentation
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/LalithaHasaP/Propensity-Prediction-Model---Advance-U-Internship.git
   cd Propensity-Prediction-Model---Advance-U-Internship
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook "Cleaned up Project code.ipynb"
   ```
4. **Supply your own data.** The original dataset is not included, so update the data-loading cells to point to a dataset with a similar structure (a binary giving target plus constituent flags, age fields, and appeal/channel counts).

## Skills Demonstrated

- Binary classification and predictive modeling (Random Forest, XGBoost)
- Hyperparameter tuning with cross-validated grid search
- Model evaluation with confusion matrices, precision, recall, and F1
- Feature importance analysis and business interpretation
- Exploratory data analysis and data visualization
- Critical assessment of model limitations (class imbalance, leakage)
- Responsible handling of sensitive data

## Contact

**[Your Name]**
- GitHub: [@LalithaHasaP](https://github.com/LalithaHasaP)
- LinkedIn: [add your link]
- Email: [add your email]
