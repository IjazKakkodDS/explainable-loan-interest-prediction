
# Interpretable ML for Loan Pricing: Understanding What Drives Interest Rates in Credit Risk

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-EDA-lightgrey?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Modeling-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Core_Model-FF7043?logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-XAI_Global%2FLocal-00C49A?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI_App-ff4b4b?logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-In_Progress-lightgrey?logo=flask&logoColor=black)
![GitHub Actions](https://img.shields.io/github/actions/workflow/status/IjazKakkodDS/explainable-loan-interest-prediction/ci.yml?label=CI&logo=githubactions&style=flat-square)

This project provides a complete solution for understanding the key drivers of loan interest rates using advanced machine learning techniques integrated with explainable AI methods. It focuses on identifying and interpreting feature impact on interest rate decisions while ensuring predictive strength and transparency.

---

## Table of Contents

* [Project Background &amp; Motivation](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#project-background--motivation)
* [Problem Statement](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#problem-statement)
* [Objectives](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#objectives)
* [Impact](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#impact)
* [Key Contributions](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#key-contributions)
* [Tech Stack](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#tech-stack)
* [Data Engineering &amp; Preprocessing](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#data-engineering--preprocessing)
* [Methodological Framework](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#methodological-framework)
  * [Phase 1: Data Preparation &amp; EDA](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#phase-1-data-preparation--eda)
  * [Phase 2: Model Development &amp; Evaluation](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#phase-2-model-development--evaluation)
* [Explainability with SHAP](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#explainability-with-shap)
* [Results &amp; Insights](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#results--insights)
* [Trade-offs and Improvements](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#trade-offs-and-improvements)
* [Deployment Strategy](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#deployment-strategy)
* [Further Improvements &amp; Recommendations](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#further-improvements--recommendations)
* [Repository Structure](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#repository-structure)
* [How to Run Locally](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#how-to-run-locally)
* [Contact Information](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#contact-information)
* [License](https://chatgpt.com/c/67f6ca9d-b8e0-8004-a9f4-c258f1902fac#license)

---

## Project Background & Motivation

Financial institutions must balance accuracy in pricing loans with the need for transparency. Traditional risk models are giving way to machine learning algorithms that are highly predictive yet often work as “black boxes.” This project addresses these challenges by comparing diverse modeling techniques on both raw and engineered data while integrating explainability methods.

---

## Problem Statement

The challenge is to predict interest rates accurately while making the decision-making process interpretable for regulatory and financial transparency. This project tackles that by combining model performance with explainable AI.

---

## Objectives

* Collect and preprocess a decade-long dataset of loan applications.
* Build and evaluate ML models including Linear Regression, Decision Trees, Random Forest, XGBoost, and FNN.
* Apply feature selection (LassoCV, XGBoost importance).
* Integrate SHAP for global/local interpretability.
* Assess trade-offs using RMSE, MAE, R².
* Create deployment-ready app with UI and backend support.

---

## Impact

* Enhances interest rate forecasting accuracy and fairness.
* Builds trust through transparent explainability (SHAP).
* Enables deployment of ML solutions in production-grade environments.
* Useful for risk analysts, regulators, and business teams.

---

## Key Contributions

* 🔍 Dual-dataset analysis: raw vs engineered feature impact
* 🧠 Multiple ML model comparisons with XAI integration
* ✅ SHAP analysis for global & local explanations
* 🚀 Streamlit UI for rapid interaction and demo
* ⚙️ Flask backend planned for full-stack deployment

---

## Tech Stack

* **Languages:** Python 3.8+
* **Data Libraries:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **ML:** Scikit-learn, XGBoost, TensorFlow
* **Explainability:** SHAP
* **UI (Live):** Streamlit
* **Backend (Planned):** Flask API
* **Deployment:** Render, GitHub Actions

---

## Data Engineering & Preprocessing

* Engineered dataset: missing value handling, outlier removal, Z-score normalization
* Raw dataset: minimal transformation for baseline model
* EDA: Levene’s, Kruskal–Wallis, Chi-Square, correlation heatmaps

---

## Methodological Framework

### Phase 1: Data Preparation & EDA

* 📊 Data source: LendingClub loan applications (Kaggle)
* 🔧 Preprocessing: outlier detection (Tukey, Isolation Forest), imputation strategies
* 🛠️ Feature Engineering: categorical mapping, ratio features

### Phase 2: Model Development & Evaluation

* 🤖 Models: Linear Regression, Decision Tree, Random Forest, XGBoost, Feedforward Neural Network
* 📏 Metrics: RMSE, MAE, R²
* 🔎 Validation: GridSearchCV + k-fold CV
* 🧪 Feature Selection: LassoCV, XGBoost importance

---

## Explainability with SHAP

* 🧠 Global interpretability: SHAP summary plots
* 🔍 Local explanations: Individual prediction breakdowns
* ✅ XGBoost provided strongest balance of transparency and accuracy

---

## Results & Insights

| Model             | RMSE     | R² Score | Explainability | Notes                                  |
| ----------------- | -------- | --------- | -------------- | -------------------------------------- |
| Linear Regression | Moderate | Moderate  | ✅ High        | Baseline model                         |
| Decision Tree     | Moderate | Moderate  | ✅ Moderate    | Easy splits, risk of overfitting       |
| Random Forest     | Low      | High      | ⚠️ Limited   | Strong performance, less interpretable |
| **XGBoost** | ✅ Best  | ✅ Best   | ✅ Good        | Best balance                           |
| FNN               | ✅ Best  | ✅ Best   | ❌ Very Low    | High accuracy, black-box               |

---

## Trade-offs and Improvements

* Accuracy vs interpretability clearly outlined
* Future enhancements include:
  * Hybrid/distilled models
  * Real-time dashboards for SHAP
  * Expanded use cases for loan types

---

## Deployment Strategy

### ✅ Current: Streamlit Application

* Built with Streamlit for rapid UI + SHAP visualization
* Deployed via Render or Streamlit Cloud
* User inputs, prediction output, and explainability on one screen

---

### 🚧 In Development: Flask API (Enterprise-Ready)

* Backend modularization into REST API using Flask
* Will support front-end agnostic deployments
* Suitable for AWS, Docker, CI/CD integration

---

## Further Improvements & Recommendations

* Integrate SHAP dashboard for stakeholder interactivity
* Monitor for data/model drift
* Test alternate modeling frameworks (e.g., LightGBM, CatBoost)
* Build CI/CD pipelines with Docker & unit tests

---

## Repository Structure

```
explainable-loan-interest-prediction/
├── app/              # Streamlit app files
├── data/             # Raw and processed datasets
├── models/           # Saved model files
├── notebooks/        # Jupyter Notebooks for EDA & modeling
├── src/              # Python scripts for preprocessing
├── requirements.txt  # Dependency list
├── Procfile          # (For deployment)
├── .gitignore        # Git exclusions
└── README.md         # Project overview
```

---

## How to Run Locally

1. **Clone the Repository**

```bash
git clone https://github.com/IjazKakkodDS/explainable-loan-interest-prediction.git
cd explainable-loan-interest-prediction
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit App**

```bash
cd app
streamlit run app.py
```

---

## Contact Information

* **Email:** [ijazkakkod@gmail.com](mailto:ijazkakkod@gmail.com)
* **LinkedIn:** [linkedin.com/in/ijazkakkod](https://linkedin.com/in/ijazkakkod)
* **GitHub:** [github.com/IjazKakkodDS](https://github.com/IjazKakkodDS)

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://chatgpt.com/c/LICENSE) file for details.
