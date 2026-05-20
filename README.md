# AI-Powered Student Risk & Employability Predictor

> **Predictive Analytics · Supervised Machine Learning · Multiclass Classification**  
> Group 2 · Instructor: Maryanne Mwikali

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Understanding](#2-business-understanding)
3. [Team](#3-team)
4. [Project Structure](#4-project-structure)
5. [Datasets](#5-datasets)
6. [Methodology](#6-methodology)
7. [Key Findings](#7-key-findings)
8. [Models & Results](#8-models--results)
9. [Installation & Usage](#9-installation--usage)
10. [Streamlit App](#10-streamlit-app)
11. [Tableau Dashboard](#11-tableau-dashboard)
12. [Notebooks](#12-notebooks)

---

## 1. Project Overview

Higher education institutions face two compounding crises: persistently high student dropout rates and a widening gap between academic graduation and job-market readiness. This project addresses both by building an end-to-end AI-powered prediction platform that:

- **Predicts student dropout risk** before it's too late to intervene.
- **Assesses employability readiness** so career services can be proactive.
- **Classifies academic performance** into actionable risk tiers.
- **Recommends job categories** from raw resume text using NLP.

The system spans four datasets, four classification tasks, and delivers interpretable predictions via SHAP explainability — designed for use by academic advisors, career services, and university administrators.

---

## 2. Business Understanding

### Stakeholders

| Stakeholder | Primary Interest |
|---|---|
| University Administration | Institutional retention rates and reputation |
| Academic Advisors | Early-warning signals for at-risk students |
| Students | Self-assessment and career planning |
| Career Services | Aligning student skills with industry demand |
| Employers | Identifying job-ready candidates |

### Business Problem

Institutions lack systematic, data-driven tools to flag struggling students early and connect graduates with career pathways that match their actual skill profiles. Manual processes are reactive — by the time a student is identified as at-risk, intervention opportunities are already limited.

### Key Questions Answered

1. What are the strongest predictors of student dropout?
2. How strongly do first-semester grades correlate with final graduation outcomes?
3. What impact do financial factors — scholarships and debt — have on student success?
4. Can we accurately predict a student's employability level from their academic and technical skill profile?

---

## 3. Team

| Name | |
|---|---|
| **Ted Mwenda** 
| **Angela Wachira** 
| **Bobbin Bodo** 
| **Andrew Nyakiba** 
| **Mercy Chepkoech** 

---

## 4. Project Structure

```
├── data/
│   ├── data.csv                          # Raw dropout dataset (semicolon-delimited)
│   ├── Academic Risk Prediction DW.csv   # Raw academic risk dataset
│   ├── UECD_1200_dataset.csv             # Raw employability dataset
│   ├── Resume text.csv.gz                # Raw resume dataset (gzip + BOM encoded)
│   ├── df_dropout_cleaned.csv            # Cleaned (output of Notebook 1)
│   ├── df_academic_cleaned.csv
│   ├── df_employ_cleaned.csv
│   ├── df_resume_cleaned.csv
│   ├── df_dropout_preprocessed.csv       # SMOTE-balanced + scaled (output of Notebook 2)
│   ├── df_academic_preprocessed.csv
│   ├── df_employ_preprocessed.csv
│   └── df_resume_preprocessed.csv
├── models/
│   ├── dropout_best_model.pkl
│   ├── dropout_tuned_model.pkl
│   ├── academic_best_model.pkl
│   ├── academic_tuned_model.pkl
│   ├── employ_best_model.pkl
│   ├── employ_tuned_model.pkl
│   └── resume_best_model.pkl
├── outputs/                              # All saved charts and figures
├── notebooks/
│   ├── Student_Risk_Employability_Analysis-Intro_and_Cleaning.ipynb
│   ├── Student_Risk_Employability_Analysis-EDA.ipynb
│   └── Student_Risk_Employability_Analysis-Modelling.ipynb
├── app/                                  # Streamlit application
├── requirements.txt
└── README.md
```

---

## 5. Datasets

Four datasets were used across the four prediction tasks:

| Dataset | File | Purpose |
|---|---|---|
| UCI Dropout | `data.csv` | Predict Graduate / Enrolled / Dropout (semicolon-delimited) |
| Academic Risk DW | `Academic Risk Prediction DW.csv` | Predict High Performance / Average / At Risk |
| UECD Employability | `UECD_1200_dataset.csv` | Predict High / Medium / Low Readiness |
| Resume NLP | `Resume text.csv.gz` | Classify resume text into job categories |

> **Note on loading:** `data.csv` uses semicolons as delimiters (`sep=';'`). The resume file uses gzip compression with BOM encoding and requires a custom parser — see Notebook 1 for the `load_resume_csv()` helper function.

---

## 6. Methodology

The project follows the CRISP-DM framework across three sequential notebooks.

### 6.1 Data Cleaning (Notebook 1)

- Column names standardised to lowercase with underscores across all four datasets
- Corrected a misspelled column (`nacionality` → `nationality`) in the dropout dataset
- Dropped 36 fully-null rows and rows with missing target scores from the employability dataset
- Mode-imputed 18 missing values in `research_participation`
- Dropped 20 rows with missing `resume_text` from the resume dataset
- Created a binary `has_experience` flag from malformed `experience_years` values (values > 100 or ≤ 0 treated as no experience)
- All four cleaned datasets exported to `../data/` as CSV files

### 6.2 Exploratory Data Analysis (Notebook 2)

EDA was conducted separately for each dataset with targeted visualisations:

**Dropout dataset:** outcome distribution, age-at-enrolment histogram, first-semester grades by outcome, financial factor comparisons (scholarship & tuition status), outcome by debt status, combined financial pressure heatmap, gender split, curricular units approved per semester, and a full correlation matrix.

**Academic dataset:** grade distribution (A–D), grade split by gender, mean exam score by topic and school stage, study behaviour scatter plots (hours studied vs score, sleep vs score, absences vs score).

**Employability dataset:** employment competitiveness score distribution, GPA vs score scatter with trend line, top 8 features correlated with employability, and individual skill-score scatter plots.

**Resume dataset:** top 12 job category bar chart, most frequent keyword frequency bar chart.

### 6.3 Feature Engineering & Preprocessing (Notebook 2)

**Dropout dataset:**
- Engineered `total_units_approved` (Sem 1 + Sem 2 approved), `grade_improvement` (Sem 2 − Sem 1 grade), and `financial_pressure` flag (debtor = 1 AND tuition not up to date)
- Label-encoded the target column; StandardScaler applied to continuous features
- SMOTE applied to balance the three outcome classes

**Academic dataset:**
- Created `academic_status` target variable from grade letters (A/B → High Performance, C → Average, D → At Risk)
- Engineered `engagement_score` (mean of `raisedhands`, `visited_resources`, `announcements_view`, `discussion`) and `study_efficiency` (avg score / hours studied + 1)
- StandardScaler applied; SMOTE used to balance the three academic status classes

**Employability dataset:**
- Grouped raw skill columns into three composite scores: `tech_score_avg` (programming, tooling, domain knowledge, core subject), `soft_score_avg` (communication, teamwork, leadership, adaptability), and `career_score_avg` (career clarity, market awareness, learning motivation)
- Readiness level target created using 33rd/66th percentile thresholds on `employment_competitiveness_score`
- Classes were naturally balanced — no SMOTE required

**Resume dataset:**
- TF-IDF vectorisation (1,000 features, English stop-words removed)
- TruncatedSVD (LSA) compressed 1,000 word-features down to 10 latent topic components
- Final features: 10 topic components + `has_experience` binary flag
- Category label-encoded for model training

### 6.4 Modelling (Notebook 3)

For each of the four prediction tasks, models were trained, evaluated on a held-out 20% test set, and compared using weighted F1 score. The best model per domain was then hyperparameter-tuned.

**Models trained per task:**
- Dropout, Academic Risk, Employability: **Random Forest** vs **XGBoost**
- Resume NLP: **Logistic Regression** vs **Random Forest**

**Validation:**
- Stratified 80/20 train-test split (random seed 42 throughout)
- 5-fold Stratified Cross-Validation on the best model per domain to confirm stability
- RandomizedSearchCV (20 iterations, 5-fold CV) for hyperparameter tuning

**Explainability:**
- SHAP TreeExplainer applied to all three tree-based best models (Dropout, Academic, Employability)
- SHAP summary plots and global mean |SHAP value| bar charts generated for each domain

**Persistence:**
- Best and tuned models saved to `../models/` via `joblib` for deployment

---

## 7. Key Findings

### Dropout Prediction

- **First-semester grades are the single most powerful early signal.** Dropouts average 3.4 grade points lower than graduates in Semester 1 — flagging is possible as early as Week 14 of Year 1.
- **Units approved per semester matter more than raw grades.** Graduates approve ~5.9 units per semester; dropouts approve fewer than 2. A student approving zero or one unit in Semester 1 is a near-certain dropout signal.
- **Financial pressure compounds risk sharply.** Students who are both in debt and have overdue tuition face the highest dropout rates — the Debtor × Tuition heatmap is the most actionable single chart for institutional intervention.
- **Scholarship holders graduate at nearly double the rate of non-holders.** Students with tuition up-to-date graduate at ~82% vs ~35% for those who are behind.
- **Male students are disproportionately overrepresented among dropouts** (~58%) relative to their total enrolment share, despite female students showing stronger graduation rates overall.

### Academic Risk

- GPA and engagement score (a composite of class participation, resource visits, and discussions) are the dominant predictors of academic risk tier.
- Low-engagement students can be flagged before exam season, enabling early tutoring referrals.
- Study efficiency (score per hour studied) matters more than raw hours studied alone.

### Employability

- **GPA alone does not determine job readiness.** A student with a 3.8 GPA can score anywhere from 50 to 95 on the employment competitiveness scale — technical skills, career clarity, and soft skills add substantial independent predictive power.
- Tech skill composite score is the top SHAP driver of High Readiness predictions.
- Career clarity and market awareness add value beyond both GPA and raw skill scores.

### Resume NLP

- TF-IDF + LSA (10 topics) successfully captures latent semantic structure from raw resume text.
- The `has_experience` flag contributes meaningfully alongside topic features.
- Random Forest outperformed Logistic Regression on the multi-class NLP task.

---

## 8. Models & Results


### Dropout Prediction

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Random Forest | 0.7323 | 0.7325 |
| XGBoost | 0.7345 | 0.7336 |

**Features used:** `age_at_enrollment`, `admission_grade`, `total_units_approved`, `scholarship_holder`, `debtor`  
**Classes:** Graduate · Enrolled · Dropout

---

### Academic Risk Prediction

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Random Forest | 0.8939 | 0.8953 |
| XGBoost | 0.8876 | 0.8891 |


**Features used:** `gpa`, `absences`, `hours_studied`, `engagement_score`, `study_efficiency`  
**Classes:** High Performance · Average · At Risk

---

### Employability Readiness

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Random Forest | 0.5323 | 0.5320 |
| XGBoost | 0.4758 | 0.4776 |


**Features used:** `gpa`, `tech_score_avg`, `soft_score_avg`, `career_score_avg`  
**Classes:** High Readiness · Medium Readiness · Low Readiness

---

### Resume NLP Classification

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | 0.4225 | 0.3760 |
| Random Forest | 0.5123 | 0.4922 |

**Features used:** `topic_0` through `topic_9` (LSA components) + `has_experience`  
**Classes:** 25 job categories

---

### Cross-Validation Summary

All models were validated with 5-fold Stratified Cross-Validation to confirm that scores are stable and not an artefact of a lucky train/test split.

| Domain | Mean F1 (5-Fold) | Std |
|---|---|---|
| Dropout | 0.7430 | 0.0065 |
| Academic Risk | 0.8954 | 0.0028 |
| Employability | 0.6014 | 0.0265 |
| Resume NLP | 0.4946 | 0.0301 |

---

## 9. Installation & Usage

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or a Python virtual environment
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/student-risk-employability.git
cd student-risk-employability

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK resources (required for NLP pipeline — run once)
python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4','punkt']]"
```

### Running the Notebooks

Open Jupyter and run the three notebooks **in order**:

```bash
jupyter notebook
```

| Order | Notebook | What it does |
|---|---|---|
| 1 | `Intro_and_Cleaning.ipynb` | Loads raw data, cleans, exports cleaned CSVs |
| 2 | `EDA.ipynb` | Exploratory analysis, feature engineering, SMOTE, exports preprocessed CSVs |
| 3 | `Modelling.ipynb` | Trains models, evaluates, tunes, saves `.pkl` files |

> **Important:** Notebooks must be run in sequence. Each notebook reads the CSV outputs of the previous one from the `../data/` directory.

---

## 10. Streamlit App

The Streamlit application provides a real-time prediction interface for advisors and students.

```bash
cd app
streamlit run app.py
```

The app loads the tuned `.pkl` models from `../models/` and accepts user inputs to predict:
- Dropout risk (Graduate / Enrolled / Dropout)
- Academic performance tier (High Performance / Average / At Risk)
- Employability readiness (High / Medium / Low)
- Resume job category (from pasted resume text)

---

## 11. Tableau Dashboard

Interactive dashboards covering dropout risk, academic performance, employability readiness, and model performance are published on Tableau Public.

**📊 Tableau Public Link:** https://public.tableau.com/app/profile/ted.mutuma/viz/CapstoneProject_17791341105610/AIPOWEREDSTUDENTRISKANDEMPLOYABILITYPREDICTOR

The dashboards include:
- Student Dropout Risk Dashboard (outcome distribution, financial pressure heatmap, grade analysis, gender split)
- Academic Performance Risk Dashboard (grade distribution, GPA box plots, engagement and absence analysis)
- Employability Readiness Dashboard (readiness distribution, GPA vs competitiveness scatter, skill score comparisons, top resume categories)
- ML Model Performance Dashboard (F1 score comparisons, before/after tuning, feature importances)
- A 7-point Tableau Story narrating the full analysis journey

---

## 12. Notebooks

### Notebook 1 — Introduction & Cleaning
Covers project framing, business understanding, stakeholder mapping, raw data loading (including special parsing for semicolon-delimited and BOM-encoded files), exploratory data typing, missing value analysis, and systematic cleaning of all four datasets. Exports four cleaned CSVs.

### Notebook 2 — EDA & Feature Engineering
Covers 30+ targeted visualisations across all four datasets, engineered features (composite scores, binary flags, LSA topic components), class imbalance diagnosis, SMOTE balancing for Dropout and Academic datasets, StandardScaler normalisation, TF-IDF + TruncatedSVD for the resume NLP pipeline, and label encoding of all target variables. Exports four preprocessed, model-ready CSVs.

### Notebook 3 — Modelling
Covers model training (Random Forest, XGBoost, Logistic Regression), evaluation via classification reports and confusion matrices, model comparison summaries, 5-fold Stratified Cross-Validation, RandomizedSearchCV hyperparameter tuning (20 iterations, 5-fold), SHAP TreeExplainer explainability analysis (summary plots and global importance bar charts), before-vs-after tuning comparison, and serialisation of all best and tuned models to `../models/`.

---

*Group 2 · AI-Powered Student Risk & Employability Predictor · Instructor: Maryanne Mwikali*
