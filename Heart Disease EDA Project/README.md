# Heart Disease EDA & Preprocessing Project

Exploratory Data Analysis, data cleaning, and preprocessing performed on the
**Heart Failure Prediction** dataset (`heart.csv`), with the goal of preparing
a clean, model-ready dataset for predicting the presence of heart disease.

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Source file | `heart.csv` |
| Rows | 918 |
| Columns | 12 |
| Duplicate rows | 0 |
| Missing values | 0 |
| Target variable | `HeartDisease` (binary: 1 = disease present, 0 = no disease) |

**Raw columns:**

| Column | Type | Description |
|---|---|---|
| `Age` | numeric | Age of the patient |
| `Sex` | categorical | `M` / `F` |
| `ChestPainType` | categorical | `ATA`, `NAP`, `ASY`, `TA` |
| `RestingBP` | numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | numeric | Serum cholesterol (mg/dl) |
| `FastingBS` | binary | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `RestingECG` | categorical | Resting electrocardiogram results (`Normal`, `ST`, `LVH`) |
| `MaxHR` | numeric | Maximum heart rate achieved |
| `ExerciseAngina` | categorical | Exercise-induced angina (`Y` / `N`) |
| `Oldpeak` | numeric | ST depression induced by exercise relative to rest |
| `ST_Slope` | categorical | Slope of the peak exercise ST segment (`Up`, `Flat`, `Down`) |
| `HeartDisease` | binary | **Target** — 1 = heart disease, 0 = no heart disease |

**Target class balance:** 508 positive cases (heart disease) vs. 410 negative
cases — a reasonably balanced dataset (~55% / 45%).

---

## 2. Project Workflow

The notebook (`Heart_EDA_Project.ipynb`) is organized into two main stages:

### Stage 1 — Exploratory Data Analysis (EDA)
- Loaded the dataset and inspected shape (918 × 12), dtypes, and
  `describe()` summary statistics.
- Confirmed **zero duplicate rows** and **zero null values**.
- Checked target class balance (`HeartDisease`: 508 vs. 410).
- Plotted **histograms with KDE** for the numeric columns `Age`, `RestingBP`,
  `Cholesterol`, and `MaxHR` in a 2×2 grid.
- **Discovered a data quality issue:** both `Cholesterol` and `RestingBP`
  contained physiologically impossible **0 values** (a value of zero for
  either measurement is not medically plausible and almost certainly
  represents missing/unrecorded data encoded as 0). 172 rows had
  `Cholesterol == 0`.
- Used **count plots segmented by `HeartDisease`** to explore categorical
  relationships:
  - `Sex` vs `HeartDisease` → males show a visibly higher rate of heart
    disease in this dataset.
  - `ChestPainType` vs `HeartDisease`
  - `FastingBS` vs `HeartDisease`
- Used **box plots** and a **violin plot** to compare `Cholesterol` and `Age`
  distributions between disease/no-disease groups.
- Plotted a **correlation heatmap** of numeric features.

### Stage 2 — Data Cleaning & Preprocessing
- **Fixed invalid zero values:**
  - Replaced `Cholesterol == 0` with the **mean of all non-zero cholesterol
    values**, rounded to 2 decimals.
  - Replaced `RestingBP == 0` with the **mean of all non-zero resting BP
    values**, rounded to 2 decimals.
  - Re-plotted the histograms after the fix to confirm the distributions
    were corrected (no more spike at zero).
- Re-verified shape, columns, null counts, and dtypes after cleaning.
- **Encoded all categorical variables** using `pd.get_dummies(drop_first=True)`,
  producing the following one-hot encoded columns (baseline category dropped
  from each in parentheses):
  - `Sex_M` (baseline: F)
  - `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA` (baseline: ASY)
  - `RestingECG_Normal`, `RestingECG_ST` (baseline: LVH)
  - `ExerciseAngina_Y` (baseline: N)
  - `ST_Slope_Flat`, `ST_Slope_Up` (baseline: Down)
- Cast all columns to integer type for consistency.
- Applied **StandardScaler** to scale the numeric columns `Age`, `RestingBP`,
  `Cholesterol`, `MaxHR`, and `Oldpeak` to zero mean / unit variance.

---

## 3. Key Findings

- **Sex** shows a strong visual association with heart disease — males have
  a noticeably higher incidence in this dataset.
- **Cholesterol** and **RestingBP** both contained invalid `0` entries that
  needed imputation (mean of non-zero values) before they could be used
  meaningfully — a critical data-quality catch made during EDA.
- `ChestPainType` (particularly `ASY` — asymptomatic) and `ST_Slope` appear
  visually associated with heart disease risk based on the count plots.
- `Age` and `Cholesterol` distributions differ noticeably between the
  disease and no-disease groups, as shown by the box/violin plots.
- The target class is reasonably balanced (55% positive / 45% negative),
  so accuracy alone should be a reasonable — though not the only — metric
  for any downstream classification model.

---

## 4. Project / File Structure

```
.
├── Heart_EDA_Project.ipynb   # Main analysis notebook
├── heart.csv                 # Raw dataset (must be placed alongside notebook)
└── README.md                 # This file
```

---

## 5. Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 6. How to Run

1. Place `heart.csv` in the same directory as the notebook.
2. Launch Jupyter:
   ```bash
   jupyter notebook Heart_EDA_Project.ipynb
   ```
3. Run all cells top to bottom. The notebook will produce:
   - Distribution plots for numeric features (before and after cleaning)
   - Count plots of categorical features segmented by `HeartDisease`
   - Box/violin plots comparing `Age` and `Cholesterol` across target classes
   - A correlation heatmap
   - A cleaned, one-hot encoded, and scaled dataframe (`df_encode`),
     ready for machine learning

---

## 7. Possible Next Steps

- Add a formal statistical feature-selection step (e.g. Pearson correlation
  and/or Chi-squared tests against `HeartDisease`, as done in the companion
  Insurance EDA project) to quantitatively rank feature importance.
- Train classification models (Logistic Regression, Random Forest,
  Gradient Boosting) on `df_encode` to predict `HeartDisease`.
- Evaluate with precision/recall/F1 and ROC-AUC, not just accuracy, given
  the clinical context where false negatives (missed disease) are costly.
- Consider outlier treatment for `Cholesterol` and `Oldpeak`, which show
  skew/outliers in the box plots.
- Validate the zero-value imputation strategy against domain knowledge or
  compare against alternative approaches (e.g. median imputation, or
  imputation conditioned on other features).
