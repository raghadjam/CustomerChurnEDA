# Customer Churn Analysis

Exploratory data analysis (EDA) pipeline for customer churn prediction. This script preprocesses customer data, handles missing values and outliers, standardizes features, and generates visualizations to uncover patterns linked to churn behavior.

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## Dataset

The script expects a file named `customer_data.csv` in the working directory with the following columns:

| Column | Type | Description |
|---|---|---|
| `Age` | Numeric | Customer age |
| `Income` | Numeric | Annual income |
| `Tenure` | Numeric | Months as a customer |
| `SupportCalls` | Numeric | Number of support calls made |
| `Gender` | Categorical | Customer gender |
| `ProductType` | Categorical | `0` = Basic, `1` = Premium |
| `ChurnStatus` | Binary | `0` = Stayed, `1` = Churned |

---

## Pipeline Overview

**1. Data Loading & Inspection** — Loads the CSV and prints a summary of shape, dtypes, and descriptive statistics.

**2. Missing Value Imputation** — Fills missing values in numeric columns with their median.

**3. Outlier Capping (IQR Method)** — Caps extreme values at 1.5× IQR above/below Q1/Q3.

**4. Feature Scaling** — Applies `StandardScaler` to all numeric columns (zero mean, unit variance).

**5. Visualizations** — Generates the following plots:
- Histograms for each numeric feature
- Count plots for categorical features
- Box plots: numeric features vs. churn status
- Grouped bar charts: gender & product type vs. churn
- Correlation heatmap of numeric features
- Churn breakdown by income range and tenure bins

---

## Usage

```bash
python Code.py
```

All plots are displayed inline via `plt.show()`. No files are saved to disk by default.

---

## Notes

- Scaling is applied per-column using separate `StandardScaler` fits. For model training, fit the scaler on training data only to avoid data leakage.
- `ChurnStatus` is included in the correlation matrix as a numeric target for reference.
- The `IncomeRange` and `ChurnLabel` columns are derived during visualization and not part of the original dataset.
