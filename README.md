# 🚘 Automobile Price Prediction with Ridge Regression

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Ridge%20Regression-ML-9cf" />
  <img src="https://img.shields.io/badge/GridSearchCV-Hyperparameter%20Tuning-orange" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" />
</p>

## 📌 Overview

This project builds a **Ridge Regression** model to predict automobile prices based on a wide range of vehicle specifications. Ridge Regression is used to handle **multicollinearity** among features — a common challenge in datasets where many variables (e.g., engine size, weight, horsepower) are correlated. The optimal regularization parameter (`alpha`) is selected via **GridSearchCV** with 5-fold cross-validation.

---

## 📂 Dataset

The dataset (`AutoMobileData.csv`) contains detailed specifications of automobiles along with their market prices.

**Key Features Include:**

| Category | Features |
|---|---|
| Identity | `symboling`, `make` (car brand) |
| Engine | `engine-type`, `engine-size`, `horsepower`, `peak-rpm`, `fuel-system`, `aspiration` |
| Dimensions | `wheel-base`, `length`, `width`, `height`, `curb-weight` |
| Transmission | `drive-wheels`, `num-of-doors`, `body-style`, `engine-location` |
| Performance | `city-mpg`, `highway-mpg`, `compression-ratio`, `bore`, `stroke` |
| **Target** | `price` |

---

## 🔍 Project Workflow

### 1. Data Loading & Initial Cleaning
- Loaded the CSV using **Pandas**
- Identified missing values encoded as `"?"` — replaced with `NaN`
- Visualized missing data with a **Seaborn heatmap**
- Converted relevant columns to numeric types using `pd.to_numeric()`
- Imputed missing `normalized-losses` values with the **column mean**
- Dropped remaining rows with null values

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for continuous features: `wheel-base`, `curb-weight`, `stroke`, `compression-ratio`, `price`
- **Pairplot** of numeric variables against `price`
- **Correlation Heatmap** to identify multicollinearity

**Key Observations from Heatmap:**
- `price` is strongly positively correlated with: `wheel-base`, `length`, `width`, `curb-weight`, `engine-size`, `horsepower`
- `price` is negatively correlated (~−0.70) with `city-mpg` and `highway-mpg`
- Many independent variables are highly intercorrelated → **multicollinearity** is present → Ridge Regression is appropriate

### 3. Feature Engineering
- Extracted `car_company` from the `make` column using **Regex**
- Dropped the original `make` and `symboling` columns
- Applied **One-Hot Encoding** (with `drop_first=True`) to all categorical variables using `pd.get_dummies()`

### 4. Data Preparation
- **Train/Test Split:** 70% training, 30% testing (`random_state=100`)
- Applied **StandardScaler** to normalize continuous numerical features (fit on train, transform on test)

### 5. Ridge Regression with GridSearchCV
- Defined an alpha search space:  
  `[0.0001, 0.001, 0.01, 0.05, 0.1, ..., 1000]`
- Ran **5-fold cross-validation** with `neg_mean_absolute_error` scoring
- Visualized training vs. test scores across alpha values
- **Best alpha = 15** selected based on CV results

### 6. Model Evaluation

| Metric | Value |
|---|---|
| R² Score | High (model explains most variance) |
| Mean Absolute Error | Low |
| Median Absolute Error | Low |
| Explained Variance Score | High |
| Max Error | Present (some outliers) |

Actual vs. Predicted scatter plot confirms good predictive alignment.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Pandas / NumPy | Data cleaning and manipulation |
| Matplotlib / Seaborn | EDA and result visualization |
| scikit-learn | Ridge Regression, GridSearchCV, StandardScaler, metrics |
| Regex (`re`) | Extracting car company names |
| Jupyter Notebook | Interactive development |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook
```bash
git clone https://github.com/MehmetErtass/Ridge_Regression.git
cd Ridge_Regression
jupyter notebook Ridge_Regression_.ipynb
```

---

## 📁 Project Structure

```
Ridge_Regression/
│
├── Ridge_Regression_.ipynb    # Main regression notebook
├── AutoMobileData.csv         # Dataset
└── README.md                  # Project documentation
```

---

## 📌 Why Ridge Regression?

> Standard Linear Regression struggles when features are highly correlated (multicollinearity), leading to unstable coefficients. **Ridge Regression** adds an L2 regularization penalty (`α * ||w||²`) that shrinks coefficients, preventing overfitting and producing more stable, generalizable predictions.

---

## 👨‍💻 Author

**Mehmet Ertaş**  
[![GitHub](https://img.shields.io/badge/GitHub-MehmetErtass-181717?logo=github)](https://github.com/MehmetErtass)
