# California Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Regression-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Status](https://img.shields.io/badge/Status-Active-success)

A comprehensive machine learning project for predicting California housing prices using advanced regression techniques, featuring extensive exploratory data analysis, feature engineering, and an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Models](#-models)
- [Results](#-results)
- [Technologies](#-technologies)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a complete machine learning pipeline to predict median house values in California districts. It combines rigorous statistical analysis, advanced feature engineering, and multiple state-of-the-art regression models to achieve optimal prediction accuracy.

### Key Highlights

- **20+ Machine Learning Models** including XGBoost, LightGBM, CatBoost, and ensemble methods
- **Interactive EDA Dashboard** built with Streamlit for real-time data exploration
- **Advanced Feature Engineering** with 20+ derived features
- **Model Optimization** using hyperparameter tuning and ensemble techniques
- **Production-Ready Code** with comprehensive preprocessing pipeline

---

## ✨ Features

### 🔍 Exploratory Data Analysis

- Comprehensive statistical analysis of housing features
- Distribution analysis with skewness and kurtosis metrics
- Correlation heatmaps and feature relationships
- Geographic visualization of housing prices
- Missing value analysis and data quality scoring
- Interactive filtering and data exploration

### 🛠️ Feature Engineering

- **Ratio Features**: rooms_per_household, bedrooms_per_room, population_per_household
- **Income-Based Features**: income_squared, income_cubed, income-rooms interaction
- **Age-Based Features**: is_new, is_old, age_squared
- **Density Features**: people_per_room, rooms_density
- **Geographic Features**: lat_lon_interaction, distance_to_center, distance_to_coast
- **Log Transformations**: for skewed distributions

### 🤖 Machine Learning

- **12+ Base Models**: Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, AdaBoost
- **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameters
- **Ensemble Methods**: Voting Regressor, Stacking Regressor, Weighted Voting
- **Cross-Validation**: K-Fold CV for robust performance estimation
- **Feature Selection**: Recursive feature elimination

### 📊 Interactive Dashboard

- Real-time data filtering and exploration
- Multiple visualization tabs (Overview, Distributions, Correlations, Geographic, etc.)
- Statistical tests and hypothesis testing
- Price prediction interface
- Data quality metrics and monitoring
- Export capabilities for reports

---

## 📁 Project Structure

```
ai2-project/
│
├── data/
│   └── 1553768847-housing.csv          # California housing dataset
│
├── notebooks/
│   ├── eda.ipynb                       # Exploratory Data Analysis
│   ├── feature-engineering.ipynb       # Feature Engineering Process
│   ├── model.ipynb                     # Model Training & Evaluation
│   └── catboost_info/                  # CatBoost training logs
│       ├── catboost_training.json
│       ├── learn_error.tsv
│       ├── test_error.tsv
│       └── time_left.tsv
│
├── models/
│   ├── best_model.pkl                  # Trained best model
│   ├── scaler.pkl                      # Feature scaler
│   └── model_metadata.pkl              # Model metadata and metrics
│
├── strm/
│   └── eda-streamlit.py                # Streamlit dashboard application
│
└── README.md                           # Project documentation
```

---

## 📊 Dataset

The project uses the **California Housing Dataset** containing information from the 1990 California census.

### Dataset Features

| Feature                  | Description                          | Type      |
| ------------------------ | ------------------------------------ | --------- |
| `longitude`              | Longitude coordinate                 | Float     |
| `latitude`               | Latitude coordinate                  | Float     |
| `housing_median_age`     | Median age of houses in the block    | Integer   |
| `total_rooms`            | Total number of rooms                | Integer   |
| `total_bedrooms`         | Total number of bedrooms             | Integer   |
| `population`             | Block population                     | Integer   |
| `households`             | Number of households                 | Integer   |
| `median_income`          | Median income (in tens of thousands) | Float     |
| `ocean_proximity`        | Distance to ocean (categorical)      | String    |
| **`median_house_value`** | **Median house value (Target)**      | **Float** |

### Dataset Statistics

- **Rows**: 20,640 housing districts
- **Missing Values**: ~207 in total_bedrooms (1%)
- **Geographic Coverage**: California state
- **Time Period**: 1990 Census data

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/m07amed25/AI2-Project-CIS.git
cd ai2-project
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install xgboost lightgbm catboost
pip install streamlit plotly
pip install jupyter notebook
```

### Required Libraries

```python
# Data Processing
pandas>=1.5.0
numpy>=1.23.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0

# Machine Learning
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.1.0

# Web Application
streamlit>=1.28.0

# Utilities
scipy>=1.9.0
pickle5>=0.0.11
```

---

## 💻 Usage

### 1. Exploratory Data Analysis

Open and run the EDA notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

This notebook provides:

- Dataset overview and statistics
- Missing value analysis
- Distribution visualizations
- Correlation analysis
- Outlier detection

### 2. Feature Engineering

Run the feature engineering notebook:

```bash
jupyter notebook notebooks/feature-engineering.ipynb
```

This creates:

- 20+ engineered features
- Transformed variables
- Feature importance analysis

### 3. Model Training

Execute the model training notebook:

```bash
jupyter notebook notebooks/model.ipynb
```

This notebook:

- Trains 12+ models
- Performs hyperparameter tuning
- Creates ensemble models
- Saves the best model

### 4. Interactive Dashboard

Launch the Streamlit application:

```bash
cd strm
streamlit run eda-streamlit.py
```

Then open your browser at `http://localhost:8501`

### Dashboard Features:

- **Overview Tab**: Dataset statistics and quality metrics
- **Distributions Tab**: Feature distributions with interactive histograms
- **Correlations Tab**: Correlation matrices and relationships
- **Geographic Tab**: Geographic price visualization
- **Feature Engineering Tab**: Engineered features analysis
- **Statistical Tests Tab**: Hypothesis testing
- **Price Prediction Tab**: Real-time predictions with trained model

---

## 🔬 Methodology

### 1. Data Preprocessing

- **Missing Value Treatment**: Median imputation for total_bedrooms
- **Outlier Handling**: IQR-based outlier detection and treatment
- **Categorical Encoding**: One-hot encoding for ocean_proximity
- **Scaling**: RobustScaler for numerical features (robust to outliers)

### 2. Feature Engineering Process

#### Ratio Features

```python
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households
```

#### Income Interactions

```python
income_squared = median_income ** 2
income_cubed = median_income ** 3
income_rooms_interaction = median_income * rooms_per_household
```

#### Geographic Features

```python
distance_to_center = sqrt((latitude - 34.0)² + (longitude + 118.0)²)
distance_to_coast = abs(longitude + 120.0)
lat_lon_interaction = latitude * longitude
```

### 3. Model Training Pipeline

1. **Train-Test Split**: 80-20 split
2. **Preprocessing**: Imputation + Scaling
3. **Base Model Training**: 12+ algorithms
4. **Cross-Validation**: 5-fold CV for stability
5. **Hyperparameter Tuning**: RandomizedSearchCV
6. **Ensemble Creation**: Voting, Stacking, Weighted Voting
7. **Model Selection**: Best model based on RMSE and R²
8. **Model Serialization**: Save best model and preprocessors

---

## 🤖 Models

### Base Models Implemented

| Category          | Models                                                   |
| ----------------- | -------------------------------------------------------- |
| **Linear Models** | Linear Regression, Ridge, Lasso, ElasticNet              |
| **Tree-Based**    | Decision Tree, Random Forest                             |
| **Boosting**      | Gradient Boosting, XGBoost, LightGBM, CatBoost, AdaBoost |
| **Ensembles**     | Voting Regressor, Stacking Regressor, Weighted Voting    |

### Hyperparameter Tuning

#### Random Forest

```python
{
    'n_estimators': [200, 300],
    'max_depth': [25, 30, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 0.3]
}
```

#### XGBoost

```python
{
    'n_estimators': [300, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
```

#### LightGBM

```python
{
    'n_estimators': [300, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [7, -1],
    'num_leaves': [31, 50]
}
```

### Ensemble Methods

1. **Voting Regressor**: Averages predictions from top 4 models
2. **Stacking Regressor**: Uses meta-learner (Ridge) on top model predictions
3. **Weighted Voting**: Weights based on inverse RMSE squared

---

## 📈 Results

### Model Performance (Expected Metrics)

| Model                 | Test RMSE ($) | Test R² | CV RMSE ($) |
| --------------------- | ------------- | ------- | ----------- |
| **Best Ensemble**     | ~$47,000      | ~0.83   | ~$48,000    |
| XGBoost (Tuned)       | ~$48,500      | ~0.82   | ~$49,000    |
| LightGBM (Tuned)      | ~$49,000      | ~0.81   | ~$50,000    |
| Random Forest (Tuned) | ~$50,000      | ~0.80   | ~$51,000    |
| CatBoost              | ~$51,000      | ~0.79   | ~$52,000    |

_Note: Actual results may vary depending on preprocessing and hyperparameters_

### Key Insights

1. **Income is the strongest predictor** of house values (correlation ~0.68)
2. **Ocean proximity significantly impacts** prices (coastal properties 40% higher)
3. **Age shows non-linear relationship** with price (newer and very old houses command premiums)
4. **Population density negatively correlates** with individual house values
5. **Ensemble methods provide best generalization** with lowest test errors

---

## 🛠️ Technologies

### Core Technologies

- **Python 3.8+**: Primary programming language
- **Jupyter Notebook**: Interactive development environment
- **Streamlit**: Web application framework

### Data Science Stack

- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions

### Visualization

- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive visualizations

### Machine Learning

- **scikit-learn**: ML algorithms and utilities
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Light gradient boosting machine
- **CatBoost**: Categorical boosting
