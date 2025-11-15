# US Market Lead–Lag Analysis (1978–2025) #


## Econometric foundations for forecasting U.S. market downturns ##
 ### Project Overview ###

This project analyzes the dynamic relationships between three major U.S. market indicators:

Transportation sector – DJT

Equity market – S&P 500

Housing market – U.S. Housing Price Index

The goal is to identify which market segment moves first during expansions and downturns and build a foundation for machine learning–based early-warning models in the next project stage.

This repository contains the full econometric pipeline used to uncover
lead–lag patterns, long-run equilibrium relationships, and shock transmission dynamics.

### Current Progress ###
 1. Data Collection 

Collected multi-source monthly datasets (1978–2025):

DJT (Yahoo Finance + historical extension via FRED)

S&P 500 (Yahoo Finance)

Housing Index (FRED)

 2. Data Cleaning & Alignment

Converted all datasets to end-of-month frequency

Standardized date formats

Harmonized column names

Merged into a unified DataFrame

Applied log transforms and stationarity preparation (log-differences)

 3. Exploratory Data Analysis

Log-level trend analysis

Monthly return behavior

Crisis period visualization (e.g., 2008, 2020)

 4. Econometric Analysis 

The project now includes:

ADF stationarity testing

Cross-correlation functions (CCF) for lead–lag detection

Granger causality tests (1–12 month lags)

Johansen cointegration test

VECM (Vector Error Correction Model)

Impulse Response Functions (IRF)

Forecast Error Variance Decomposition (FEVD)

These analyses provide a robust econometric basis for the upcoming predictive modeling phase.



These dynamics will directly inform feature engineering for the ML forecasting stage.

 Repository Structure
 ```
.
├── data/                         # Raw & processed datasets
├── Lead_Lag_Analysis_US_Markets.ipynb  # Main econometric notebook
└── README.md                     # Project overview
```


 ### Next Steps ###

The upcoming phase will focus on machine learning downturn forecasting, including:

Target construction (downturn indicators)

Feature engineering using econometric insights

Training ML models (linear, tree-based, sequence models)

Time-series cross-validation

Performance and interpretability analysis

Optional dashboard for visualization


### ---- ###

Author

Efe Derinçay
