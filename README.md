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
 Current Progress
1) Data Collection (1978–2025, monthly)

DJT (Yahoo Finance + historical extension)

S&P 500 (Yahoo Finance)

Housing Index (FRED)

2) Data Cleaning & Alignment

End-of-month alignment

Standardized date formats and column names

Unified dataset creation

Log transforms + log-differences for return-style analysis

3) Econometric & Statistical Diagnostics

Stationarity checks (ADF)

Cross-correlation / lead–lag exploration across multiple lags

Additional econometric tests (Granger, cointegration, VECM, IRF/FEVD) in the notebook

4) ML Downturn Modeling (Reality Check)

Multiple models tested (Ridge, RandomForest, XGBoost, LSTM)

Result: no strong predictive signal; performance is limited, especially under class imbalance

Ridge shows only a weak linear edge; non-linear models are near chance in this setup

5) Streamlit Diagnostic Dashboard

A Streamlit dashboard was built to communicate findings clearly:

Home: key diagnostics + top lag evidence

Market Lab: time-series + rolling correlation + crisis window shading

Lead–Lag Lab: interactive pair selection, lag range, best-lag KPI, top lags table

ML Reality Check: downturn rarity stats + Ridge confusion matrix + model summary

Dashboard artefacts are exported as CSV files from the notebook and loaded by Streamlit.

 Repository Structure
 ```plaintext
.
├── datas/                                 # Raw/processed datasets (project pipeline)
├── Lead_Lag_Analysis_US_Markets.ipynb      # Main notebook (econometrics + ML)
├── Dashboard/
│   ├── app.py                             # Streamlit diagnostic dashboard
│   ├── dashboard_datas/                   # Exported CSV artefacts for the dashboard
│   ├── dashboard_figures/                 # Report-ready figures (PNG)
│   └── Readme_dashboard_contract.txt      # Data/artefact contract notes
└── README.md

```
How to Run the Dashboard
From the project root:
```
streamlit run ./Dashboard/app.py
```

## Presentation
-https://drive.google.com/file/d/1K83PHi1vDpY7VzrGRYUQuKOwjYW1VARn/view?usp=sharing




Author

Efe Derinçay
