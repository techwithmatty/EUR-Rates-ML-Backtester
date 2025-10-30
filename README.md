# EUR Rates ML Backtester

A machine learning–driven backtesting framework for EUR 10-year rates, designed to model, test, and evaluate trading signals with intraday TP/SL execution.

This project provides an object-oriented foundation for quantitative strategy research, blending econometric indicators, technical features, and systematic trading logic.

## Features
- pandas-ta indicators
- Walk-forward ML (Logistic Regression / Random Forest)
- Intraday TP/SL with first-hit execution
- Transaction cost modeling on execution day
- Performance metrics (Sharpe, Sortino, Calmar, Drawdown)
- Feature and TP/SL sweeps

## Quickstart
```powershell
py -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
