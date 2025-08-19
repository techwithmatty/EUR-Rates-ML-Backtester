# Rates Quant Trading EUR 10s Backtester

An object-oriented backtesting framework for EUR 10-year rates with ML-driven signals and intraday TP/SL.

## Features
- pandas-ta indicators (no TA-Lib)
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