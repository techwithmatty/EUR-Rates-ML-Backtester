from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Performance:
    @staticmethod
    def scorecard(final_df: pd.DataFrame, trade_log: pd.DataFrame, N: int = 252, rf_bps: float = 0.0) -> Dict[str, float]:
        df = final_df.copy()
        total_pnl = df["DAILY_PNL"].sum()
        years = max((df["Time"].iloc[-1] - df["Time"].iloc[0]).days / 365.25, 1e-9)

        mean_daily = df["DAILY_PNL"].mean()
        ann_pnl = mean_daily * N
        daily_std = df["DAILY_PNL"].std()
        ann_std = daily_std * np.sqrt(N)

        equity = df["DAILY_PNL"].cumsum()
        runmax = equity.cummax()
        dd = equity - runmax
        max_drawdown = dd.min()
        md_end_idx = dd.idxmin()
        md_start_idx = equity.loc[:md_end_idx].idxmax()
        md_duration = (df.loc[md_end_idx, "Time"] - df.loc[md_start_idx, "Time"]).days

        sharpe = (ann_pnl - rf_bps) / ann_std if ann_std > 0 else np.nan
        neg = df.loc[df["DAILY_PNL"] < 0, "DAILY_PNL"]
        sortino = (ann_pnl - rf_bps) / (neg.std() * np.sqrt(N)) if len(neg) and neg.std() > 0 else np.nan
        calmar = (ann_pnl / abs(max_drawdown)) if max_drawdown != 0 else np.nan

        long_trades  = trade_log["ACTION_DETAIL"].str.contains("OPEN_LONG", na=False).sum() if not trade_log.empty else 0
        short_trades = trade_log["ACTION_DETAIL"].str.contains("OPEN_SHORT", na=False).sum() if not trade_log.empty else 0

        return {
            "total_pnl": float(total_pnl),
            "annualised_pnl": float(ann_pnl),
            "annualised_std": float(ann_std),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_days": int(md_duration),
            "no_long_trades": int(long_trades),
            "no_short_trades": int(short_trades),
            "years": float(years),
        }

    @staticmethod
    def plot_performance(name: str, df: pd.DataFrame):
        plt.figure(figsize=(15, 6))
        plt.plot(df["Time"], df["TOTAL_PNL"])
        plt.title(f"{name} – Strategy Performance")
        plt.xlabel("Date"); plt.ylabel("Performance (bps, cumulative)")
        plt.show()
