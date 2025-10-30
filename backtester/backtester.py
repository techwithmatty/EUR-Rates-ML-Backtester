from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from .model_wrapper import ModelWrapper
from .data_utils import DataUtils

@dataclass
class BacktestConfig:
    tp_level: float           # multiplier vs 30D_VOL for TP
    sl_level: float           # used only if vol_stop=False
    tc: float                 # transaction cost per side (bps)
    features: List[str]
    model_name: str = "lr"
    dynamic_sl: bool = True
    vol_stop: bool = True     # True => use ATR_short as SL base; else static sl_level
    n_train: int = 131

class Backtester:
    def __init__(self, daily_df: pd.DataFrame, intraday_df: pd.DataFrame, cfg: BacktestConfig):
        self.daily = daily_df.copy()
        self.intraday = intraday_df.copy()
        self.cfg = cfg
        self.intraday["_day"] = self.intraday["Time"].dt.strftime("%Y-%m-%d")

    def _apply_tc_daily(self, final_df: pd.DataFrame, tc: float):
        mask_tp_sl = final_df["ACTION_DETAIL"].isin(["TP", "SL"])
        final_df.loc[mask_tp_sl, "DAILY_PNL"] = final_df.loc[mask_tp_sl, "DAILY_PNL"] - tc
        oc = final_df["ACTION_DETAIL"].isin(["OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"])
        idx = final_df.index[oc].to_numpy()
        if idx.size:
            idx_exec = np.minimum(idx + 1, len(final_df) - 1)
            final_df.loc[idx_exec, "DAILY_PNL"] = final_df.loc[idx_exec, "DAILY_PNL"] - tc

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        cfg = self.cfg
        model = ModelWrapper(cfg.model_name)

        position = 0
        entries, exits, entry_dates, exit_dates = [], [], [], []
        action_detail, signals, delta = [], [], []
        coeffs_all = []

        for i in range(cfg.n_train, len(self.daily) - 1):
            hist = self.daily.iloc[i - cfg.n_train:i]
            X_train = hist[cfg.features]
            y_train = hist["Target"]
            model.fit(X_train, y_train)

            X_now = self.daily[cfg.features].iloc[[i]]
            curr_signal = int(model.predict(X_now)[0])
            signals.append(curr_signal)

            today = self.daily["Time"].iloc[i].strftime("%Y-%m-%d")
            intraday_day = self.intraday[self.intraday["_day"] == today].copy().reset_index(drop=True)

            coefs = model.coefficients(cfg.features)
            if coefs is not None:
                coeffs_all.append([f"{v:.3f}" for v in coefs])

            # Setup TP/SL levels for current day
            base_close = self.daily["Close"].iloc[i]
            tp = base_close + (self.daily["30D_VOL"].iloc[i] * cfg.tp_level) if position >= 0 else \
                 base_close - (self.daily["30D_VOL"].iloc[i] * cfg.tp_level)
            if cfg.vol_stop:
                base = self.daily["ATR_short"].iloc[i]
            else:
                base = cfg.sl_level
            sl = base_close - base if position >= 0 else base_close + base

            # If flat: open on tomorrow's open when signal is non-zero
            if position == 0:
                if curr_signal == 1:
                    position = 1
                    entries.append(self.daily["Open"].iloc[i + 1])
                    entry_dates.append(self.daily["Time"].iloc[i + 1])
                    action_detail.append("OPEN_LONG")
                    delta.append(position)
                elif curr_signal == -1:
                    position = -1
                    entries.append(self.daily["Open"].iloc[i + 1])
                    entry_dates.append(self.daily["Time"].iloc[i + 1])
                    action_detail.append("OPEN_SHORT")
                    delta.append(position)
                else:
                    action_detail.append("NONE")
                    entries.append(np.nan)
                    entry_dates.append(np.nan)
                    delta.append(0)
                exits.append(np.nan)
                exit_dates.append(np.nan)
                continue

            # If in a position, scan intraday for first hit
            is_long = position == 1
            if not intraday_day.empty:
                hit, j, tag = DataUtils.first_tp_sl_hit(intraday_day["Close"], tp=tp, sl=sl, is_long=is_long)
            else:
                hit, j, tag = (False, None, None)

            if hit:
                exit_price = intraday_day["Close"].iloc[j + 1] if (j + 1) < len(intraday_day) else intraday_day["Close"].iloc[j]
                exits.append(exit_price)
                exit_dates.append(self.daily["Time"].iloc[i])
                action_detail.append(tag)
                delta.append(0)
                position = 0

                # Optional reopen next day if signal persists
                if curr_signal == 1:
                    position = 1
                    entries.append(self.daily["Open"].iloc[i + 1])
                    entry_dates.append(self.daily["Time"].iloc[i + 1])
                    action_detail[-1] = f"{tag} -> OPEN_LONG"
                    delta[-1] = position
                elif curr_signal == -1:
                    position = -1
                    entries.append(self.daily["Open"].iloc[i + 1])
                    entry_dates.append(self.daily["Time"].iloc[i + 1])
                    action_detail[-1] = f"{tag} -> OPEN_SHORT"
                    delta[-1] = position
                else:
                    entries.append(np.nan)
                    entry_dates.append(np.nan)
            else:
                # No intraday hit: exit at EOD (flat overnight)
                exits.append(intraday_day["Close"].iloc[-1] if not intraday_day.empty else base_close)
                exit_dates.append(self.daily["Time"].iloc[i])
                action_detail.append("NONE")
                delta.append(position)
                position = 0

                if curr_signal == 1:
                    position = 1
                    entries.append(self.daily["Open"].iloc[i + 1])
                    entry_dates.append(self.daily["Time"].iloc[i + 1])
                    action_detail[-1] = "NONE -> OPEN_LONG"
                    delta[-1] = position
                elif curr_signal == -1:
                    position = -1
                    entries.append(self.daily["Open"].iloc[i + 1])
                    entry_dates.append(self.daily["Time"].iloc[i + 1])
                    action_detail[-1] = "NONE -> OPEN_SHORT"
                    delta[-1] = position
                else:
                    entries.append(np.nan)
                    entry_dates.append(np.nan)

        trades = pd.DataFrame({
            "ACTION_DETAIL": action_detail,
            "ENTRY": entries,
            "EXIT": exits,
            "ENTRY_DATE": pd.to_datetime(entry_dates),
            "EXIT_DATE": pd.to_datetime(exit_dates)
        })

        final_df = self.daily.copy()
        final_df["ACTION_DETAIL"] = trades["ACTION_DETAIL"].values
        final_df["DELTA"] = pd.Series(delta, index=final_df.index, dtype="float64")
        final_df["SIGNAL"] = pd.Series(signals, index=final_df.index, dtype="float64")

        # Base daily PnL from price changes
        final_df["DAILY_PNL"] = final_df["Close"].diff().fillna(0)

        # Overwrite entry/exit day PnL for intraday executions
        idx_by_date = pd.Series(final_df.index.values, index=final_df["Time"])
        for _, row in trades.dropna(subset=["ENTRY_DATE", "EXIT_DATE"]).iterrows():
            e_date = row["ENTRY_DATE"]; x_date = row["EXIT_DATE"]
            if pd.isna(e_date) or pd.isna(x_date): 
                continue
            e_idx = int(idx_by_date.get(e_date, np.nan))
            x_idx = int(idx_by_date.get(x_date, np.nan))
            if e_date == x_date:
                final_df.loc[x_idx, "DAILY_PNL"] = row["EXIT"] - row["ENTRY"]
            else:
                e_prev = final_df["Close"].iloc[e_idx - 1] if e_idx > 0 else final_df["Close"].iloc[e_idx]
                final_df.loc[e_idx, "DAILY_PNL"] = final_df["Close"].iloc[e_idx] - e_prev
                x_prev = final_df["Close"].iloc[x_idx - 1]
                final_df.loc[x_idx, "DAILY_PNL"] = row["EXIT"] - x_prev

        # Position-weighted PnL
        final_df["DAILY_PNL"] = final_df["DELTA"].shift(1).fillna(0) * final_df["DAILY_PNL"].fillna(0)

        # Transaction costs
        self._apply_tc_daily(final_df, cfg.tc)

        final_df["TOTAL_PNL"] = final_df["DAILY_PNL"].cumsum()

        coef_df = None
        if cfg.model_name == "lr" and len(coeffs_all) > 0:
            coef_df = pd.DataFrame(coeffs_all, columns=[f"coef_{i+1}" for i in range(len(coeffs_all[0]))],
                                   index=self.daily.index[cfg.n_train:cfg.n_train+len(coeffs_all)])

        return final_df, trades, coef_df