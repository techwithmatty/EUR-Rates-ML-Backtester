import numpy as np
import pandas as pd

class DataUtils:
    @staticmethod
    def normalize_ohlcv_cols(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
        """Standardize column names and types."""
        rename = {
            "time": "Time", "date": "Time",
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "volume": "Volume"
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        return df

    @staticmethod
    def first_tp_sl_hit(series: pd.Series, tp: float, sl: float, is_long: bool):
        """
        Return (hit, idx, tag) for first crossing of TP/SL in the price series.
        tag ∈ {'TP','SL'} when hit=True.
        """
        a = series.values
        if is_long:
            tp_idx = np.where(a >= tp)[0]
            sl_idx = np.where(a <= sl)[0]
        else:
            tp_idx = np.where(a <= tp)[0]
            sl_idx = np.where(a >= sl)[0]
        i_tp = int(tp_idx[0]) if tp_idx.size else None
        i_sl = int(sl_idx[0]) if sl_idx.size else None
        if i_tp is None and i_sl is None:
            return False, None, None
        if i_tp is None:
            return True, i_sl, "SL"
        if i_sl is None:
            return True, i_tp, "TP"
        return (True, i_tp, "TP") if i_tp < i_sl else (True, i_sl, "SL")