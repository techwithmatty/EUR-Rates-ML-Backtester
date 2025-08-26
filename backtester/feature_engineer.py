import numpy as np
import pandas as pd
import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, short_period: int = 5, long_period: int = 10, avg_n_days: int = 5):
        self.short = short_period
        self.long = long_period
        self.avg_n = avg_n_days

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Momentum
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=self.long)
        df["ADX"] = adx[f"ADX_{self.long}"]
        df["ADX_chng"] = df["ADX"].diff()

        df["CCI"] = ta.cci(df["High"], df["Low"], df["Close"], length=self.long)
        df["CCI_chng"] = df["CCI"].diff()

        df["RSI"] = ta.rsi(df["Close"], length=self.long)
        df["RSI_chng"] = df["RSI"].diff()

        # Overlays
        df["SMA"] = ta.sma(df["Close"], length=self.long)
        df["SMA_chng"] = df["SMA"].diff()

        df["EMA"] = ta.ema(df["Close"], length=self.long)
        df["EMA_chng"] = df["EMA"].diff()

        df["KAMA"] = ta.kama(df["Close"], length=self.short)
        df["KAMA_chng"] = df["KAMA"].diff()

        psar = ta.psar(df["High"], df["Low"], df["Close"])
        left = psar.filter(like="PSARl_").iloc[:, 0]
        right = psar.filter(like="PSARs_").iloc[:, 0]
        df["SAR"] = left.combine_first(right)
        df["SAR_chng"] = df["SAR"].diff()

        # Volatility
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.long)
        df["ATR_chng"] = df["ATR"].diff()
        df["ATR_short"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.short)

        df["TRANGE"] = ta.true_range(df["High"], df["Low"], df["Close"])
        df["TRANGE_chng"] = df["TRANGE"].diff()

        # Extras
        if "YIELD_CHANGE" not in df.columns:
            df["YIELD_CHANGE"] = df["Close"].diff()
        df["30D_VOL"] = df["YIELD_CHANGE"].rolling(30).std()
        df["SAR_close_diff"] = df["SAR"] - df["Close"]

        df["Target_MA"] = df["Close"].rolling(self.avg_n).mean().shift(-self.avg_n)
        df["Target"] = np.where(df["Target_MA"] > 1, 1, np.where(df["Target_MA"] < -1, -1, 0))

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df