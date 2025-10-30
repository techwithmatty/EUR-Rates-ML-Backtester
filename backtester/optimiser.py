from typing import Dict, List
import itertools
import numpy as np
import pandas as pd
from dataclasses import replace
from .backtester import Backtester, BacktestConfig

class Optimiser:
    """Grid and feature-set optimisation routines."""

    @staticmethod
    def grid_tp_sl(backtester_factory, sl_grid, tp_grid):
        records = []
        for sl, tp in itertools.product(sl_grid, tp_grid):
            bt = backtester_factory(tp, sl)
            final_df, trades, _ = bt.run()
            total = final_df["TOTAL_PNL"].iloc[-1] if not final_df.empty else np.nan
            records.append((sl, tp, total))
        return pd.DataFrame(records, columns=["STOP_LOSS", "TAKE_PROFIT", "TOTAL_PNL"])\
                 .sort_values("TOTAL_PNL", ascending=False)

    @staticmethod
    def feature_combos(daily, intraday, base_cfg: BacktestConfig, test_features: List[str]) -> Dict[str, Dict]:
        combos: Dict[str, Dict] = {}
        for a, b in itertools.combinations(test_features, 2):
            cfg = replace(base_cfg, features=[a, b])
            final_df, trade_log, coef_df = Backtester(daily, intraday, cfg).run()
            total = final_df["TOTAL_PNL"].iloc[-1] if not final_df.empty else np.nan
            combos[f"{a} | {b}"] = {"df": final_df, "log": trade_log, "coef": coef_df, "total_pnl": total}
        return dict(sorted(combos.items(), key=lambda kv: kv[1]["total_pnl"], reverse=True))
