"""
FSupertrendStrategy – v2.1 (whipsaw-hardening + bug-fix)
"""

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, merge_informative_pair
from pandas import DataFrame


class FSupertrendStrategy(IStrategy):
    INTERFACE_VERSION: int = 3
    can_short = True

    # ===== unchanged core settings ============================================
    # Buy hyperspace params:
    buy_params = {
        "buy_m1": 1,
        "buy_m2": 1,
        "buy_m3": 2,
        "buy_p1": 11,
        "buy_p2": 7,
        "buy_p3": 21,
        "buy_supported_m": 7,
        "buy_supported_p": 13,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 5,
        "sell_m2": 1,
        "sell_m3": 1,
        "sell_p1": 9,
        "sell_p2": 11,
        "sell_p3": 12,
        "sell_supported_m": 1,
        "sell_supported_p": 16,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.077,
        "388": 0.05,
        "602": 0.024,
        "807": 0
    }

    # Stoploss:
    stoploss = -0.19

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.024
    trailing_only_offset_is_reached = True

    timeframe = "1h"
    startup_candle_count = 50

    informative_tf = "5m"

    # ===== hyperopt ranges ====================================================
    buy_m1 = IntParameter(1, 7, default=1)
    buy_m2 = IntParameter(1, 7, default=3)
    buy_m3 = IntParameter(1, 7, default=4)
    buy_p1 = IntParameter(7, 21, default=14)
    buy_p2 = IntParameter(7, 21, default=10)
    buy_p3 = IntParameter(7, 21, default=10)

    sell_m1 = IntParameter(1, 7, default=1)
    sell_m2 = IntParameter(1, 7, default=3)
    sell_m3 = IntParameter(1, 7, default=4)
    sell_p1 = IntParameter(7, 21, default=14)
    sell_p2 = IntParameter(7, 21, default=10)
    sell_p3 = IntParameter(7, 21, default=10)
    
    buy_supported_m = IntParameter(1, 7, default=1)
    sell_supported_m = IntParameter(1, 7, default=1)
    buy_supported_p = IntParameter(7, 21, default=14)
    sell_supported_p = IntParameter(7, 21, default=14)

    # -------------------------------------------------------------------------
    # Leverage
    # -------------------------------------------------------------------------
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return min(3.0, max_leverage)

    # -------------------------------------------------------------------------
    # Indicators
    # -------------------------------------------------------------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        # --- 1 h core ----------------------------------------------------------
        # df["TR"] = ta.TRANGE(df)
        all_periods = (
            set(self.buy_p1.range) | set(self.buy_p2.range) | set(self.buy_p3.range) |
            set(self.sell_p1.range) | set(self.sell_p2.range) | set(self.sell_p3.range) |
            set(self.buy_supported_p.range) | set(self.sell_supported_p.range)
        )
        for p in all_periods:
            df[f"ATR_{p}"] = ta.ATR(df, timeperiod=p).bfill()

        make_cols: dict[str, pd.Series] = {}
        hyper = self.config.get("runmode", "normal") == "hyperopt"
        if hyper:
            grids = [
                (1, self.buy_m1.range,  self.buy_p1.range,  "buy"),
                (2, self.buy_m2.range,  self.buy_p2.range,  "buy"),
                (3, self.buy_m3.range,  self.buy_p3.range,  "buy"),
                (1, self.sell_m1.range, self.sell_p1.range, "sell"),
                (2, self.sell_m2.range, self.sell_p2.range, "sell"),
                (3, self.sell_m3.range, self.sell_p3.range, "sell"),
            ]
            for idx, m_range, p_range, side in grids:
                for m in m_range:
                    for p in p_range:
                        sti = self._supertrend(df, m, p, f"ATR_{p}")
                        # direction column (needed for every grid point)
                        make_cols[f"supertrend_{idx}_{side}_{m}_{p}"] = sti["STX"]
                        # line column needed only for the mid-band filters
                        if idx == 2:
                            make_cols[f"supertrend_{idx}_{side}_{m}_{p}_line"] = sti["ST"]

        else:
            combos = [
                (1, self.buy_m1.value,  self.buy_p1.value,  "buy"),
                (2, self.buy_m2.value,  self.buy_p2.value,  "buy"),
                (3, self.buy_m3.value,  self.buy_p3.value,  "buy"),
                (1, self.sell_m1.value, self.sell_p1.value, "sell"),
                (2, self.sell_m2.value, self.sell_p2.value, "sell"),
                (3, self.sell_m3.value, self.sell_p3.value, "sell"),
            ]
            for idx, m, p, side in combos:
                sti = self._supertrend(df, m, p, f"ATR_{p}")
                make_cols[f"supertrend_{idx}_{side}_{m}_{p}"]      = sti["STX"]
                make_cols[f"supertrend_{idx}_{side}_{m}_{p}_line"] = sti["ST"]

        if hyper:
            grids = [
                (self.buy_supported_m.range,  self.buy_supported_p.range,  "buy"),
                (self.sell_supported_m.range, self.sell_supported_p.range, "sell"),
            ]

            # 1) Pull the informative dataframe just once
            inf = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_tf)
            # if your _supertrend or later logic needs a RangeIndex, do it once here:
            inf = inf.reset_index()

            # 2) Determine all unique ATR periods we’ll need
            all_supported_ps = set(self.buy_supported_p.range) | set(self.sell_supported_p.range)

            for p in all_supported_ps:
                atr_col = f"INF_ATR_{p}"
                # compute ATR once
                inf[atr_col] = ta.ATR(inf, timeperiod=p).bfill()

                # 3) for each grid side that uses this p, run all m’s
                for m_range, p_range, side in grids:
                    if p not in p_range:
                        continue
                    for m in m_range:
                        st = self._supertrend(inf, m, p, atr_col)
                        make_cols[f"supertrend_supported_{side}_{m}_{p}"] = st["STX"]

                # 4) drop the temporary ATR column to free memory
                inf.drop(columns=[atr_col], inplace=True)

        else:
            inf = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_tf)
            combos = [
                (self.buy_supported_m.value,  self.buy_supported_p.value,  "buy"),
                (self.sell_supported_m.value,  self.sell_supported_p.value,  "sell"),
            ]
            for m, p, side in combos:
                
                inf[f"INF_ATR_{p}"] = ta.ATR(inf, timeperiod=p).bfill()
                inf_st = self._supertrend(inf, m, p, f"INF_ATR_{p}")
                inf[f"supertrend_supported_{side}_{m}_{p}"] = inf_st["STX"]
                inf = inf.reset_index()
                make_cols[f"supertrend_supported_{side}_{m}_{p}"] = inf_st["STX"]

        df = df.join(pd.DataFrame(make_cols, index=df.index))

        return df

    # -------------------------------------------------------------------------
    # Entry
    # -------------------------------------------------------------------------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        buy1  = f"supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}"
        buy2  = f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"
        buy3  = f"supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}"
        sell1 = f"supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}"
        sell2 = f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"
        sell3 = f"supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}"
        supported_buy  = f"supertrend_supported_buy_{self.buy_supported_m.value}_{self.buy_supported_p.value}"
        supported_sell = f"supertrend_supported_sell_{self.sell_supported_m.value}_{self.sell_supported_p.value}"

        df.loc[
            (
                 (df[buy1] == "up") & (df[buy2] == "up") & (df[buy3] == "up")
                & (df[supported_buy] == "up")
            ),
            "enter_long",
        ] = 1

        df.loc[
            (
                 (df[sell1] == "down") & (df[sell2] == "down") & (df[sell3] == "down")
                & (df[supported_sell] == "down")
            ),
            "enter_short",
        ] = 1

        return df

    # -------------------------------------------------------------------------
    # Exit
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        sell2 = f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"
        buy2  = f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"
        supported_buy  = f"supertrend_supported_buy_{self.buy_supported_m.value}_{self.buy_supported_p.value}"
        supported_sell = f"supertrend_supported_sell_{self.sell_supported_m.value}_{self.sell_supported_p.value}"

        df.loc[
            (df[sell2] == "down") 
            & (df[supported_sell] == 'down')
            , "exit_long"] = 1
        
        df.loc[
            (df[buy2]  == "up")
            & (df[supported_buy] == 'up')
            , "exit_short"] = 1
        return df

    # -------------------------------------------------------------------------
    # Supertrend core
    # -------------------------------------------------------------------------
    @staticmethod
    def _supertrend(df: DataFrame, multiplier: int, period: int, atr_col: str = "ATR") -> DataFrame:
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        atr = df[atr_col].values

        ln = len(df)
        st, final_ub, final_lb = (np.full(ln, np.nan) for _ in range(3))
        stx = np.full(ln, None, dtype=object)

        hl2      = (high + low) / 2
        basic_ub = hl2 + multiplier * atr
        basic_lb = hl2 - multiplier * atr

        start = np.argmax(~np.isnan(atr))
        if np.isnan(atr[start]):
            return DataFrame({"ST": st, "STX": stx}, index=df.index)

        final_ub[start], final_lb[start] = basic_ub[start], basic_lb[start]
        st[start], stx[start] = final_ub[start], "down"

        for i in range(start + 1, ln):
            final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
            final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

            if st[i - 1] == final_ub[i - 1] and close[i] <= final_ub[i]:
                st[i] = final_ub[i]
            elif st[i - 1] == final_ub[i - 1] and close[i] > final_ub[i]:
                st[i] = final_lb[i]
            elif st[i - 1] == final_lb[i - 1] and close[i] >= final_lb[i]:
                st[i] = final_lb[i]
            else:
                st[i] = final_ub[i]

            stx[i] = "down" if close[i] < st[i] else "up"

        return DataFrame({"ST": st, "STX": stx}, index=df.index)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    @property
    def plot_config(self):
        return {
            "main_plot": {
                f"supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}_line":  {"color": "green"},
                f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}_line":  {"color": "green"},
                f"supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}_line":  {"color": "green"},
                f"supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}_line": {"color": "red"},
                f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}_line": {"color": "red"},
                f"supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}_line": {"color": "red"},
            }
        }