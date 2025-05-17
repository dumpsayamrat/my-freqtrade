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
        "buy_m1": 3,
        "buy_m2": 1,
        "buy_m3": 3,
        "buy_p1": 13,
        "buy_p2": 19,
        "buy_p3": 13,
        "buy_supported_m": 1,
        "buy_supported_p": 13,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 1,
        "sell_m2": 1,
        "sell_m3": 1,
        "sell_p1": 7,
        "sell_p2": 7,
        "sell_p3": 14,
        "sell_supported_m": 1,
        "sell_supported_p": 11,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.08,
        "132": 0.054,
        "343": 0.026,
        "1603": 0
    }

    # Stoploss:
    stoploss = -0.177

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.032
    trailing_only_offset_is_reached = True


    # Max Open Trades:
    max_open_trades = 5  # value loaded from strategy

    timeframe = "5m"
    startup_candle_count = 50

    informative_tf = "1h"

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
        return min(1.0, max_leverage)

    # -------------------------------------------------------------------------
    # Indicators
    # -------------------------------------------------------------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        runmode_hyperopt = self.config.get("runmode", "normal") == "hyperopt"

        make_cols: dict[str, pd.Series] = {}

        # === 1h informative timeframe indicators ===
        inf = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_tf).reset_index()

        def collect_combos(idx_prefix: int, m, p, side: str):
            if runmode_hyperopt:
                return [(idx_prefix, mi, pi, side) for mi in m.range for pi in p.range]
            else:
                return [(idx_prefix, m.value, p.value, side)]

        all_combos = (
            collect_combos(1, self.buy_m1, self.buy_p1, "buy") +
            collect_combos(2, self.buy_m2, self.buy_p2, "buy") +
            collect_combos(3, self.buy_m3, self.buy_p3, "buy") +
            collect_combos(1, self.sell_m1, self.sell_p1, "sell") +
            collect_combos(2, self.sell_m2, self.sell_p2, "sell") +
            collect_combos(3, self.sell_m3, self.sell_p3, "sell")
        )

        informative_atrs = {p for _, _, p, _ in all_combos}
        for p in informative_atrs:
            inf[f"INF_ATR_{p}"] = ta.ATR(inf, timeperiod=p).bfill()

        for idx, m, p, side in all_combos:
            atr_col = f"INF_ATR_{p}"
            st = self._supertrend(inf, m, p, atr_col)
            make_cols[f"supertrend_{idx}_{side}_{m}_{p}"] = st["STX"]
            if idx == 2:
                make_cols[f"supertrend_{idx}_{side}_{m}_{p}_line"] = st["ST"]

        # Merge informative data into df
        inf = merge_informative_pair(df, inf, self.timeframe, self.informative_tf, ffill=True)
        df = df.join(pd.DataFrame({k: v for k, v in make_cols.items() if v is not None}, index=df.index))

        # === Supported supertrend on main timeframe ===
        support_cols: dict[str, pd.Series] = {}

        support_combos = [
            ("buy", self.buy_supported_m, self.buy_supported_p),
            ("sell", self.sell_supported_m, self.sell_supported_p),
        ]

        all_support_periods = set()
        for _, _, p in support_combos:
            if runmode_hyperopt:
                all_support_periods |= set(p.range)

        for p in all_support_periods:
            df[f"ATR_{p}"] = ta.ATR(df, timeperiod=p).bfill()

        for side, m_param, p_param in support_combos:
            m_range = m_param.range if runmode_hyperopt else [m_param.value]
            p_range = p_param.range if runmode_hyperopt else [p_param.value]

            for m in m_range:
                for p in p_range:
                    atr_col = f"ATR_{p}"
                    if atr_col not in df.columns:
                        df[atr_col] = ta.ATR(df, timeperiod=p).bfill()
                    st = self._supertrend(df, m, p, atr_col)
                    support_cols[f"supertrend_supported_{side}_{m}_{p}"] = st["STX"]

        # Join all supported columns at once to avoid fragmentation
        df = pd.concat([df, pd.DataFrame(support_cols, index=df.index)], axis=1)

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