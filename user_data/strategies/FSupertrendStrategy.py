import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, CategoricalParameter, Trade, DecimalParameter, BooleanParameter
import talib.abstract as ta

class FSupertrendStrategy(IStrategy):
    """
    Refactored Supertrend Strategy
    * Generates 3 Supertrend indicators for 'buy' and 3 for 'sell'.
    * Buys when all 3 buy signals are 'up', shorts when all 3 sell signals are 'down'.
    * RSI threshold is hyper-optimizable (45, 50, 55, 60).
    * Filters entries when candle size is too large (hyper-optimizable max_candle_size).
    * Requires a recent swing low for longs and swing high for shorts (hyper-optimized separate look-backs).
    """
    INTERFACE_VERSION = 3
    can_short = True
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 1.0

     # Buy hyperspace params:
    buy_params = {
        "buy_m1": 2,
        "buy_m2": 1,
        "buy_m3": 1,
        "buy_p1": 20,
        "buy_p2": 9,
        "buy_p3": 19,
        "min_exit_p_long": 0.011,
        "rsi_buy_threshold": 45,
    }

    # Sell hyperspace params:
    sell_params = {
        "min_exit_p_short": 0.004,
        "rsi_sell_threshold": 60,
        "sell_m1": 1,
        "sell_m2": 1,
        "sell_m3": 4,
        "sell_p1": 12,
        "sell_p2": 12,
        "sell_p3": 14,
    }

    # Protection hyperspace params:
    protection_params = {
        "cool_down": 9,
        "enable_slg": False,
        "lpp_lookback": 95,
        "lpp_pause": 14,
        "lpp_profit": -0.019,
        "lpp_trades": 3,
        "slg_limit": 1,
        "slg_lookback": 124,
        "slg_pause": 26,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.258,
        "305": 0.119,
        "912": 0.068,
        "2287": 0
    }

    # Stoploss:
    stoploss = -0.068

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.062
    trailing_only_offset_is_reached = True

    timeframe = "1h"
    startup_candle_count = 18

    # Supertrend hyperopt parameters
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

    # Parameterize RSI threshold
    rsi_buy_threshold = CategoricalParameter([45, 50, 55, 60], default=50, space="buy")
    rsi_sell_threshold = CategoricalParameter([45, 50, 55, 60], default=50, space="sell")

    # Parameterize max candle size (as (high-low)/close)
    buy_max_candle_size = CategoricalParameter([0.05, 0.07, 0.09], default=0.07, space="buy")
    sell_max_candle_size = CategoricalParameter([0.05, 0.07, 0.09], default=0.07, space="sell")

    # Separate look-back for swing lows (long entries) and swing highs (short entries)
    swing_low_look_back = CategoricalParameter([3, 4, 5, 6], default=6, space="buy")
    swing_high_look_back = CategoricalParameter([3, 4, 5, 6], default=6, space="sell")
    
    # individual buffers (DecimalParameters so you can hyper-opt them)
    min_exit_p_long  = DecimalParameter(0.004, 0.012, decimals=3,
                                        default=0.008, space="buy")   # 0.4 % – 1.2 %
    min_exit_p_short = DecimalParameter(0.002, 0.008, decimals=3,
                                        default=0.004, space="sell")   # 0.2 % – 0.8 %
    
    # RSI period (length) – three classic choices
    rsi_length = CategoricalParameter([7, 14, 21], default=14)

    # MACD component EMAs
    macd_fast   = CategoricalParameter([8, 12, 15],  default=12)
    macd_slow   = CategoricalParameter([21, 26, 30], default=26)
    macd_signal = CategoricalParameter([5, 9, 12],   default=9)

    # Minimum / maximum MACD-histogram bias you want to see
    macd_hist_buy  = DecimalParameter(0.000, 0.005,  decimals=3, default=0.000, space="buy")
    macd_hist_sell = DecimalParameter(-0.005, 0.000, decimals=3, default=0.000, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # True Range
        dataframe["TR"] = ta.TRANGE(dataframe)

        # ATR for each unique period
        periods = (
            set(self.buy_p1.range) | set(self.buy_p2.range) | set(self.buy_p3.range) |
            set(self.sell_p1.range) | set(self.sell_p2.range) | set(self.sell_p3.range)
        )
        for p in periods:
            dataframe[f"ATR_{p}"] = ta.ATR(dataframe, timeperiod=p).bfill()

        # Supertrend combos
        is_hopt = self.config.get("runmode", "normal") == "hyperopt"
        if is_hopt:
            cols = {}
            combos = [
                (1, self.buy_m1.range,  self.buy_p1.range,  "buy"),
                (2, self.buy_m2.range,  self.buy_p2.range,  "buy"),
                (3, self.buy_m3.range,  self.buy_p3.range,  "buy"),
                (1, self.sell_m1.range, self.sell_p1.range, "sell"),
                (2, self.sell_m2.range, self.sell_p2.range, "sell"),
                (3, self.sell_m3.range, self.sell_p3.range, "sell"),
            ]
            for idx, m_range, p_range, side in combos:
                for m in m_range:
                    for p in p_range:
                        st = self._supertrend(dataframe, m, p, f"ATR_{p}")
                        cols[f"supertrend_{idx}_{side}_{m}_{p}"]       = st["STX"]
                        cols[f"supertrend_{idx}_{side}_{m}_{p}_line"]  = st["ST"]
            dataframe = dataframe.join(pd.DataFrame(cols, index=dataframe.index))
        else:
            cols = {}
            combos = [
                (1, self.buy_m1.value,  self.buy_p1.value,  "buy"),
                (2, self.buy_m2.value,  self.buy_p2.value,  "buy"),
                (3, self.buy_m3.value,  self.buy_p3.value,  "buy"),
                (1, self.sell_m1.value, self.sell_p1.value, "sell"),
                (2, self.sell_m2.value, self.sell_p2.value, "sell"),
                (3, self.sell_m3.value, self.sell_p3.value, "sell"),
            ]
            for idx, m, p, side in combos:
                st_df = self._supertrend(dataframe, m, p, f"ATR_{p}")
                cols[f"supertrend_{idx}_{side}_{m}_{p}"] = st_df["STX"]
                cols[f"supertrend_{idx}_{side}_{m}_{p}_line"] = st_df["ST"]
            dataframe = dataframe.join(pd.DataFrame(cols, index=dataframe.index))

        # --- RSI ---------------------------------------------------------------
        dataframe['rsi'] = ta.RSI(
            dataframe,
            timeperiod=self.rsi_length.value
        )

        # --- MACD --------------------------------------------------------------
        macd = ta.MACD(
            dataframe,
            fastperiod   = self.macd_fast.value,
            slowperiod   = self.macd_slow.value,
            signalperiod = self.macd_signal.value,
        )
        dataframe['macd']       = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist']   = macd['macdhist']

        # Candle size ratio
        dataframe['candle_size'] = (dataframe['high'] - dataframe['low']) / dataframe['close']
        
        if is_hopt:
            for n in [2, 3, 4, 5, 6]:
                # --- Swing low / Swing high (non-repainting & leak-safe) ----------------
                lb_low  = int(n)
                win_low = lb_low * 2 + 1

                lo_roll   = dataframe['low'].rolling(window=win_low, center=True,
                                                    min_periods=win_low).min()
                pivot_low = dataframe['low'] == lo_roll
                # shift + fill_value keeps the dtype strictly boolean → no down-casting warning
                dataframe[f'swing_low_{n}'] = pivot_low.shift(lb_low, fill_value=False)

                lb_high  = int(n)
                win_high = lb_high * 2 + 1

                hi_roll    = dataframe['high'].rolling(window=win_high, center=True,
                                                    min_periods=win_high).max()
                pivot_high = dataframe['high'] == hi_roll
                dataframe[f'swing_high_{n}'] = pivot_high.shift(lb_high, fill_value=False)
        else:
            # --- Swing low / Swing high (non-repainting & leak-safe) ----------------
            lb_low  = int(self.swing_low_look_back.value)
            win_low = lb_low * 2 + 1

            lo_roll   = dataframe['low'].rolling(window=win_low, center=True,
                                                min_periods=win_low).min()
            pivot_low = dataframe['low'] == lo_roll
            # shift + fill_value keeps the dtype strictly boolean → no down-casting warning
            dataframe[f'swing_low_{self.swing_low_look_back.value}'] = pivot_low.shift(lb_low, fill_value=False)

            lb_high  = int(self.swing_high_look_back.value)
            win_high = lb_high * 2 + 1

            hi_roll    = dataframe['high'].rolling(window=win_high, center=True,
                                                min_periods=win_high).max()
            pivot_high = dataframe['high'] == hi_roll
            dataframe[f'swing_high_{self.swing_high_look_back.value}'] = pivot_high.shift(lb_high, fill_value=False)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_mask = (
            (dataframe[f"supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}"] == "up") &
            (dataframe[f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"] == "up") &
            (dataframe[f"supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}"] == "up") &
            (dataframe['macdhist'] > self.macd_hist_buy.value) &
            (dataframe['rsi'] > self.rsi_buy_threshold.value) &
            (dataframe['volume'] > 0) &
            (dataframe['candle_size'] < self.buy_max_candle_size.value) &
            (dataframe[f'swing_low_{self.swing_low_look_back.value}'])
        )
        dataframe.loc[buy_mask, 'enter_long'] = 1

        sell_mask = (
            (dataframe[f"supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}"] == "down") &
            (dataframe[f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"] == "down") &
            (dataframe[f"supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}"] == "down") &
            (dataframe['macdhist'] < self.macd_hist_sell.value) &
            (dataframe['rsi'] < self.rsi_sell_threshold.value) &
            (dataframe['volume'] > 0) &
            (dataframe['candle_size'] < self.sell_max_candle_size.value) &
            (dataframe[f'swing_high_{self.swing_high_look_back.value}'])
        )
        dataframe.loc[sell_mask, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long = dataframe[f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"] == "down"
        dataframe.loc[exit_long, 'exit_long'] = 1

        exit_short = dataframe[f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"] == "up"
        dataframe.loc[exit_short, 'exit_short'] = 1
        return dataframe

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """
        Advanced exit filter:
        - Accepts ROI, stoploss, trailing exit without question.
        - For 'exit_signal', adds profit threshold AND confirms with MACD/RSI fade.
        - Forces exit if max duration exceeded (e.g. 24 candles = 24 hours on 1h).
        """
        # Allow standard exit reasons
        if exit_reason != "exit_signal":
            return True

        # Profit calculation
        profit_ratio = (rate - trade.open_rate) / trade.open_rate
        if trade.is_short:
            profit_ratio = -profit_ratio
            min_profit = self.min_exit_p_short.value
        else:
            min_profit = self.min_exit_p_long.value

        # Retrieve the latest candle for this pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return profit_ratio > min_profit  # fallback

        last_candle = dataframe.iloc[-1]

        # Momentum confirmation: MACD histogram and RSI divergence
        macd_confirms = (
            (trade.is_short and last_candle['macdhist'] > self.macd_hist_buy.value * 2) or
            (not trade.is_short and last_candle['macdhist'] < self.macd_hist_sell.value * 2)
        )

        rsi_confirms = (
            (trade.is_short and last_candle['rsi'] > self.rsi_sell_threshold.value + 20) or
            (not trade.is_short and last_candle['rsi'] < self.rsi_buy_threshold.value - 20)
        )

        momentum_fade = macd_confirms and rsi_confirms

        return profit_ratio > min_profit or momentum_fade


    @staticmethod
    def _supertrend(
        df: DataFrame,
        multiplier: int,
        period: int,
        atr_col: str = "ATR"
    ) -> DataFrame:
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        atr = df[atr_col].values

        length = len(df)
        st = np.full(length, np.nan)
        final_ub = np.full(length, np.nan)
        final_lb = np.full(length, np.nan)
        stx = np.full(length, None, dtype=object)

        hl2 = (high + low) / 2
        basic_ub = hl2 + multiplier * atr
        basic_lb = hl2 - multiplier * atr

        start = np.argmax(~np.isnan(atr))
        if np.isnan(atr[start]):
            return pd.DataFrame({"ST": st, "STX": stx}, index=df.index)

        final_ub[start] = basic_ub[start]
        final_lb[start] = basic_lb[start]
        st[start] = final_ub[start]
        stx[start] = "down"

        for i in range(start + 1, length):
            final_ub[i] = (
                basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1])
                else final_ub[i-1]
            )
            final_lb[i] = (
                basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1])
                else final_lb[i-1]
            )

            if   st[i-1] == final_ub[i-1] and close[i] <= final_ub[i]:
                st[i] = final_ub[i]
            elif st[i-1] == final_ub[i-1] and close[i] >  final_ub[i]:
                st[i] = final_lb[i]
            elif st[i-1] == final_lb[i-1] and close[i] >= final_lb[i]:
                st[i] = final_lb[i]
            else:
                st[i] = final_ub[i]

            stx[i] = "down" if close[i] < st[i] else "up"

        return pd.DataFrame({"ST": st, "STX": stx}, index=df.index)
    
     # ——————————————————————————————————————————
    # 4. Protections – all parameterised
    # ——————————————————————————————————————————
    cool_down    = IntParameter(2, 48, default=5,
                                space="protection")
    slg_lookback = IntParameter(24, 144, default=72,
                                space="protection")     # 24 = 6 h
    slg_limit    = IntParameter(1, 6, default=3,
                                space="protection")
    slg_pause    = IntParameter(6, 48, default=12,
                                space="protection")
    lpp_lookback = IntParameter(24, 96, default=48,
                                space="protection")
    lpp_trades   = IntParameter(2, 6, default=3,
                                space="protection")
    lpp_profit   = DecimalParameter(-0.02, 0.02, default=0.0,
                                    decimals=3, space="protection")
    lpp_pause    = IntParameter(12, 48, default=24,
                                space="protection")
    enable_slg   = BooleanParameter(default=True, space="protection")

    # ——————————————————————————————————————————
    # 4.a  Protections callback
    # ——————————————————————————————————————————
    @property
    def protections(self):
        prot = [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cool_down.value
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": self.lpp_lookback.value,
                "trade_limit": self.lpp_trades.value,
                "required_profit": self.lpp_profit.value,
                "stop_duration_candles": self.lpp_pause.value,
                "only_per_pair": True
            }
        ]

        if self.enable_slg.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": self.slg_lookback.value,
                "trade_limit": self.slg_limit.value,
                "stop_duration_candles": self.slg_pause.value,
                "only_per_pair": True
            })

        return prot

    @property
    def plot_config(self):
        return {
            "main_plot": {
                f"supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}_line": {},
                f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}_line": {},
                f"supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}_line": {},
                f"supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}_line": {},
                f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}_line": {},
                f"supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}_line": {}
            },
            "subplots": {
                "MACD": { "macd": {}, "macdsignal": {}, "macdhist": {} },
                "RSI":  { "rsi": {} }
            }
        }
