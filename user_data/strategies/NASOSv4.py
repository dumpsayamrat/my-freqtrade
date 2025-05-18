# flake8: noqa: F401
# isort: skip_file

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from functools import reduce
from typing import Dict, List, Optional
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    Trade,
    DecimalParameter,
    IntParameter,
    merge_informative_pair,
    stoploss_from_open,
)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import technical.indicators as ftt
from technical.util import resample_to_interval, resampled_merge

# @Rallipanos
# @pluxury

# Buy hyperspace params (defaults)
buy_params = {
    "base_nb_candles_buy": 8,
    "ewo_high": 2.403,
    "ewo_high_2": -5.585,
    "ewo_low": -14.378,
    "lookback_candles": 3,
    "low_offset": 0.984,
    "low_offset_2": 0.942,
    "profit_threshold": 1.008,
    "rsi_buy": 72
}

# Sell hyperspace params (defaults)
sell_params = {
    "base_nb_candles_sell": 16,
    "high_offset": 1.084,
    "high_offset_2": 1.401,
    "pHSL": -0.15,
    "pPF_1": 0.016,
    "pPF_2": 0.024,
    "pSL_1": 0.014,
    "pSL_2": 0.022
}


def EWO(dataframe: DataFrame, ema_length: int = 5, ema2_length: int = 35) -> pd.Series:
    """
    Elliott Wave Oscillator: (EMA1 - EMA2) / low * 100
    """
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe['low'] * 100


class NASOSv4(IStrategy):
    INTERFACE_VERSION = 3  # <- Updated for latest Freqtrade :contentReference[oaicite:2]{index=2}
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # holds how many times we've blocked exit for each pair
        self._slippage_retries_state: Dict[str, int] = {}
    
    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 9,
        "lookback_candles": 19,
        "profit_threshold": 1.02,
        "ewo_high": 2.403,  # value loaded from strategy
        "ewo_high_2": -5.585,  # value loaded from strategy
        "ewo_low": -14.378,  # value loaded from strategy
        "low_offset": 0.984,  # value loaded from strategy
        "low_offset_2": 0.942,  # value loaded from strategy
        "rsi_buy": 72,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 19,
        "high_offset": 0.993,
        "high_offset_2": 1.006,
        "pHSL": -0.15,  # value loaded from strategy
        "pPF_1": 0.016,  # value loaded from strategy
        "pPF_2": 0.024,  # value loaded from strategy
        "pSL_1": 0.014,  # value loaded from strategy
        "pSL_2": 0.022,  # value loaded from strategy
    }

    # Protection hyperspace params:
    protection_params = {
        "max_slippage": -0.025,
        "slippage_retries": 8,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.186,
        "34": 0.066,
        "91": 0.031,
        "177": 0
    }

    # Stoploss:
    stoploss = -0.311

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.272
    trailing_stop_positive_offset = 0.32
    trailing_only_offset_is_reached = True

    # Hyperopt-optimizable parameters
    base_nb_candles_buy = IntParameter(2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(2, 25, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset       = DecimalParameter(0.90, 0.99, default=buy_params['low_offset'],  space='buy', optimize=False)
    low_offset_2     = DecimalParameter(0.90, 0.99, default=buy_params['low_offset_2'],space='buy', optimize=False)
    high_offset      = DecimalParameter(0.95, 1.10, default=sell_params['high_offset'],space='sell',optimize=True)
    high_offset_2    = DecimalParameter(0.99, 1.50, default=sell_params['high_offset_2'],space='sell',optimize=True)

    lookback_candles  = IntParameter(1, 24, default=buy_params['lookback_candles'], space='buy', optimize=True)
    profit_threshold  = DecimalParameter(1.00, 1.03, default=buy_params['profit_threshold'], space='buy', optimize=True)
    rsi_buy           = IntParameter(50, 100, default=buy_params['rsi_buy'], space='buy', optimize=False)

    # EWO protection bands
    fast_ewo = 50
    slow_ewo = 200
    ewo_low    = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'],   space='buy', optimize=False)
    ewo_high   = DecimalParameter( 2.0, 12.0, default=buy_params['ewo_high'],  space='buy', optimize=False)
    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=buy_params['ewo_high_2'],space='buy', optimize=False)

    # Trailing-stop hyperopt params
    pHSL  = DecimalParameter(-0.200, -0.040, default=sell_params['pHSL'],  decimals=3, space='sell', optimize=False)
    pPF_1 = DecimalParameter( 0.008,  0.020, default=sell_params['pPF_1'], decimals=3, space='sell', optimize=False)
    pSL_1 = DecimalParameter( 0.008,  0.020, default=sell_params['pSL_1'], decimals=3, space='sell', optimize=False)
    pPF_2 = DecimalParameter( 0.040,  0.100, default=sell_params['pPF_2'], decimals=3, space='sell', optimize=False)
    pSL_2 = DecimalParameter( 0.020,  0.070, default=sell_params['pSL_2'], decimals=3, space='sell', optimize=False)


    use_exit_signal      = True
    exit_profit_only     = False
    exit_profit_offset   = 0.01
    ignore_roi_if_entry_signal = False

    order_time_in_force = {
        'entry':  'gtc',
        'exit': 'ioc'
    }

    timeframe = '5m'
    inf_1h    = '1h'

    process_only_new_candles = True
    startup_candle_count     = 200
    use_custom_stoploss      = False  # switch per backtest vs live

    plot_config = {
        'main_plot': {
            'ma_buy':  {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }
    
    slippage_retries = IntParameter(1, 10, default=3, space='protection', optimize=True)
    max_slippage     = DecimalParameter(-0.10, 0.00, default=-0.02, decimals=3, space='protection', optimize=True)


    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs
    ) -> float:
        HSL  = self.pHSL.value
        PF1  = self.pPF_1.value
        SL1  = self.pSL_1.value
        PF2  = self.pPF_2.value
        SL2  = self.pSL_2.value

        if current_profit > PF2:
            sl = SL2 + (current_profit - PF2)
        elif current_profit > PF1:
            sl = SL1 + ((current_profit - PF1) * (SL2 - SL1) / (PF2 - PF1))
        else:
            sl = HSL

        return stoploss_from_open(sl, current_profit)

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        sell_reason: str,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """
        Decide whether to allow an exit (sell) order to proceed.
        Blocks exits under certain technical conditions or excessive slippage,
        retrying up to `slippage_retries` times if slippage is worse than `max_slippage`.
        """
        # Fetch latest candle
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last = df.iloc[-1]

        # 1) Block exit on custom technical condition
        if sell_reason == 'sell_signal':
            # if HMA50*1.149 > EMA100 AND close < EMA100*0.951, hold the position
            if (last['hma_50'] * 1.149 > last['ema_100']) and (last['close'] < last['ema_100'] * 0.951):
                return False

        # 2) Slippage protection
        # Ensure we have a retry counter dict
        if not hasattr(self, '_slippage_retries_state'):
            self._slippage_retries_state: Dict[str, int] = {}

        # Compute slippage: how far off our desired price we are
        slippage = (rate / last['close']) - 1
        retries = self._slippage_retries_state.get(pair, 0)

        # If slippage is worse than threshold and we haven’t exhausted retries, block exit
        if slippage < self.max_slippage.value and retries < self.slippage_retries.value:
            self._slippage_retries_state[pair] = retries + 1
            return False

        # Otherwise reset retry count and allow exit
        self._slippage_retries_state[pair] = 0
        return True

    def informative_pairs(self) -> List[tuple]:
        return [(pair, self.inf_1h) for pair in self.dp.current_whitelist()]

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        # 1h EMA, RSI, BB, etc. (uncomment as needed)
        df1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # df1h['ema_50'] = ta.EMA(df1h, timeperiod=50)
        return df1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        # fast EMAs
        for n in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{n}'] = ta.EMA(dataframe, timeperiod=n)
        for n in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{n}'] = ta.EMA(dataframe, timeperiod=n)

        dataframe['hma_50']   = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100']  = ta.EMA(dataframe, timeperiod=100)
        dataframe['sma_9']    = ta.SMA(dataframe, timeperiod=9)
        dataframe['EWO']      = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe['rsi']      = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        # merge 1h and 5m indicators
        df1h = self.informative_1h_indicators(dataframe.copy(), metadata)
        dataframe = merge_informative_pair(dataframe, df1h, self.timeframe, self.inf_1h, ffill=True)
        return self.normal_tf_indicators(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        # prevent buys without enough upside
        no_upside = (
            dataframe['close_1h']
                .rolling(self.lookback_candles.value)
                .max()
            < dataframe['close'] * self.profit_threshold.value
        )
        dataframe.loc[no_upside, 'enter_long'] = 0

        entries = [
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value),
                'ewo1'
            ),
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value) &
                (dataframe['rsi'] < 25),
                'ewo2'
            ),
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value),
                'ewolow'
            )
        ]

        for cond, tag in entries:
            dataframe.loc[cond, ['enter_long', 'enter_tag']] = (1, tag)

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        conditions = [
            (
                (dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            ),
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )
        ]

        mask = reduce(lambda a, b: a | b, conditions)
        dataframe.loc[mask, ['exit_long', 'exit_tag']] = (1, 'exit_signal')

        return dataframe
