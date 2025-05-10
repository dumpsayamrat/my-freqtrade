# -*- coding: utf-8 -*-
# freqtrade strategy file for Pin Bar with Bollinger Bands + EMA5

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class PinbarStrategy(IStrategy):
    """
    Pin Bar with Bollinger Bands + EMA5 Strategy
    """

    # ROI table:
    minimal_roi = {
        "0": 0.257,
        "39": 0.082,
        "85": 0.021,
        "190": 0
    }

    # Stoploss:
    stoploss = -0.301

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.208
    trailing_stop_positive_offset = 0.252
    trailing_only_offset_is_reached = False

    # Strategy timeframe and indicator settings
    timeframe = '5m'
    bb_window = 40
    bb_std = 2.0
    ema_period = 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bb_upperband, bb_middleband, bb_lowerband = ta.BBANDS(
            dataframe['close'],
            timeperiod=self.bb_window,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )
        dataframe['bb_upperband'] = bb_upperband
        dataframe['bb_middleband'] = bb_middleband
        dataframe['bb_lowerband'] = bb_lowerband

        # EMA with 5-period
        dataframe['ema5'] = ta.EMA(dataframe['close'], timeperiod=self.ema_period)

        # Candle range and tails
        dataframe['candle_range'] = dataframe['high'] - dataframe['low']
        dataframe['lower_tail'] = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        dataframe['upper_tail'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)

        # Pin bar flags
        dataframe['bullish_pin'] = (
            (dataframe['close'] > dataframe['open']) &
            (dataframe['candle_range'] > 0) &
            (dataframe['lower_tail'] >= 0.75 * dataframe['candle_range'])
        )
        dataframe['bearish_pin'] = (
            (dataframe['candle_range'] > 0) &
            (dataframe['upper_tail'] >= 0.75 * dataframe['candle_range'])
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Entry long: previous candle was a bullish pin at/below lower BB and current close > EMA5
        dataframe.loc[
            (
                dataframe['bullish_pin'].shift(1) &
                (dataframe['low'].shift(1) <= dataframe['bb_lowerband'].shift(1)) &
                (dataframe['close'] > dataframe['ema5'])
            ), 'enter_long'] = 1

        # Entry short: previous candle was a bearish pin at/above upper BB and current close < EMA5
        dataframe.loc[
            (
                dataframe['bearish_pin'].shift(1) &
                (dataframe['high'].shift(1) >= dataframe['bb_upperband'].shift(1)) &
                (dataframe['close'] < dataframe['ema5'])
            ), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long: current bearish pin at/above upper BB
        dataframe.loc[
            (dataframe['bearish_pin'] &
            (dataframe['close'] >= dataframe['bb_upperband'])),
            'exit_long'] = 1

        # Exit short: current bullish pin at/below lower BB
        dataframe.loc[
            (
                dataframe['bullish_pin'] &
                (dataframe['close'] <= dataframe['bb_lowerband'])
            ), 'exit_short'] = 1

        return dataframe

    @property
    def plot_config(self):
        return {
            'main_plot': {
                'bb_upperband': {'color': 'red'},
                'bb_middleband': {'color': 'blue'},
                'bb_lowerband': {'color': 'green'},
                'ema5': {'color': 'orange'},
            },
            'subplots': {
                "Tail": {
                    'tail': {'color': 'purple'},
                }
            }
        }
