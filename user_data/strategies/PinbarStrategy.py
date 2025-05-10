# five_minute_pinbar_strategy.py
# A Freqtrade strategy for Binance Futures on 5-minute timeframe
# Pinbar entries using Bollinger Bands and EMA(5)

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class PinbarStrategy(IStrategy):
    """
    Futures strategy: Bullish/Bearish pinbar + EMA5 filter.
    Only one position open at a time (long or short).
    Pinbar cancellation and trade exits based on EMA or pinbar invalidation.
    """
    INTERFACE_VERSION: int = 3

    # Strategy configuration
    timeframe = '5m'
    can_long = True
    can_short = True

     # ROI table:
    minimal_roi = {
        "0": 0.141,
        "27": 0.034,
        "53": 0.011,
        "151": 0
    }

    # Stoploss:
    stoploss = -0.053

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.146
    trailing_stop_positive_offset = 0.176
    trailing_only_offset_is_reached = True

    # Warmup period
    startup_candle_count: int = 20

    # Bollinger Bands parameters
    bb_window = 60
    bb_stddev = 2.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA(5)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=10)

        # Bollinger Bands
        bb = ta.BBANDS(
            dataframe,
            timeperiod=self.bb_window,
            nbdevup=self.bb_stddev,
            nbdevdn=self.bb_stddev,
            matype=0
        )
        dataframe['bb_upperband'] = bb['upperband']
        dataframe['bb_middleband'] = bb['middleband']
        dataframe['bb_lowerband'] = bb['lowerband']

        # Candle geometry for pinbar detection
        dataframe['range'] = dataframe['high'] - dataframe['low']
        # Wicks
        dataframe['lower_tail'] = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        dataframe['upper_tail'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        # Overall tail (max of wicks)
        dataframe['tail'] = dataframe[['lower_tail', 'upper_tail']].max(axis=1)

        # Bullish pinbar: wick >=75% of range and low touching lower BB
        dataframe['bull_pinbar'] = (
            (dataframe['low'] <= dataframe['bb_lowerband']) &
            (dataframe['tail'] >= 0.75 * dataframe['range'])
        ).astype(int)

        # Bearish pinbar: wick >=75% of range and high touching upper BB
        dataframe['bear_pinbar'] = (
            (dataframe['high'] >= dataframe['bb_upperband']) &
            (dataframe['tail'] >= 0.75 * dataframe['range'])
        ).astype(int)

        # Pinbar price levels for cancellation/exit logic
        dataframe['pin_low'] = dataframe['low'].shift(1).where(dataframe['bull_pinbar'].shift(1) == 1)
        dataframe['pin_high'] = dataframe['high'].shift(1).where(dataframe['bear_pinbar'].shift(1) == 1)
        dataframe['pin_low_ffill'] = dataframe['pin_low'].ffill()
        dataframe['pin_high_ffill'] = dataframe['pin_high'].ffill()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry: previous bar pinned bullish, close > EMA5, and not broken below pinbar low
        dataframe.loc[
            (
                (dataframe['bull_pinbar'].shift(1) == 1) &
                (dataframe['close'] > dataframe['ema5']) &
                (dataframe['close'] >= dataframe['pin_low_ffill'])
            ),
            'enter_long'
        ] = 1

        # Short entry: previous bar pinned bearish, close < EMA5, and not broken above pinbar high
        dataframe.loc[
            (
                (dataframe['bear_pinbar'].shift(1) == 1) &
                (dataframe['close'] < dataframe['ema5']) &
                (dataframe['close'] <= dataframe['pin_high_ffill'])
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long: price falls below EMA5 or invalidates pinbar low
        # dataframe.loc[
        #     (
        #         (dataframe['close'] < dataframe['ema5'])
        #     ),
        #     'exit_long'
        # ] = 1

        # # Exit short: price rises above EMA5 or invalidates pinbar high
        # dataframe.loc[
        #     (
        #         (dataframe['close'] > dataframe['ema5'])
        #     ),
        #     'exit_short'
        # ] = 1

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
