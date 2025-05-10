from functools import reduce
from pandas import DataFrame
import pandas as pd
import numpy as np

import talib.abstract as ta
from freqtrade.strategy import IStrategy

class TrendFollowingStrategy(IStrategy):

    INTERFACE_VERSION: int = 3
    # ROI table:
    minimal_roi = {
        "0": 0.253,
        "18": 0.071,
        "48": 0.032,
        "165": 0
    }
    # minimal_roi = {"0": 1}

    # Stoploss:
    stoploss = -0.349

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive =  0.018
    trailing_stop_positive_offset = 0.066
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    
    # Define startup candles required for indicators
    startup_candle_count = 30
    
    # Hyperopt configuration
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate OBV
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        
        # Add trend following indicators - use more efficient calculation
        dataframe['trend'] = ta.EMA(dataframe, timeperiod=20)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Add trend following buy signals
        conditions_long = [
            (dataframe['close'] > dataframe['trend']),
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)),
            (dataframe['obv'] > dataframe['obv'].shift(1))
        ]
        
        if all(condition.dtype == 'bool' for condition in conditions_long):
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                'enter_long'
            ] = 1
        
        # Add trend following sell signals
        conditions_short = [
            (dataframe['close'] < dataframe['trend']),
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)),
            (dataframe['obv'] < dataframe['obv'].shift(1))
        ]
        
        if all(condition.dtype == 'bool' for condition in conditions_short):
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_short),
                'enter_short'
            ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Add trend following exit signals for long positions
        conditions_exit_long = [
            (dataframe['close'] < dataframe['trend']),
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)),
            (dataframe['obv'] > dataframe['obv'].shift(1))
        ]
        
        if all(condition.dtype == 'bool' for condition in conditions_exit_long):
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_exit_long),
                'exit_long'
            ] = 1
        
        # Add trend following exit signals for short positions
        conditions_exit_short = [
            (dataframe['close'] > dataframe['trend']),
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)),
            (dataframe['obv'] < dataframe['obv'].shift(1))
        ]
        
        if all(condition.dtype == 'bool' for condition in conditions_exit_short):
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_exit_short),
                'exit_short'
            ] = 1
        
        return dataframe
