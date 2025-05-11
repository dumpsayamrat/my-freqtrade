# Freqtrade Bot

This is a trading bot based on the [Freqtrade](https://www.freqtrade.io/) framework.

## Setup Instructions

### Configuration

This project uses a split configuration approach to keep sensitive data out of version control:

1. `config.json` - Contains general, non-sensitive settings that can be safely committed to GitHub
2. `config-private.json` - Contains sensitive information like API keys and passwords (not tracked in git)
3. `config-private.json.example` - A template for creating your own private config file

### First-time Setup

1. Clone this repository
2. Copy `user_data/config-private.json.example` to `user_data/config-private.json`
3. Edit `user_data/config-private.json` and add your exchange API keys, Telegram settings, etc.

### Data Files

- Trading data is stored in the `user_data/data/` directory (ignored by git)
- Backtest results are stored in `user_data/backtest_results/` (ignored by git)
- Trade history is stored in `user_data/tradesv3.sqlite` (ignored by git)

## Running the Bot

```bash
docker-compose up -d
```

## Development and Contributions

Feel free to submit pull requests. Make sure not to commit any sensitive data!

To run a backtest for your NFI5MOHO strategy using Docker Compose, you can use the following command:

```bash
docker-compose run --rm freqtrade backtesting --strategy TrendFollowingStrategy --timeframe 5m --timerange 20250201-
```


```bash
docker-compose run --rm freqtrade download-data --config user_data/config.json --timerange 20250201- --timeframe 5m 1h

docker-compose run --rm freqtrade download-data --config user_data/config.json --timerange 20240511-20250201 --timeframe 1h --prepend
```

```bash
 docker-compose run --rm  freqtrade hyperopt --config user_data/config.json --timerange 20250401- --hyperopt-loss SharpeHyperOptLoss --strategy TrendFollowingStrategy -e 100 --spaces roi stoploss trailing -j 30
```

```bash
docker-compose run --rm freqtrade plot-dataframe --strategy PinbarStrategy --timeframe 5m --pair BTC/USDT:USDT --timerange 20250201-

```
