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

## Production Deployment

For production environments, use the production docker-compose configuration which includes:
- Stable version of freqtrade
- API exposed on all network interfaces (0.0.0.0)
- Production-specific configuration

```bash
docker-compose -f docker-compose.prod.yml up -d
```

To stop the production service:

```bash
docker-compose -f docker-compose.prod.yml down
```

To view logs from the production service:

```bash
docker logs freqtrade-prod
```

⚠️ **Security Note**: The production configuration exposes the API on all network interfaces. Ensure proper security measures are in place if running on a public server.

## Development and Contributions

Feel free to submit pull requests. Make sure not to commit any sensitive data!

To run a backtest for your NFI5MOHO strategy using Docker Compose, you can use the following command:

```bash
docker-compose run --rm freqtrade backtesting --config /freqtrade/user_data/config-backtest.json --strategy TrendFollowingStrategy --timeframe 5m --timerange 20250201-
```

```bash
docker-compose run --rm freqtrade download-data --config /freqtrade/user_data/config-backtest.json --timerange 20250201- --timeframe 5m 1h

docker-compose run --rm freqtrade download-data --config /freqtrade/user_data/config-backtest.json --timerange 20240511-20250201 --timeframe 1h --prepend
```

```bash
 docker-compose run --rm  freqtrade hyperopt --config /freqtrade/user_data/config-backtest.json --timerange 20250401- --hyperopt-loss SharpeHyperOptLoss --strategy TrendFollowingStrategy -e 100 --spaces roi stoploss trailing -j 30
```

```bash
docker-compose run --rm freqtrade plot-dataframe --config /freqtrade/user_data/config-backtest.json --strategy PinbarStrategy --timeframe 5m --pair BTC/USDT:USDT --timerange 20250201-
```

## Production Commands

To run commands using the production configuration, use the `-f docker-compose.prod.yml` flag:

```bash
# Run backtesting with production configuration
docker-compose -f docker-compose.prod.yml run --rm freqtrade backtesting --config /freqtrade/user_data/config-prod.json --strategy CryptoFrog --timeframe 1h --timerange 20250201-

# Download data with production configuration
docker-compose -f docker-compose.prod.yml run --rm freqtrade download-data --config /freqtrade/user_data/config-prod.json --timerange 20250201- --timeframe 1h 4h

# Plot dataframe with production configuration
docker-compose -f docker-compose.prod.yml run --rm freqtrade plot-dataframe --config /freqtrade/user_data/config-prod.json --strategy CryptoFrog --timeframe 1h --pair BTC/USDT:USDT --timerange 20250201-
```
