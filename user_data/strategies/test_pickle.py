import pickle
from CryptoFrog import CryptoFrog  # Adjust this import

# Create a mock config (you can adjust it as needed)
mock_config = {
    "$schema": "https://schema.freqtrade.io/schema.json",
    "max_open_trades": 6,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",
    "dry_run": True,
    "dry_run_wallet": 1000,
    "timeframe": "5m",
    "cancel_open_orders_on_exit": False,
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "unfilledtimeout": {
        "entry": 10,
        "exit": 10,
        "exit_timeout_count": 0,
        "unit": "minutes"
    },
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": True,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": False,
            "bids_to_ask_delta": 1
        }
    },
    "exit_pricing": {
        "price_side": "same",
        "use_order_book": True,
        "order_book_top": 1
    },
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "ccxt_config": {},
        "ccxt_async_config": {},
        "pair_whitelist": [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
            "XRP/USDT:USDT",
            "ADA/USDT:USDT",
            "MATIC/USDT:USDT",
            "TRX/USDT:USDT",
            "AVAX/USDT:USDT",
            "DOGE/USDT:USDT",
            "SHIB/USDT:USDT",
            "XLM/USDT:USDT",
            "FIL/USDT:USDT",
            "LDO/USDT:USDT",
            "SAND/USDT:USDT",
            "MANA/USDT:USDT",
            "AAVE/USDT:USDT",
            "CRV/USDT:USDT",
            "COMP/USDT:USDT",
            "YFI/USDT:USDT",
            "ALGO/USDT:USDT",
            "ZRX/USDT:USDT",
            "BAT/USDT:USDT",
            "CHZ/USDT:USDT",
            "1INCH/USDT:USDT",
            "SUSHI/USDT:USDT",
            "FTM/USDT:USDT"
        ],
        "pair_blacklist": [
            "BNB/.*"
        ]
    },
    "pairlists": [
        {
            "method": "StaticPairList"
        }
    ],
    "telegram": {
        "enabled": False,
        "token": "",
        "chat_id": ""
    },
    "api_server": {
        "enabled": True,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080,
        "verbosity": "error",
        "enable_openapi": False,
        "jwt_secret_key": "",
        "ws_token": "",
        "CORS_origins": [],
        "username": "",
        "password": ""
    },
    "bot_name": "freqtrade",
    "initial_state": "running",
    "force_entry_enable": False,
    "internals": {
        "process_throttle_secs": 5
    }
}


# Pass the mock config to the strategy
try:
    strategy_instance = CryptoFrog(config=mock_config)
    pickle.dumps(strategy_instance)
    print("Strategy is serializable.")
except Exception as e:
    print("Serialization failed:", e)
