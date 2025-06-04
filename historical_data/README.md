# 📊 Historical Data Directory

This directory contains historical cryptocurrency price data for backtesting.

## 🔄 How to Download Data

Use the data downloader script to fetch historical data:

```bash
# Download BTC/USDT data for 2024
python data_downloader.py --symbol BTC/USDT --timeframe 15m --startdate 2024-01-01 --enddate 2024-12-31

# Download specific date range
python data_downloader.py --symbol BTC/USDT --timeframe 15m --startdate 2024-03-01 --enddate 2024-04-30
```

## 📁 File Naming Convention

Files are automatically named using this pattern:
```
{SYMBOL}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv
```

Examples:
- `BTCUSDT_15m_20240101_20241231.csv`
- `ETHUSDT_1h_20240301_20240430.csv`

## 📋 CSV Format

Each CSV file contains the following columns:
- `timestamp`: ISO 8601 timestamp with timezone
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## ⚠️ Note

- CSV files are excluded from Git due to size limitations
- Directory structure is preserved with `.gitkeep`
- Download data locally before running backtests
