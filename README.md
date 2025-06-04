# 🚀 Enhanced AlgoBot - Advanced Cryptocurrency Trading System

## 📊 Overview
Advanced cryptocurrency trading bot with enhanced momentum strategies, risk management, and comprehensive backtesting capabilities.

## ✨ Features
- **Enhanced Momentum Strategy**: Multi-timeframe momentum analysis
- **Advanced Risk Management**: Dynamic position sizing and stop-loss
- **Comprehensive Backtesting**: Historical data testing with detailed analytics
- **Real-time Trading**: Live trading with Binance API integration
- **Portfolio Management**: Advanced position and trade tracking
- **Performance Analytics**: Detailed trade analysis and reporting

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/AlgoBotBinance.git
cd AlgoBotBinance

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp config.json.example config.json
# Edit config.json with your API keys
```

## 📈 Quick Start

### Backtesting
```bash
# Download historical data
python data_downloader.py --symbol BTC/USDT --timeframe 15m --startdate 2024-01-01 --enddate 2024-12-31

# Run backtest
python backtest_runner.py --symbol BTC/USDT --timeframe 15m --start-date 2024-03-01 --end-date 2024-04-30 --data-file historical_data/BTCUSDT_15m_20240101_20241231.csv
```

### Live Trading
```bash
# Start live trading
python main.py
```

## 📁 Project Structure
```
AlgoBotBinance/
├── strategies/           # Trading strategies
│   ├── momentum_optimized.py
│   └── base_strategy.py
├── utils/               # Utility modules
│   ├── portfolio.py     # Portfolio management
│   ├── risk.py         # Risk management
│   ├── config.py       # Configuration
│   └── logger.py       # Logging system
├── historical_data/     # Historical price data
├── logs/               # Trading logs
├── backtest_runner.py  # Backtesting system
├── data_downloader.py  # Data download utility
└── main.py            # Main trading bot
```

## ⚙️ Configuration

### API Keys
Edit `config.json`:
```json
{
    "binance": {
        "api_key": "your_binance_api_key",
        "secret_key": "your_binance_secret_key",
        "testnet": true
    },
    "trading": {
        "symbol": "BTC/USDT",
        "timeframe": "15m",
        "trade_amount_usdt": 190.0
    }
}
```

## 📊 Performance Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

## 🔧 Advanced Features
- Multi-strategy support
- Dynamic position sizing
- Real-time risk monitoring
- Comprehensive logging
- Performance analytics
- Market condition adaptation

## 📋 Requirements
- Python 3.8+
- pandas, numpy, ccxt
- python-binance
- asyncio support

## ⚠️ Disclaimer
This trading bot is for educational and research purposes. Cryptocurrency trading involves significant risk. Never trade with money you cannot afford to lose.

## 📄 License
MIT License - see LICENSE file for details

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support
- Open issues for bugs/features
- Discussion in repository discussions
- Documentation in Wiki

---
⭐ **Star this repository if you find it useful!** ⭐
