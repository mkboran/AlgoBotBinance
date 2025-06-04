# backtest_runner.py - Enhanced Backtest System with CLI Arguments

import pandas as pd
import numpy as np
import asyncio
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

from utils.config import settings
from utils.logger import logger
import logging

backtest_logger = logging.getLogger("algobot.backtest")
backtest_logger.setLevel(logging.INFO)
if not backtest_logger.handlers:
    Path("logs").mkdir(exist_ok=True)
    handler = logging.FileHandler("logs/backtest.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    backtest_logger.addHandler(handler)
    backtest_logger.propagate = False
from utils.portfolio import Portfolio
from utils.risk import RiskManager
from strategies.momentum_optimized import MomentumStrategy

class BacktestRunner:
    """🔬 Enhanced Backtest System"""
    
    def __init__(self, initial_capital: float = 1000):
        self.portfolio = Portfolio(initial_capital)
        self.risk_manager = RiskManager()
        self.strategies = [
            MomentumStrategy(portfolio=self.portfolio, symbol=settings.SYMBOL)
        ]
        
        # Backtest metrics
        self.trade_count = 0
        self.results = {
            "trades": [],
            "portfolio_history": [],
            "performance_metrics": {}
        }
        
        logger.info(f"✅ BacktestRunner initialized with ${initial_capital} capital")
        backtest_logger.info(f"INIT capital=${initial_capital}")
    
    async def run_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest on historical data"""
        logger.info(f"🚀 Starting backtest with {len(df)} data points")
        logger.info(f"📊 Data range: {df.index[0]} to {df.index[-1]}")
        backtest_logger.info(f"START {len(df)} bars {df.index[0]} to {df.index[-1]}")
        
        try:
            for i, (timestamp, row) in enumerate(df.iterrows()):
                # Create current data slice
                current_data = df.iloc[:i+1].copy()
                
                if len(current_data) < 50:  # Need minimum data for indicators
                    continue
                
                current_price = row['close']
                
                # Set current time for strategies
                for strategy in self.strategies:
                    strategy._current_backtest_time = timestamp
                
                # Process each strategy
                for strategy in self.strategies:
                    try:
                        await strategy.process_data(current_data)
                    except Exception as e:
                        logger.error(f"Strategy {strategy.strategy_name} error at {timestamp}: {e}")
                
                # Record portfolio state
                portfolio_value = self.portfolio.get_total_portfolio_value_usdt(current_price)
                self.results["portfolio_history"].append({
                    "timestamp": timestamp,
                    "portfolio_value": portfolio_value,
                    "price": current_price,
                    "positions": len(self.portfolio.positions)
                })
                
                # Progress logging
                if i % 500 == 0:
                    progress = (i / len(df)) * 100
                    logger.info(f"📊 Backtest progress: {progress:.1f}% ({i}/{len(df)})")
                    backtest_logger.info(f"PROGRESS {progress:.1f}% {i}/{len(df)}")
            
            # Calculate final results
            final_price = df['close'].iloc[-1]
            final_portfolio_value = self.portfolio.get_total_portfolio_value_usdt(final_price)
            
            # Performance metrics
            total_return = (final_portfolio_value - self.portfolio.initial_capital_usdt) / self.portfolio.initial_capital_usdt
            
            # Calculate win rate
            winning_trades = len([t for t in self.portfolio.closed_trades if t["profit_usd"] > 0])
            total_trades = len(self.portfolio.closed_trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            self.results["performance_metrics"] = {
                "initial_capital": self.portfolio.initial_capital_usdt,
                "final_value": final_portfolio_value,
                "total_return_pct": total_return * 100,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": win_rate
            }
            
            logger.info(f"✅ Backtest completed: {total_return*100:.2f}% return, {total_trades} trades")
            backtest_logger.info(
                f"DONE return={total_return*100:.2f}% trades={total_trades} win_rate={win_rate*100:.1f}%"
            )
            
            return self.results
            
        except KeyboardInterrupt:
            logger.info("🛑 Backtest interrupted by user")
            backtest_logger.info("INTERRUPTED")
            raise
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            backtest_logger.error(f"ERROR {e}")
            raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AlgoBot Backtest Runner")
    
    parser.add_argument("--symbol", type=str, default="BTC/USDT", 
                       help="Trading symbol (default: BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default="15m",
                       help="Timeframe (default: 15m)")
    parser.add_argument("--start-date", type=str, required=True,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategy", type=str, default="momentum",
                       help="Strategy to test (default: momentum)")
    parser.add_argument("--initial-capital", type=float, default=1000,
                       help="Initial capital in USDT (default: 1000)")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to historical data CSV file")
    
    return parser.parse_args()

async def run_test():
    """Run the backtest with CLI arguments"""
    try:
        args = parse_arguments()
        
        # Create necessary directories
        Path("historical_data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Log dosyalarını hazırla
        from utils.logger import ensure_csv_header
        ensure_csv_header(settings.TRADES_CSV_LOG_PATH)
        
        logger.info(f"🔧 Backtest Configuration:")
        logger.info(f"   Symbol: {args.symbol}")
        logger.info(f"   Timeframe: {args.timeframe}")
        logger.info(f"   Period: {args.start_date} to {args.end_date}")
        logger.info(f"   Strategy: {args.strategy}")
        logger.info(f"   Initial Capital: ${args.initial_capital}")
        logger.info(f"   Data File: {args.data_file}")
        backtest_logger.info(
            f"CONFIG symbol={args.symbol} timeframe={args.timeframe} start={args.start_date} end={args.end_date} capital={args.initial_capital}"
        )
        
        # Check if data file exists
        data_path = Path(args.data_file)
        if not data_path.exists():
            logger.error(f"Data file not found: {args.data_file}")
            
            # Try alternative file names
            alternative_files = [
                f"historical_data/{args.symbol.replace('/', '')}_{args.timeframe}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv",
                f"historical_data/BTCUSDT_{args.timeframe}_20240101_20241231.csv",
                f"historical_data/BTCUSDT_15m_20240301_20241231.csv"
            ]
            
            logger.info("🔍 Looking for alternative data files:")
            backtest_logger.info("MISSING_DATA looking for alternatives")
            for alt_file in alternative_files:
                alt_path = Path(alt_file)
                logger.info(f"   Checking: {alt_file} - {'✅ Found' if alt_path.exists() else '❌ Not found'}")
                backtest_logger.info(f"CHECK {alt_file} {'found' if alt_path.exists() else 'not_found'}")
                if alt_path.exists():
                    logger.info(f"✅ Using alternative file: {alt_file}")
                    backtest_logger.info(f"USING {alt_file}")
                    data_path = alt_path
                    break
            else:
                logger.error("❌ No data files found. Please run data downloader first:")
                logger.error(f"   python data_downloader.py --symbol {args.symbol} --timeframe {args.timeframe} --startdate {args.start_date} --enddate {args.end_date}")
                backtest_logger.info("NO_DATA_ABORT")
                logger.info("\n📁 Expected file location:")
                expected_file = f"historical_data/{args.symbol.replace('/', '')}_{args.timeframe}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv"
                logger.info(f"   {expected_file}")
                return
        
        # Load and prepare data
        logger.info(f"📊 Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
        else:
            logger.error("No 'timestamp' column found in data file")
            return
        
        # ⚡ TZ-AWARE DATETIME KARŞILAŞTIRMASI - SORUN ÇÖZÜLDİ!
        start_date = pd.to_datetime(args.start_date, utc=True)
        end_date = pd.to_datetime(args.end_date, utc=True)
        
        # Sadece ilgili tarih aralığını filtrele
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df.empty:
            logger.error(f"No data found in specified date range: {args.start_date} to {args.end_date}")
            return
        
        logger.info(f"📊 Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        backtest_logger.info(f"DATA_LOADED rows={len(df)}")
        
        # Update settings for backtest
        settings.SYMBOL = args.symbol
        settings.TIMEFRAME = args.timeframe
        
        # Run backtest
        runner = BacktestRunner(initial_capital=args.initial_capital)
        results = await runner.run_backtest(df)
        
        # Print results
        print("\n" + "="*60)
        print("📈 BACKTEST RESULTS")
        print("="*60)
        
        metrics = results["performance_metrics"]
        print(f"Initial Capital:     ${metrics['initial_capital']:,.2f}")
        print(f"Final Value:         ${metrics['final_value']:,.2f}")
        print(f"Total Return:        {metrics['total_return_pct']:+.2f}%")
        print(f"Total Trades:        {metrics['total_trades']}")
        print(f"Winning Trades:      {metrics['winning_trades']}")
        print(f"Losing Trades:       {metrics['losing_trades']}")
        print(f"Win Rate:            {metrics['win_rate']*100:.1f}%")
        
        if metrics['total_trades'] > 0:
            total_profit = sum(t['profit_usd'] for t in runner.portfolio.closed_trades)
            avg_profit = total_profit / metrics['total_trades']
            print(f"Average P&L/Trade:   ${avg_profit:+.2f}")
            
            # En iyi ve en kötü trade
            best_trade = max(runner.portfolio.closed_trades, key=lambda x: x['profit_usd'])
            worst_trade = min(runner.portfolio.closed_trades, key=lambda x: x['profit_usd'])
            print(f"Best Trade:          ${best_trade['profit_usd']:+.2f} ({best_trade['profit_pct']*100:+.1f}%)")
            print(f"Worst Trade:         ${worst_trade['profit_usd']:+.2f} ({worst_trade['profit_pct']*100:+.1f}%)")
        
        print("="*60)
        
        # Save detailed results
        results_file = f"backtest_results_{args.symbol.replace('/', '')}_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Backtest Results\n")
            f.write(f"================\n")
            f.write(f"Symbol: {args.symbol}\n")
            f.write(f"Strategy: {args.strategy}\n")
            f.write(f"Period: {args.start_date} to {args.end_date}\n")
            f.write(f"Initial Capital: ${args.initial_capital}\n")
            f.write(f"Final Value: ${metrics['final_value']:.2f}\n")
            f.write(f"Total Return: {metrics['total_return_pct']:+.2f}%\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
            f.write(f"Win Rate: {metrics['win_rate']*100:.1f}%\n\n")
            
            f.write("Trade Details:\n")
            f.write("==============\n")
            for i, trade in enumerate(runner.portfolio.closed_trades, 1):
                f.write(f"Trade {i:3d}: {trade['profit_usd']:+7.2f} USD ({trade['profit_pct']*100:+6.2f}%) - {trade['exit_reason']}\n")
        
        logger.info(f"📋 Detailed results saved to: {results_file}")
        backtest_logger.info(f"RESULTS_SAVED {results_file}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Backtest interrupted by user")
        backtest_logger.info("INTERRUPTED")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        backtest_logger.error(f"ERROR {e}")
        raise

def main():
    """Main entry point"""
    asyncio.run(run_test())

if __name__ == "__main__":
    main()