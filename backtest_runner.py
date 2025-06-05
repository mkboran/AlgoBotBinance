# backtest_runner.py - Enhanced Momentum-Only Backtest

import pandas as pd
import asyncio
from pathlib import Path
from datetime import datetime, timezone
import sys
from typing import Optional

from utils.config import settings
from utils.logger import logger
from utils.portfolio import Portfolio
from strategies.momentum_optimized import MomentumStrategy

class MomentumBacktester:
    """🚀 Momentum-Only Backtest Runner"""
    
    def __init__(self, csv_path: str, initial_capital: float = 1000.0):
        self.csv_path = Path(csv_path)
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital_usdt=initial_capital)
        self.strategy = MomentumStrategy(portfolio=self.portfolio, symbol="BTC/USDT")
        
        # Performance tracking
        self.total_bars = 0
        self.processed_bars = 0
        self.start_time = None
        self.last_progress_log = 0
        
    def load_data(self) -> pd.DataFrame:
        """Load historical data from CSV"""
        logger.info(f"📊 Loading data from {self.csv_path}")
        
        try:
            df = pd.read_csv(self.csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    async def run_backtest(self) -> dict:
        """Run the momentum-only backtest"""
        logger.info("🚀 Starting Momentum-Only Backtest")
        logger.info(f"💰 Initial Capital: ${self.initial_capital:,.2f} USDT")
        
        # Load data
        df = self.load_data()
        self.total_bars = len(df)
        self.start_time = datetime.now()
        
        # Setup strategy with backtest mode
        lookback_window = 50  # Need enough data for indicators
        
        logger.info(f"📈 Processing {self.total_bars:,} bars...")
        logger.info(f"🔄 Lookback window: {lookback_window} bars")
        
        try:
            for i in range(lookback_window, len(df)):
                # Get data window for strategy
                data_window = df.iloc[i-lookback_window:i+1].copy()
                current_bar = df.iloc[i]
                current_time = current_bar.name
                
                # Set backtest time for strategy
                self.strategy._current_backtest_time = current_time
                
                # Process data through strategy
                await self.strategy.process_data(data_window)
                
                self.processed_bars += 1
                
                # Progress logging (every 1000 bars)
                if self.processed_bars % 1000 == 0:
                    await self._log_progress(current_time, current_bar['close'])
            
            # Final analysis
            final_price = df.iloc[-1]['close']
            final_stats = await self._generate_final_report(final_price)
            
            return final_stats
            
        except KeyboardInterrupt:
            logger.info("🛑 Backtest interrupted by user")
            final_price = df.iloc[self.processed_bars + lookback_window - 1]['close'] if self.processed_bars > 0 else df.iloc[-1]['close']
            return await self._generate_final_report(final_price)
        except Exception as e:
            logger.error(f"❌ Backtest error: {e}")
            raise
    
    async def _log_progress(self, current_time: datetime, current_price: float):
        """Log backtest progress"""
        progress_pct = (self.processed_bars / (self.total_bars - 50)) * 100
        elapsed = datetime.now() - self.start_time
        
        # Portfolio metrics
        portfolio_value = self.portfolio.get_total_portfolio_value_usdt(current_price)
        profit_pct = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        # Speed metrics
        bars_per_sec = self.processed_bars / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        logger.info(f"📊 Progress: {progress_pct:5.1f}% | "
                   f"Date: {current_time.strftime('%Y-%m-%d %H:%M')} | "
                   f"BTC: ${current_price:,.0f} | "
                   f"Portfolio: ${portfolio_value:,.0f} ({profit_pct:+.1f}%) | "
                   f"Trades: {len(self.portfolio.closed_trades)} | "
                   f"Speed: {bars_per_sec:.0f} bars/s")
    
    async def _generate_final_report(self, final_price: float) -> dict:
        """Generate comprehensive final report"""
        logger.info("\n" + "="*80)
        logger.info("🏁 MOMENTUM BACKTEST FINAL REPORT")
        logger.info("="*80)
        
        # Basic metrics
        final_value = self.portfolio.get_total_portfolio_value_usdt(final_price)
        total_profit = final_value - self.initial_capital
        total_profit_pct = (total_profit / self.initial_capital) * 100
        
        # Trade statistics
        trades = self.portfolio.closed_trades
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = [t for t in trades if t["profit_usd"] > 0]
            losing_trades = [t for t in trades if t["profit_usd"] <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades) * 100
            
            total_wins = sum(t["profit_usd"] for t in winning_trades)
            total_losses = abs(sum(t["profit_usd"] for t in losing_trades))
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            avg_win = total_wins / win_count if win_count > 0 else 0
            avg_loss = total_losses / loss_count if loss_count > 0 else 0
            
            max_win = max((t["profit_usd"] for t in winning_trades), default=0)
            max_loss = min((t["profit_usd"] for t in losing_trades), default=0)
            
            # Hold times
            hold_times = [t.get("hold_minutes", 0) for t in trades]
            avg_hold_time = sum(hold_times) / len(hold_times)
        else:
            win_rate = profit_factor = avg_win = avg_loss = max_win = max_loss = avg_hold_time = 0
            win_count = loss_count = 0
            total_wins = total_losses = 0
        
        # Performance summary
        performance_emoji = "🚀" if total_profit_pct > 10 else "💰" if total_profit_pct > 0 else "📉"
        
        logger.info(f"💰 PORTFOLIO PERFORMANCE:")
        logger.info(f"   Initial Capital:    ${self.initial_capital:>10,.2f} USDT")
        logger.info(f"   Final Value:        ${final_value:>10,.2f} USDT")
        logger.info(f"   Total P&L:          ${total_profit:>+10,.2f} USDT ({total_profit_pct:>+6.2f}%) {performance_emoji}")
        logger.info(f"   Final BTC Price:    ${final_price:>10,.2f} USDT")
        
        if total_trades > 0:
            logger.info(f"\n📊 TRADING STATISTICS:")
            logger.info(f"   Total Trades:       {total_trades:>10}")
            logger.info(f"   Winning Trades:     {win_count:>10} ({win_rate:>5.1f}%)")
            logger.info(f"   Losing Trades:      {loss_count:>10} ({100-win_rate:>5.1f}%)")
            logger.info(f"   Profit Factor:      {profit_factor:>10.2f}")
            logger.info(f"   Average Hold Time:  {avg_hold_time:>10.1f} minutes")
            
            logger.info(f"\n💵 PROFIT BREAKDOWN:")
            logger.info(f"   Total Wins:         ${total_wins:>+10.2f} USDT")
            logger.info(f"   Total Losses:       ${-total_losses:>+10.2f} USDT")
            logger.info(f"   Average Win:        ${avg_win:>+10.2f} USDT")
            logger.info(f"   Average Loss:       ${-avg_loss:>+10.2f} USDT")
            logger.info(f"   Best Trade:         ${max_win:>+10.2f} USDT")
            logger.info(f"   Worst Trade:        ${max_loss:>+10.2f} USDT")
        
        # Runtime statistics
        elapsed = datetime.now() - self.start_time if self.start_time else datetime.now()
        bars_per_sec = self.processed_bars / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        logger.info(f"\n⏱️  BACKTEST PERFORMANCE:")
        logger.info(f"   Processed Bars:     {self.processed_bars:>10,}")
        logger.info(f"   Runtime:            {elapsed}")
        logger.info(f"   Speed:              {bars_per_sec:>10.0f} bars/second")
        
        logger.info("="*80)
        
        # Return stats dictionary
        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_profit": total_profit,
            "total_profit_pct": total_profit_pct,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_win": max_win,
            "max_loss": max_loss,
            "avg_hold_time": avg_hold_time,
            "processed_bars": self.processed_bars,
            "runtime_seconds": elapsed.total_seconds()
        }

async def main():
    """Main backtest execution"""
    # Find historical data file
    data_dir = Path("historical_data")
    data_files = list(data_dir.glob("BTCUSDT_5m_*2024*.csv"))
    
    if not data_files:
        print("❌ No historical data found! Please run:")
        print("python data_downloader.py --symbol BTC/USDT --timeframe 5m --startdate 2024-03-01 --enddate 2024-12-31")
        return
    
    # Use the most recent file
    data_file = sorted(data_files)[-1]
    
    logger.info(f"🚀 Starting Momentum-Only Backtest")
    logger.info(f"📁 Data file: {data_file}")
    
    # Create and run backtest
    backtester = MomentumBacktester(data_file, initial_capital=1000.0)
    
    try:
        results = await backtester.run_backtest()
        
        # Performance rating
        profit_pct = results["total_profit_pct"]
        if profit_pct > 50:
            rating = "🌟 EXCEPTIONAL"
        elif profit_pct > 20:
            rating = "🚀 EXCELLENT"
        elif profit_pct > 10:
            rating = "💎 GREAT"
        elif profit_pct > 5:
            rating = "👍 GOOD"
        elif profit_pct > 0:
            rating = "⚖️ POSITIVE"
        elif profit_pct > -10:
            rating = "👎 POOR"
        else:
            rating = "🔴 CRITICAL"
        
        print(f"\n🏆 BACKTEST RATING: {rating}")
        print(f"💰 FINAL RESULT: {results['total_profit']:+.2f} USD ({results['total_profit_pct']:+.2f}%)")
        
    except KeyboardInterrupt:
        logger.info("🛑 Backtest interrupted by user")
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())