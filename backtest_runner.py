# backtest_runner.py - Enhanced Standalone & Optuna-Compatible Backtester

import pandas as pd
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta 
import sys
import argparse 
from typing import Optional, Dict, Any
import optuna # KeyboardInterrupt'ı TrialPruned'a çevirmek için
import os


from utils.config import settings
from utils.logger import logger, ensure_csv_header
from utils.portfolio import Portfolio
from strategies.momentum_optimized import MomentumStrategy
import logging

# Bu dosyaya özel logları ana logger (algobot) üzerinden ama farklı bir isimle yapalım
backtest_runner_logger = logging.getLogger("algobot.backtest_runner")

class MomentumBacktester:
    """Optuna ile uyumlu ve tek başına çalışabilen Momentum Backtester"""

    def __init__(self, 
                 csv_path: str, 
                 initial_capital: float, 
                 start_date: str, 
                 end_date: str, 
                 symbol: str,
                 # Optuna'dan gelen örnekler için
                 portfolio_instance: Optional[Portfolio] = None,
                 strategy_instance: Optional[MomentumStrategy] = None
                ):
        self.csv_path = Path(csv_path)
        self.initial_capital = initial_capital
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.symbol = symbol if symbol else settings.SYMBOL
        
        self.start_date_dt = pd.to_datetime(start_date, utc=True) if start_date else None
        self.end_date_dt = pd.to_datetime(end_date, utc=True) if end_date else None
        
        # Eğer dışarıdan verilmemişse, kendi portfolyo ve stratejisini oluştur
        # Bu, standalone çalıştırma için gereklidir.
        if portfolio_instance:
            self.portfolio = portfolio_instance
        else:
            self.portfolio = Portfolio(initial_capital_usdt=self.initial_capital)
        
        if strategy_instance:
            self.strategy = strategy_instance
        else:
            self.strategy = MomentumStrategy(portfolio=self.portfolio, symbol=self.symbol)
        
        self.lookback_window = 0 
        self.total_bars = 0
        self.processed_bars = 0
        self.start_time = None
        
        backtest_runner_logger.info(f"Backtester instance created for '{self.symbol}'")


    # backtest_runner.py dosyasının içine, MomentumBacktester sınıfına eklenecek yeni metod

    def calculate_performance_metrics(self, portfolio_history: pd.Series, closed_trades: list) -> dict:
        """
        Backtest sonunda detaylı performans metrikleri hesaplar.
        """
        if portfolio_history.empty or not closed_trades:
            return {
                "max_drawdown_pct": 0.0,
                "sortino_ratio": 0.0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "avg_trade_duration_min": 0.0,
            }

        # Maksimum Düşüş (Max Drawdown) Hesaplanması
        peak = portfolio_history.expanding(min_periods=1).max()
        drawdown = (portfolio_history - peak) / peak
        max_drawdown_pct = abs(drawdown.min()) * 100

        # Sortino Oranı Hesaplanması
        daily_returns = portfolio_history.pct_change(1).dropna()
        # Sadece negatif getirileri (kayıpları) dikkate al
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std()
        
        # Eğer hiç kayıp yoksa veya getiri yoksa Sortino tanımsız olur, 0 kabul et
        if downside_std == 0 or daily_returns.mean() == 0:
            sortino_ratio = 0.0
        else:
            # Yıllıklandırılmış Sortino Oranı (günlük veri için 365, saatlik için farklı)
            # Bizim verimiz 15 dakikalık olduğu için, periyot sayısına göre ayarlayalım
            # Günde 96 tane 15dk'lık bar var. Yıllık periyot = 96 * 365
            annualization_factor = (96 * 365) ** 0.5 
            sortino_ratio = (daily_returns.mean() / downside_std) * annualization_factor

        # Kâr Faktörü (Profit Factor) ve Kazanma Oranı (Win Rate)
        total_profit = sum(trade['pnl_usdt'] for trade in closed_trades if trade['pnl_usdt'] > 0)
        total_loss = abs(sum(trade['pnl_usdt'] for trade in closed_trades if trade['pnl_usdt'] < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 999.0 # Kayıp yoksa çok yüksek bir değer ata
        
        winning_trades = sum(1 for trade in closed_trades if trade['pnl_usdt'] > 0)
        win_rate = (winning_trades / len(closed_trades)) * 100 if closed_trades else 0.0
        
        # Ortalama işlem süresi
        avg_trade_duration_min = sum(trade.get('hold_duration_minutes', 0) for trade in closed_trades) / len(closed_trades) if closed_trades else 0.0

        return {
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "profit_factor": round(profit_factor, 2),
            "win_rate": round(win_rate, 2),
            "avg_trade_duration_min": round(avg_trade_duration_min, 2),
        }
    
    def load_data(self) -> pd.DataFrame:
        backtest_runner_logger.info(f"Loading data from '{self.csv_path}' for period: {self.start_date_str or 'earliest'} to {self.end_date_str or 'latest'}")
        try:
            df = pd.read_csv(self.csv_path)
            # Zaman damgası sütununu bul ve dönüştür
            timestamp_cols = ['timestamp', 'Timestamp', 'time', 'Time', 'Date', 'date', 'datetime', 'Datetime']
            actual_timestamp_col = next((col for col in timestamp_cols if col in df.columns), None)
            if not actual_timestamp_col:
                raise ValueError(f"Timestamp column not found. Looked for: {timestamp_cols}. Found: {df.columns.tolist()}")

            df[actual_timestamp_col] = pd.to_datetime(df[actual_timestamp_col], errors='coerce', utc=True)
            df = df.dropna(subset=[actual_timestamp_col])
            
            df = df.rename(columns={actual_timestamp_col: 'timestamp'}).set_index('timestamp').sort_index()

            df.columns = df.columns.str.lower()
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns {missing_cols}. Found: {df.columns.tolist()}")

            # Tarihe göre filtrele
            if self.start_date_dt: df = df[df.index >= self.start_date_dt]
            if self.end_date_dt:
                end_filter = pd.Timestamp(self.end_date_dt).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                df = df[df.index <= end_filter]
            
            if df.empty:
                backtest_runner_logger.warning(f"No data after filtering for period: {self.start_date_str} to {self.end_date_str}.")
                return pd.DataFrame()

            backtest_runner_logger.info(f"Loaded and filtered {len(df):,} bars from {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            backtest_runner_logger.error(f"Failed to load/process data: {e}", exc_info=True)
            raise

    # backtest_runner.py dosyasındaki run_backtest metodunu bununla değiştir.
    async def run_backtest(self) -> Dict[str, Any]:
        backtest_runner_logger.info(f"Starting backtest run for {self.symbol}...")
    
        df = self.load_data()
        if df.empty:
            # Önceki adımdaki calculate_performance_metrics'i burada da kullanabiliriz.
            return self.calculate_performance_metrics(pd.Series(), [])

        self.total_bars = len(df)
        self.start_time = datetime.now()
        self.processed_bars = 0

        self.lookback_window = max(50, getattr(self.strategy, 'ema_long', 50) + 10)
    
        if self.total_bars < self.lookback_window:
            backtest_runner_logger.error(f"Not enough data ({self.total_bars}) for lookback window ({self.lookback_window}).")
            return self.calculate_performance_metrics(pd.Series(), [])

        backtest_runner_logger.info(f"Processing {self.total_bars - self.lookback_window:,} effective bars...")

        # Portföy geçmişini izlemeye başla (yeni metrikler için gerekli)
        if hasattr(self.portfolio, 'portfolio_value_history'):
            # Portfolio value history var - başlangıç değerini ekle
            if not self.portfolio.portfolio_value_history:
                self.portfolio.portfolio_value_history.append(self.portfolio.initial_capital_usdt)
        else:
            # Portfolio value history yok - varsayılan değer kullan
            logger.debug("Portfolio value history not available, using default values")
        
        final_price = df['close'].iloc[-1]
        try:
            for i in range(self.lookback_window, len(df)): 
                data_window = df.iloc[i-self.lookback_window:i+1]
                current_bar = df.iloc[i]
                final_price = current_bar['close']
                self.strategy._current_backtest_time = current_bar.name.to_pydatetime()
            
                await self.strategy.process_data(data_window)
            
                # YENİ EKLENEN SATIR: Her adımdan sonra portföy değerini kaydet
                self.portfolio.track_portfolio_value(final_price)
            
                self.processed_bars += 1

                if self.processed_bars % 2000 == 0 or self.processed_bars == (self.total_bars - self.lookback_window): 
                    await self._log_progress(self.strategy._current_backtest_time, final_price)
        
            backtest_runner_logger.info(f"Backtest run completed. Processed {self.processed_bars} bars.")
    
        except KeyboardInterrupt:
            backtest_runner_logger.warning("Backtest run interrupted by user (KeyboardInterrupt).")
            raise optuna.exceptions.TrialPruned("Backtest run interrupted by user.")
        except Exception as e:
            backtest_runner_logger.error(f"Backtest run failed: {e}", exc_info=True)
            # Hata durumunda bile temel metrikleri almayı dene
            results = self.portfolio.get_performance_summary(final_price)
            results["error_in_backtest"] = str(e)
            return results
    
        # ### --- SONUÇLARI DÖNDÜRME KISMI GÜNCELLENDİ --- ###
    
        # 1. Portföyden temel özet bilgilerini al
        results = self.portfolio.get_performance_summary(final_price)

        # 2. Adım 1.1'de eklediğimiz fonksiyonu kullanarak gelişmiş metrikleri hesapla
        portfolio_history_series = pd.Series(self.portfolio.portfolio_value_history)
        closed_trades_list = self.portfolio.get_closed_trades_for_summary()
        performance_stats = self.calculate_performance_metrics(portfolio_history_series, closed_trades_list)

        # 3. İki sonuç sözlüğünü birleştir
        results.update(performance_stats)

        backtest_runner_logger.info(f"Gelişmiş metrikler hesaplandı: Sortino={results.get('sortino_ratio')}, Drawdown={results.get('max_drawdown_pct')}%")

        # 4. Tüm metrikleri içeren birleştirilmiş sonuçları döndür
        return results


    async def _log_progress(self, current_time: datetime, current_price: float):
        effective_total_bars = self.total_bars - self.lookback_window
        progress_pct = (self.processed_bars / effective_total_bars) * 100 if effective_total_bars > 0 else 100
        elapsed_time = datetime.now() - self.start_time
        bars_per_sec = self.processed_bars / elapsed_time.total_seconds() if elapsed_time.total_seconds() > 0 else 0
        portfolio_value = self.portfolio.get_total_portfolio_value_usdt(current_price)
        profit_pct = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        backtest_runner_logger.info(
            f"Progress: {progress_pct:5.1f}% | Date: {current_time.strftime('%Y-%m-%d')} | "
            f"Portfolio: ${portfolio_value:,.0f} ({profit_pct:+.1f}%) | "
            f"Trades: {len(self.portfolio.closed_trades)} | Speed: {bars_per_sec:.0f} bars/s"
        )

# Bu dosya doğrudan çalıştırıldığında kullanılacak ana fonksiyon
async def standalone_backtest_main():
    parser = argparse.ArgumentParser(description="Standalone Momentum Backtest Runner")
    parser.add_argument("--symbol", type=str, default=settings.SYMBOL)
    parser.add_argument("--timeframe", type=str, default=settings.TIMEFRAME)
    parser.add_argument("--start-date", type=str, default="2024-03-01")
    parser.add_argument("--end-date", type=str, default="2024-05-31")
    parser.add_argument("--initial-capital", type=float, default=settings.INITIAL_CAPITAL_USDT)
    parser.add_argument("--data-file", type=str, required=True, help="Path to the historical data CSV file")
    args = parser.parse_args()

    logger.info(f"🚀 Starting Standalone Momentum Backtest for symbol {args.symbol}")
    backtest_runner_logger.info(f"Standalone Backtest initiated with args: {vars(args)}")
    
    data_file_path = Path(args.data_file)
    if not data_file_path.exists():
        logger.error(f"❌ Data file not found at: {data_file_path}")
        return

    # Standalone çalışırken CSV loglamasını aç ve özel bir dosya kullan
    original_csv_path = settings.TRADES_CSV_LOG_PATH
    settings.ENABLE_CSV_LOGGING = True
    standalone_csv_path_str = str(Path("logs") / f"trades_standalone_{args.symbol.replace('/','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    settings.TRADES_CSV_LOG_PATH = standalone_csv_path_str
    ensure_csv_header(standalone_csv_path_str) 
    
    # Strateji parametreleri config'den alınacak (Optuna'daki gibi override edilmeyecek)
    backtester = MomentumBacktester(
        csv_path=str(data_file_path),
        initial_capital=args.initial_capital,
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol
    )
    
    try:
        results = await backtester.run_backtest()
        
        if results and "error_in_backtest" not in results:
            logger.info("\n" + "="*80)
            logger.info("🏁 STANDALONE BACKTEST FINAL REPORT")
            logger.info("="*80)
            for key, value in results.items():
                if isinstance(value, float):
                    if "pct" in key.lower() or "rate" in key.lower() or "factor" in key.lower():
                        logger.info(f"   {key.replace('_', ' ').title():<25}: {value:,.2f}%")
                    else:
                        logger.info(f"   {key.replace('_', ' ').title():<25}: ${value:,.2f}")
                else:
                    logger.info(f"   {key.replace('_', ' ').title():<25}: {value}")
            logger.info("="*80)
            logger.info(f"Trade details for this run are logged in: {standalone_csv_path_str}")
        else:
            logger.error(f"Standalone backtest finished with an error: {results.get('error_in_backtest', 'Unknown error')}")

    except KeyboardInterrupt:
        logger.warning("🛑 Standalone backtest interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Standalone backtest failed: {e}", exc_info=True)
    finally:
        settings.TRADES_CSV_LOG_PATH = original_csv_path


if __name__ == "__main__":
    asyncio.run(standalone_backtest_main())