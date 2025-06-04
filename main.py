# main.py
import asyncio
import signal
import sys
from pathlib import Path
from typing import List, Union
from datetime import datetime, timedelta, timezone

# config ve logger en başta import edilmeli
from utils.config import settings
from utils.logger import logger, ensure_csv_header, log_performance_summary, log_system_startup, log_system_shutdown

import ccxt.async_support as ccxt
from utils.data import BinanceFetcher, DataFetchingError
from utils.portfolio import Portfolio
from utils.risk import RiskManager

# Bağımsız stratejileri import et
from strategies.momentum_optimized import MomentumStrategy
from strategies.bollinger_rsi_strategy import BollingerRsiStrategy


# Global shutdown event
shutdown_event = asyncio.Event()

def handle_shutdown_signal(sig, frame):
    """Kapatma sinyallerini (SIGINT, SIGTERM) yakalar ve shutdown_event'i set eder."""
    if not shutdown_event.is_set(): # Birden fazla sinyal gelirse tekrar loglamamak için
        logger.info(f"🛑 Kapatma sinyali {sig} alındı. Final analiz hazırlanıyor...")
        shutdown_event.set()
        
        # CTRL+C basıldığında hemen final analizi göster
        print("\n" + "="*80)
        print("🛑 İŞLEM DURDURULUYOR - FINAL ANALİZ HAZIRLANIYOR...")
        print("="*80)
    else:
        print("🛑 Zaten kapatılıyor, lütfen bekleyin...")

def calculate_final_statistics(portfolio: Portfolio, current_price: float) -> dict:
    """Final istatistikleri hesapla"""
    try:
        # Temel bilgiler
        initial_capital = portfolio.initial_capital_usdt
        final_value = portfolio.get_total_portfolio_value_usdt(current_price)
        total_profit = final_value - initial_capital
        total_profit_pct = (total_profit / initial_capital) * 100
        
        # Trade istatistikleri
        closed_trades = portfolio.closed_trades
        total_trades = len(closed_trades)
        
        if total_trades == 0:
            return {
                "initial_capital": initial_capital,
                "final_value": final_value,
                "total_profit": total_profit,
                "total_profit_pct": total_profit_pct,
                "total_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "max_win": 0,
                "max_loss": 0,
                "profit_factor": 0
            }
        
        # Karlı ve zararlı işlemler
        winning_trades = [t for t in closed_trades if t["profit_usd"] > 0]
        losing_trades = [t for t in closed_trades if t["profit_usd"] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        
        # Kar/zarar hesaplamaları
        total_wins = sum(t["profit_usd"] for t in winning_trades)
        total_losses = abs(sum(t["profit_usd"] for t in losing_trades))
        
        avg_win = total_wins / win_count if win_count > 0 else 0
        avg_loss = total_losses / loss_count if loss_count > 0 else 0
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        max_win = max((t["profit_usd"] for t in winning_trades), default=0)
        max_loss = min((t["profit_usd"] for t in losing_trades), default=0)
        
        # Profit factor
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Hold time analizi
        hold_times = [t["hold_minutes"] for t in closed_trades if "hold_minutes" in t]
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
        
        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_profit": total_profit,
            "total_profit_pct": total_profit_pct,
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_profit": avg_profit,
            "max_win": max_win,
            "max_loss": max_loss,
            "profit_factor": profit_factor,
            "avg_hold_time": avg_hold_time
        }
        
    except Exception as e:
        logger.error(f"Final statistics calculation error: {e}")
        return {"error": str(e)}

def print_final_summary(stats: dict, current_price: float):
    """Terminal'de detaylı final özet yazdır"""
    logger.info("="*80)
    logger.info("🏆 FINAL TRADING SESSION SUMMARY")
    logger.info("="*80)
    
    if "error" in stats:
        logger.error(f"Statistics calculation failed: {stats['error']}")
        return
    
    # Ana performans
    profit_emoji = "🚀" if stats["total_profit"] > 0 else "📉" if stats["total_profit"] < 0 else "⚖️"
    logger.info(f"💰 PORTFOLIO PERFORMANCE:")
    logger.info(f"   Initial Capital: ${stats['initial_capital']:,.2f} USDT")
    logger.info(f"   Final Value:     ${stats['final_value']:,.2f} USDT")
    logger.info(f"   Total P&L:       ${stats['total_profit']:+,.2f} USDT ({stats['total_profit_pct']:+.2f}%) {profit_emoji}")
    logger.info(f"   Current BTC:     ${current_price:,.2f} USDT")
    
    # Trade istatistikleri
    if stats["total_trades"] > 0:
        logger.info(f"\n📊 TRADING STATISTICS:")
        logger.info(f"   Total Trades:    {stats['total_trades']}")
        logger.info(f"   Winning Trades:  {stats['win_count']} ({stats['win_rate']:.1f}%)")
        logger.info(f"   Losing Trades:   {stats['loss_count']} ({100-stats['win_rate']:.1f}%)")
        logger.info(f"   Profit Factor:   {stats['profit_factor']:.2f}")
        
        logger.info(f"\n💵 PROFIT ANALYSIS:")
        logger.info(f"   Total Wins:      ${stats['total_wins']:+,.2f} USDT")
        logger.info(f"   Total Losses:    ${-stats['total_losses']:+,.2f} USDT")
        logger.info(f"   Average Win:     ${stats['avg_win']:+,.2f} USDT")
        logger.info(f"   Average Loss:    ${-stats['avg_loss']:+,.2f} USDT")
        logger.info(f"   Average Trade:   ${stats['avg_profit']:+,.2f} USDT")
        
        logger.info(f"\n🎯 EXTREMES:")
        logger.info(f"   Best Trade:      ${stats['max_win']:+,.2f} USDT")
        logger.info(f"   Worst Trade:     ${stats['max_loss']:+,.2f} USDT")
        logger.info(f"   Avg Hold Time:   {stats['avg_hold_time']:.1f} minutes")
        
        # Performance rating
        if stats["total_profit_pct"] > 10:
            rating = "EXCELLENT 🌟"
        elif stats["total_profit_pct"] > 5:
            rating = "GREAT 💎"
        elif stats["total_profit_pct"] > 1:
            rating = "GOOD 👍"
        elif stats["total_profit_pct"] > -1:
            rating = "NEUTRAL ⚖️"
        elif stats["total_profit_pct"] > -5:
            rating = "POOR 👎"
        else:
            rating = "CRITICAL 🔴"
            
        logger.info(f"\n🏆 SESSION RATING: {rating}")
        
        # ✅ CSV'YE FINAL SUMMARY EKLE
        try:
            append_summary_to_csv(stats, current_price)
        except Exception as e:
            logger.error(f"Failed to append summary to CSV: {e}")
    else:
        logger.info(f"\n📊 TRADING STATISTICS: No trades executed")
    
    logger.info("="*80)

def append_summary_to_csv(stats: dict, current_price: float):
    """CSV dosyasının sonuna summary ekle"""
    try:
        if not hasattr(settings, 'TRADES_CSV_LOG_PATH') or not settings.TRADES_CSV_LOG_PATH:
            return
            
        from pathlib import Path
        csv_path = Path(settings.TRADES_CSV_LOG_PATH)
        
        if not csv_path.exists():
            return
            
        # Summary separator ve bilgiler
        timestamp = datetime.now(timezone.utc).isoformat()
        
        summary_lines = [
            "\n# ================================ SESSION SUMMARY ================================",
            f"# Session End Time: {timestamp}",
            f"# Initial Capital: ${stats['initial_capital']:,.2f} USDT",
            f"# Final Value: ${stats['final_value']:,.2f} USDT", 
            f"# Total P&L: ${stats['total_profit']:+,.2f} USDT ({stats['total_profit_pct']:+.2f}%)",
            f"# Current BTC Price: ${current_price:,.2f} USDT",
            f"# Total Trades: {stats['total_trades']}",
            f"# Win Rate: {stats['win_rate']:.1f}% ({stats['win_count']} wins, {stats['loss_count']} losses)",
            f"# Profit Factor: {stats['profit_factor']:.2f}",
            f"# Average Trade: ${stats['avg_profit']:+,.2f} USDT",
            f"# Best Trade: ${stats['max_win']:+,.2f} USDT",
            f"# Worst Trade: ${stats['max_loss']:+,.2f} USDT", 
            f"# Average Hold Time: {stats['avg_hold_time']:.1f} minutes",
            "# ================================================================================\n"
        ]
        
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.writelines(line + '\n' for line in summary_lines)
            
        logger.info(f"✅ Session summary appended to {csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to append summary to CSV: {e}")

def print_enhanced_final_summary(portfolio: Portfolio, current_price: float):
    """🎨 CTRL+C ile gelişmiş final özet - Türkçe ve güzel format"""
    summary = portfolio.get_performance_summary(current_price)
    
    if "error" in summary:
        logger.error(f"📊 Performans özeti hatası: {summary['error']}")
        return
    
    # Ekranı temizle ve başlık göster
    print("\n" + "="*100)
    print("🏆 " + " ENHanced TRADING BOT - FINAL ANALİZ ".center(96, "=") + " 🏆")
    print("="*100)
    
    # Performans göstergeleri
    profit = summary.get("total_profit", 0)
    profit_pct = summary.get("total_profit_pct", 0)
    
    if profit > 20:
        status_emoji = "🚀🚀🚀"
        status_text = "MÜKEMMEL PERFORMANS"
        status_color = "🟢"
    elif profit > 10:
        status_emoji = "🚀🚀"
        status_text = "ÇOK İYİ PERFORMANS"
        status_color = "🟢"
    elif profit > 5:
        status_emoji = "🚀"
        status_text = "İYİ PERFORMANS"
        status_color = "🟢"
    elif profit > 0:
        status_emoji = "💰"
        status_text = "POZITIF PERFORMANS"
        status_color = "🟢"
    elif profit > -5:
        status_emoji = "⚖️"
        status_text = "NÖTR PERFORMANS"
        status_color = "🟡"
    elif profit > -15:
        status_emoji = "📉"
        status_text = "ZAYIF PERFORMANS"
        status_color = "🟠"
    else:
        status_emoji = "💥"
        status_text = "KRİTİK PERFORMANS"
        status_color = "🔴"
    
    print(f"\n{status_color} DURUM: {status_text} {status_emoji}")
    print(f"{'='*100}")
    
    # Portfolio Özeti
    print(f"\n💰 PORTFOLIO ÖZETİ:")
    print(f"├─ Başlangıç Sermayesi:   ${summary.get('initial_capital', 0):>10,.2f} USDT")
    print(f"├─ Güncel Değer:          ${summary.get('current_value', 0):>10,.2f} USDT")
    print(f"├─ Toplam Kar/Zarar:      ${profit:>+10,.2f} USDT ({profit_pct:>+6.2f}%)")
    print(f"├─ Kullanılabilir Bakiye: ${summary.get('available_usdt', 0):>10,.2f} USDT")
    print(f"└─ Güncel BTC Fiyatı:     ${current_price:>10,.2f} USDT")
    
    # Trading İstatistikleri
    total_trades = summary.get("total_trades", 0)
    if total_trades > 0:
        win_rate = summary.get("win_rate", 0)
        win_rate_emoji = "🎯" if win_rate >= 70 else "✅" if win_rate >= 50 else "⚠️" if win_rate >= 30 else "❌"
        
        print(f"\n📊 TRADING İSTATİSTİKLERİ:")
        print(f"├─ Toplam İşlem:          {total_trades:>10}")
        print(f"├─ Kazanan İşlem:         {summary.get('win_count', 0):>10} ({win_rate:>5.1f}%) {win_rate_emoji}")
        print(f"├─ Kaybeden İşlem:        {summary.get('loss_count', 0):>10} ({100-win_rate:>5.1f}%)")
        print(f"├─ Profit Factor:         {summary.get('profit_factor', 0):>10.2f}")
        print(f"└─ Ortalama İşlem:        ${summary.get('avg_trade', 0):>+9.2f} USDT")
        
        # Kar/Zarar Analizi
        pf_emoji = "🏆" if summary.get('profit_factor', 0) >= 2 else "💎" if summary.get('profit_factor', 0) >= 1.5 else "⚖️"
        print(f"\n💵 KAR/ZARAR ANALİZİ:")
        print(f"├─ Toplam Kazanç:         ${summary.get('total_wins', 0):>+10.2f} USDT")
        print(f"├─ Toplam Kayıp:          ${-summary.get('total_losses', 0):>+10.2f} USDT")
        print(f"├─ En İyi İşlem:          ${summary.get('max_win', 0):>+10.2f} USDT 💎")
        print(f"├─ En Kötü İşlem:         ${summary.get('max_loss', 0):>+10.2f} USDT 💥")
        print(f"└─ Profit Factor:         {summary.get('profit_factor', 0):>10.2f} {pf_emoji}")
        
        # Strateji Performansı
        strategy_stats = {}
        for trade in portfolio.closed_trades:
            strategy = trade.get("strategy", "Bilinmeyen")
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"trades": 0, "profit": 0}
            strategy_stats[strategy]["trades"] += 1
            strategy_stats[strategy]["profit"] += trade.get("profit_usd", 0)
        
        if strategy_stats:
            print(f"\n🎯 STRATEJİ PERFORMANSI:")
            for strategy, stats in strategy_stats.items():
                avg_profit = stats["profit"] / stats["trades"] if stats["trades"] > 0 else 0
                strategy_emoji = "🚀" if stats["profit"] > 0 else "📉"
                print(f"├─ {strategy:<12}: {stats['trades']:>3} işlem, ${stats['profit']:>+8.2f} (ort: ${avg_profit:>+6.2f}) {strategy_emoji}")
        
        # Son Performans (son 10 işlem)
        if len(portfolio.closed_trades) >= 5:
            recent_trades = portfolio.closed_trades[-10:]
            recent_profit = sum(t.get("profit_usd", 0) for t in recent_trades)
            recent_wins = sum(1 for t in recent_trades if t.get("profit_usd", 0) > 0)
            recent_win_rate = (recent_wins / len(recent_trades)) * 100
            
            print(f"\n📈 SON PERFORMANS (Son {len(recent_trades)} İşlem):")
            print(f"├─ Son Kar/Zarar:         ${recent_profit:>+10.2f} USDT")
            print(f"├─ Son Başarı Oranı:      {recent_win_rate:>10.1f}%")
            print(f"└─ Trend:                 {'🔥 SICAK ÇIZGI' if recent_win_rate >= 70 else '📈 YÜKSELİŞTE' if recent_win_rate >= 50 else '⚠️ ENDİŞELİ' if recent_win_rate >= 30 else '🆘 DİKKAT GEREKİR'}")
    
    # Mevcut Pozisyonlar
    open_positions = summary.get("open_positions", 0)
    if open_positions > 0:
        exposure_pct = summary.get("exposure_pct", 0)
        exposure_emoji = "⚠️" if exposure_pct > 80 else "✅" if exposure_pct > 50 else "🟢"
        print(f"\n🎯 MEVCUT POZİSYONLAR:")
        print(f"├─ Açık Pozisyon:         {open_positions:>10}")
        print(f"├─ Toplam Maruziyet:      ${summary.get('current_exposure', 0):>10,.2f} USDT")
        print(f"└─ Maruziyet Oranı:       {exposure_pct:>10.1f}% {exposure_emoji}")
    
    # Risk Değerlendirmesi
    risk_level = "DÜŞÜK" if abs(profit_pct) < 5 else "ORTA" if abs(profit_pct) < 15 else "YÜKSEK"
    risk_emoji = "🟢" if risk_level == "DÜŞÜK" else "🟡" if risk_level == "ORTA" else "🔴"
    
    print(f"\n🛡️ RİSK DEĞERLENDİRMESİ:")
    print(f"├─ Risk Seviyesi:         {risk_level:>10} {risk_emoji}")
    print(f"├─ Portfolio Sağlığı:     {'MÜKEMMEL' if profit_pct > 5 else 'İYİ' if profit_pct > 0 else 'DİKKAT GEREKİR':>10}")
    print(f"└─ Tavsiye:               {'TICARETİ SÜRDÜR' if profit_pct > -5 else 'STRATEJİYİ GÖZDEN GEÇİR' if profit_pct > -15 else 'DUR VE ANALİZ ET':>10}")
    
    # Oturum Bilgileri
    current_time = datetime.now(timezone.utc)
    print(f"\n⏰ OTURUM BİLGİLERİ:")
    print(f"├─ Bitiş Zamanı:          {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"├─ Sembol:                {getattr(settings, 'SYMBOL', 'BTC/USDT'):>10}")
    print(f"├─ Zaman Dilimi:          {getattr(settings, 'TIMEFRAME', '5m'):>10}")
    print(f"└─ AI Destek:             {'AÇIK' if getattr(settings, 'AI_ASSISTANCE_ENABLED', False) else 'KAPALI':>10}")
    
    # Final mesaj
    print(f"\n{'='*100}")
    if profit > 0:
        print(f"🎉 TEBRİKLER! Bu oturumda ${profit:.2f} USDT ({profit_pct:+.2f}%) kazandınız! 🎉")
    elif profit > -10:
        print(f"💪 Devam edin! Küçük kayıplar ticaretin parçasıdır. ${abs(profit):.2f} USDT ({profit_pct:+.2f}%) kaybınız var")
    else:
        print(f"🚨 Önemli kayıp tespit edildi: ${profit:.2f} USDT ({profit_pct:+.2f}%). Stratejinizi gözden geçirmeyi düşünün.")
    
    print(f"{'='*100}")
    print("📊 Detaylı işlem logları CSV formatında analiz için mevcut.")
    print("🤖 Enhanced AlgoBot kullandığınız için teşekkürler! İyi Ticaretler! 🚀")
    print("="*100 + "\n")

def calculate_final_statistics(portfolio: Portfolio, current_price: float) -> dict:
    """Final istatistikleri hesapla - ENHANCED VERSION"""
    try:
        summary = portfolio.get_performance_summary(current_price)
        if "error" in summary:
            return summary
        
        # Add additional statistics
        closed_trades = portfolio.closed_trades
        
        # Hold time analysis
        if closed_trades:
            hold_times = [t.get("hold_minutes", 0) for t in closed_trades]
            summary["avg_hold_time"] = sum(hold_times) / len(hold_times)
            summary["min_hold_time"] = min(hold_times)
            summary["max_hold_time"] = max(hold_times)
            
            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for trade in closed_trades:
                if trade.get("profit_usd", 0) > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            summary["max_consecutive_wins"] = max_consecutive_wins
            summary["max_consecutive_losses"] = max_consecutive_losses
        
        return summary
        
    except Exception as e:
        logger.error(f"Enhanced final statistics calculation error: {e}")
        return {"error": str(e)}

async def main_loop(
    fetcher: BinanceFetcher, 
    portfolio: Portfolio, 
    strategies: List[Union[MomentumStrategy, BollingerRsiStrategy]],
    risk_manager: RiskManager
):
    """Ana bot döngüsü: Veri çeker, riskleri kontrol eder ve stratejileri işler."""
    logger.info("🤖 Multi-Strategy Trading Bot başlatıldı")
    iteration_count = 0
    last_summary_time = datetime.now(timezone.utc)
    current_price = 50000.0  # Fallback price
    
    try:
        while not shutdown_event.is_set():
            iteration_count += 1
            
            try:
                # 1. Veri Çekme
                ohlcv_df = await fetcher.fetch_ohlcv_data()

                if ohlcv_df is None or ohlcv_df.empty:
                    logger.warning("⚠️ Veri alınamadı - bekleniyor...")
                    await asyncio.sleep(settings.LOOP_SLEEP_SECONDS_ON_DATA_ERROR or settings.LOOP_SLEEP_SECONDS * 2)
                    continue

                current_price = ohlcv_df['close'].iloc[-1]
                current_timestamp = ohlcv_df.index[-1].to_pydatetime()

                # 2. Portfolio durumu
                portfolio_value = portfolio.get_total_portfolio_value_usdt(current_price)
                profit_pct = ((portfolio_value - portfolio.initial_capital_usdt) / portfolio.initial_capital_usdt) * 100

                logger.info(f"Güncel {settings.SYMBOL} fiyatı: {current_price:.8f} USDT (Zaman: {current_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')})")
                logger.info(f"Portföy: {portfolio}, Toplam Değer (yaklaşık): {portfolio_value:.2f} USDT")
                logger.info(f"📈 Güncel Kar/Zarar: {profit_pct:.2f}%")

                # 3. Risk kontrolü
                if not risk_manager.check_global_risk_limits(portfolio, current_btc_price_for_value_calc=current_price):
                    logger.critical("🚨 RİSK LİMİTLERİ AŞILDI! Bot durduruluyor.")
                    shutdown_event.set()
                    break

                # 4. Stratejileri çalıştır
                for strategy in strategies:
                    if shutdown_event.is_set():
                        break
                    try:
                        await strategy.process_data(ohlcv_df.copy()) 
                    except Exception as e:
                        logger.error(f"❌ {strategy.strategy_name} strategy error: {e}")

                # 5. Periyodik performans özetleri (her 10 dakikada bir)
                current_time = datetime.now(timezone.utc)
                if (current_time - last_summary_time).total_seconds() >= 600:  # 10 dakika
                    log_performance_summary(portfolio_value, portfolio.initial_capital_usdt)
                    last_summary_time = current_time

                # 6. Döngü bekleme
                if not shutdown_event.is_set():
                    try:
                        await asyncio.wait_for(shutdown_event.wait(), timeout=settings.LOOP_SLEEP_SECONDS)
                    except asyncio.TimeoutError:
                        pass
            
            except DataFetchingError as e:
                logger.error(f"❌ Veri hatası: {e}")
                await asyncio.sleep(settings.LOOP_SLEEP_SECONDS_ON_DATA_ERROR or settings.LOOP_SLEEP_SECONDS * 3)
            except ccxt.AuthenticationError as e:
                logger.critical(f"🚨 API Authentication Error: {e}")
                shutdown_event.set()
            except Exception as e:
                logger.error(f"❌ Beklenmedik hata: {e}")
                await asyncio.sleep(settings.LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        logger.info("🛑 CTRL+C algılandı - Final analiz hazırlanıyor...")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"❌ Ana döngü kritik hata: {e}")
        shutdown_event.set()
    finally:
        # 🎯 ENHANCED FINAL ANALYSIS!
        logger.info("🏁 Final analiz oluşturuluyor...")
        try:
            print_enhanced_final_summary(portfolio, current_price)
            
            # CSV'ye özet ekle
            final_stats = calculate_final_statistics(portfolio, current_price)
            if "error" not in final_stats:
                append_summary_to_csv(final_stats, current_price)
            
        except Exception as e:
            logger.error(f"Final analiz hatası: {e}")
            # Basit yedek özet
            try:
                simple_stats = portfolio.get_performance_summary(current_price)
                print(f"\n💰 FINAL: ${simple_stats.get('total_profit', 0):+.2f} USDT ({simple_stats.get('total_profit_pct', 0):+.2f}%) | "
                      f"İşlem: {simple_stats.get('total_trades', 0)} | Başarı: {simple_stats.get('win_rate', 0):.1f}%\n")
            except:
                print("\n🏁 Bot oturumu sona erdi\n")

    logger.info("🏁 Bot kapatıldı")

async def run_bot():
    """Botu başlatır, yapılandırır ve ana döngüyü çalıştırır."""
    
    # System startup logging
    config_summary = {
        "Symbol": settings.SYMBOL,
        "Initial Capital": f"${settings.INITIAL_CAPITAL_USDT:,.2f} USDT",
        "Trade Amount": f"${settings.trade_amount_usdt:.2f} USDT",
        "Max Portfolio Drawdown": f"{getattr(settings, 'GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT', 'N/A')}%",
        "AI Assistance": "ACTIVE" if settings.AI_ASSISTANCE_ENABLED else "DISABLED",
        "Timeframe": settings.TIMEFRAME,
        "Loop Sleep": f"{settings.LOOP_SLEEP_SECONDS}s"
    }
    
    log_system_startup(config_summary)

    # CSV log başlığını kontrol et/oluştur (eğer path tanımlıysa)
    if settings.TRADES_CSV_LOG_PATH:
        ensure_csv_header(settings.TRADES_CSV_LOG_PATH)
    if settings.TRADES_JSONL_LOG_PATH: # JSONL için başlık gerekmez ama dizin oluşturulabilir
        Path(settings.TRADES_JSONL_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Bileşenleri oluştur
    fetcher = BinanceFetcher(symbol=settings.SYMBOL, timeframe=settings.TIMEFRAME)
    portfolio = Portfolio(initial_capital_usdt=settings.INITIAL_CAPITAL_USDT)
    risk_manager = RiskManager()

    # 🚀 DUAL STRATEGY SETUP: Momentum + Mean Reversion
    active_strategies: List[Union[MomentumStrategy, BollingerRsiStrategy]] = [
        MomentumStrategy(portfolio=portfolio, symbol=settings.SYMBOL),
        BollingerRsiStrategy(portfolio=portfolio, symbol=settings.SYMBOL),
    ]
    logger.info(f"📊 Active Strategies: {[s.strategy_name for s in active_strategies]}")
    
    # Strategy özellikleri
    for strategy in active_strategies:
        if hasattr(strategy, 'max_positions'):
            logger.info(f"   🎯 {strategy.strategy_name}: Max {strategy.max_positions} positions, "
                       f"${strategy.min_position_usdt}-${strategy.max_position_usdt} size")

    # Kapatma sinyallerini ayarla
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, handle_shutdown_signal)

    current_price = 50000.0  # Fallback price initialization
    
    try:
        # Ana döngüyü çalıştır
        await main_loop(fetcher, portfolio, active_strategies, risk_manager)
    except Exception as e:
        logger.critical(f"main_loop unexpected termination: {e}", exc_info=True)
    finally:
        # Final statistics
        try:
            current_price = 50000.0  # Fallback
            final_stats = calculate_final_statistics(portfolio, current_price)
            
            final_stats_summary = {
                "Total Trades": final_stats.get("total_trades", 0),
                "Final Portfolio Value": f"${final_stats.get('final_value', 0):,.2f}",
                "Total P&L": f"${final_stats.get('total_profit', 0):+,.2f}",
                "Return %": f"{final_stats.get('total_profit_pct', 0):+.2f}%",
                "Win Rate": f"{final_stats.get('win_rate', 0):.1f}%",
                "Profit Factor": f"{final_stats.get('profit_factor', 0):.2f}",
                "Runtime": "Session Completed"
            }
            
            log_system_shutdown(final_stats_summary)
            
        except Exception as e:
            logger.error(f"Final statistics logging failed: {e}")
        
        logger.info("🧹 Cleaning up resources...")
        await fetcher.close_connection()
        logger.info("🏁 Multi-Strategy Bot shutdown complete.")

if __name__ == "__main__":
    # Windows'ta asyncio için event loop policy ayarı (Python 3.8+ için)
    if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt: # Ctrl+C ile doğrudan asyncio.run seviyesinde yakalanırsa
        logger.info("KeyboardInterrupt (Ctrl+C) algılandı. Program sonlandırılıyor.")
    except Exception as e:
        logger.critical(f"Bot çalıştırılırken ölümcül bir hata oluştu: {e}", exc_info=True)
        sys.exit(1) # Hata koduyla çık