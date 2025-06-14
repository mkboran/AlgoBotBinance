# optimize_strategy.py
import optuna
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import os
import argparse


from typing import Dict
from utils.config import settings
from utils.portfolio import Portfolio
from utils.logger import logger, ensure_csv_header
from strategies.momentum_optimized import MomentumStrategy
from backtest_runner import MomentumBacktester

# --- Optimizasyon için Ayarlar - 2025 VERİSİ İÇİN GÜNCELLEME ---
INITIAL_CAPITAL = 1000.0
default_path = Path("historical_data") / "BTCUSDT_15m_20210101_20241231.csv"
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", default_path)
BACKTEST_START_DATE = "2025-01-01"  # 2025 başlangıcı
BACKTEST_END_DATE = "2025-03-31"    # İlk 3 ay test (geri kalan 2 ay validation için)
OPTIMIZATION_METRIC = "total_profit_pct"  # Kar odaklı


print(f"Veri dosyası şu yoldan okunacak: {DATA_FILE_PATH}")

def objective(trial: optuna.Trial) -> float:
    """🚀 SÜPER KAPSAMLI 500 TRIAL OPTIMIZASYON - TÜM PARAMETRELER"""
    try:
        strategy_params = {
            # ================================================================================
            # 📊 TEKNİK İNDİKATÖR PARAMETRELERİ - GENİŞLETİLMİŞ ARALIKLARDA
            # ================================================================================
            "ema_short": trial.suggest_int("ema_short", 5, 25, step=1),
            "ema_medium": trial.suggest_int("ema_medium", 18, 50, step=1),
            "ema_long": trial.suggest_int("ema_long", 35, 100, step=2),
            "rsi_period": trial.suggest_int("rsi_period", 7, 25, step=1),
            "adx_period": trial.suggest_int("adx_period", 7, 30, step=1),
            "atr_period": trial.suggest_int("atr_period", 8, 25, step=1),
            "volume_sma_period": trial.suggest_int("volume_sma_period", 10, 40, step=1),
            
            # ================================================================================
            # 💰 POZİSYON YÖNETİMİ PARAMETRELERİ - AGRESİF ARALIKLARDA
            # ================================================================================
            "max_positions": trial.suggest_int("max_positions", 1, 6, step=1),
            "base_position_size_pct": trial.suggest_float("base_position_size_pct", 8.0, 50.0, step=1.0),
            "min_position_usdt": trial.suggest_float("min_position_usdt", 50.0, 400.0, step=10.0),
            "max_position_usdt": trial.suggest_float("max_position_usdt", 100.0, 600.0, step=10.0),
            
            # ================================================================================
            # 📈 PERFORMANSA DAYALI POZİSYON BOYUTLANDIRMA - TÜM PARAMETRELER
            # ================================================================================
            "size_high_profit_pct": trial.suggest_float("size_high_profit_pct", 10.0, 35.0, step=0.5),
            "size_good_profit_pct": trial.suggest_float("size_good_profit_pct", 8.0, 25.0, step=0.5),
            "size_normal_profit_pct": trial.suggest_float("size_normal_profit_pct", 6.0, 22.0, step=0.5),
            "size_breakeven_pct": trial.suggest_float("size_breakeven_pct", 4.0, 20.0, step=0.5),
            "size_loss_pct": trial.suggest_float("size_loss_pct", 2.0, 15.0, step=0.5),
            "size_max_balance_pct": trial.suggest_float("size_max_balance_pct", 10.0, 35.0, step=0.5),
            
            # ================================================================================
            # 🎯 PERFORMANS EŞİKLERİ - DETAYLI AYARLAMA
            # ================================================================================
            "perf_high_profit_threshold": trial.suggest_float("perf_high_profit_threshold", 0.02, 0.20, step=0.005),
            "perf_good_profit_threshold": trial.suggest_float("perf_good_profit_threshold", 0.01, 0.12, step=0.005),
            "perf_normal_profit_threshold": trial.suggest_float("perf_normal_profit_threshold", -0.01, 0.08, step=0.005),
            "perf_breakeven_threshold": trial.suggest_float("perf_breakeven_threshold", -0.08, 0.02, step=0.005),
            
            # ================================================================================
            # 🛡️ RİSK YÖNETİMİ PARAMETRELERİ - GENİŞ SPEKTRUM
            # ================================================================================
            "max_loss_pct": trial.suggest_float("max_loss_pct", 0.003, 0.025, step=0.0005),
            "min_profit_target_usdt": trial.suggest_float("min_profit_target_usdt", 0.25, 5.0, step=0.1),
            "quick_profit_threshold_usdt": trial.suggest_float("quick_profit_threshold_usdt", 0.2, 3.0, step=0.1),
            "max_hold_minutes": trial.suggest_int("max_hold_minutes", 20, 180, step=5),
            "breakeven_minutes": trial.suggest_int("breakeven_minutes", 1, 15, step=1),
            
            # ================================================================================
            # 🔍 ALIM KOŞULLARI - TEMEL PARAMETRELERİ
            # ================================================================================
            "buy_min_quality_score": trial.suggest_int("buy_min_quality_score", 3, 18, step=1),
            "buy_min_ema_spread_1": trial.suggest_float("buy_min_ema_spread_1", 1e-6, 1e-3, log=True),
            "buy_min_ema_spread_2": trial.suggest_float("buy_min_ema_spread_2", 5e-6, 2e-3, log=True),
            
            # ================================================================================
            # 📊 EMA MOMENTUM PARAMETRELERİ - DETAYLI AYARLAMA
            # ================================================================================
            "buy_ema_mom_excellent": trial.suggest_float("buy_ema_mom_excellent", 0.0002, 0.005, step=0.00005),
            "buy_ema_mom_good": trial.suggest_float("buy_ema_mom_good", 0.0001, 0.002, step=0.00005),
            "buy_ema_mom_decent": trial.suggest_float("buy_ema_mom_decent", 0.00005, 0.001, step=0.00005),
            "buy_ema_mom_min": trial.suggest_float("buy_ema_mom_min", 1e-6, 2e-4, log=True),
            
            # ================================================================================
            # 📈 RSI PARAMETRELERİ - TÜM ARALIKLAR
            # ================================================================================
            "buy_rsi_excellent_min": trial.suggest_float("buy_rsi_excellent_min", 10.0, 40.0, step=1.0),
            "buy_rsi_excellent_max": trial.suggest_float("buy_rsi_excellent_max", 60.0, 90.0, step=1.0),
            "buy_rsi_good_min": trial.suggest_float("buy_rsi_good_min", 2.0, 30.0, step=1.0),
            "buy_rsi_good_max": trial.suggest_float("buy_rsi_good_max", 70.0, 98.0, step=1.0),
            "buy_rsi_extreme_min": trial.suggest_float("buy_rsi_extreme_min", 1.0, 15.0, step=0.5),
            "buy_rsi_extreme_max": trial.suggest_float("buy_rsi_extreme_max", 85.0, 99.0, step=0.5),
            
            # ================================================================================
            # 💪 ADX PARAMETRELERİ - TREND GÜCÜ
            # ================================================================================
            "buy_adx_excellent": trial.suggest_float("buy_adx_excellent", 15.0, 45.0, step=1.0),
            "buy_adx_good": trial.suggest_float("buy_adx_good", 8.0, 30.0, step=0.5),
            "buy_adx_decent": trial.suggest_float("buy_adx_decent", 5.0, 25.0, step=0.5),
            
            # ================================================================================
            # 📊 VOLUME PARAMETRELERİ - İŞLEM HACMİ ANALİZİ
            # ================================================================================
            "buy_volume_excellent": trial.suggest_float("buy_volume_excellent", 1.2, 4.0, step=0.05),
            "buy_volume_good": trial.suggest_float("buy_volume_good", 0.8, 2.5, step=0.05),
            "buy_volume_decent": trial.suggest_float("buy_volume_decent", 0.3, 2.0, step=0.05),
            
            # ================================================================================
            # 🚀 FİYAT MOMENTUM PARAMETRELERİ
            # ================================================================================
            "buy_price_mom_excellent": trial.suggest_float("buy_price_mom_excellent", 0.0001, 0.002, step=0.00005),
            "buy_price_mom_good": trial.suggest_float("buy_price_mom_good", 0.00005, 0.001, step=0.00005),
            "buy_price_mom_decent": trial.suggest_float("buy_price_mom_decent", -0.002, 0.0005, step=0.00005),
            
            # ================================================================================
            # 💸 SATIŞ KOŞULLARI - TEMEL PARAMETRELERİ
            # ================================================================================
            "sell_min_hold_minutes": trial.suggest_int("sell_min_hold_minutes", 5, 60, step=2),
            "sell_catastrophic_loss_pct": trial.suggest_float("sell_catastrophic_loss_pct", -0.15, -0.01, step=0.002),
            
            # ================================================================================
            # 💎 PREMİUM KAR SEVİYELERİ
            # ================================================================================
            "sell_premium_excellent": trial.suggest_float("sell_premium_excellent", 2.0, 12.0, step=0.2),
            "sell_premium_great": trial.suggest_float("sell_premium_great", 1.5, 8.0, step=0.1),
            "sell_premium_good": trial.suggest_float("sell_premium_good", 1.0, 5.0, step=0.1),
            
            # ================================================================================
            # ⏰ FAZ 1 PARAMETRELERİ (İlk 60 dakika)
            # ================================================================================
            "sell_phase1_excellent": trial.suggest_float("sell_phase1_excellent", 0.2, 2.5, step=0.05),
            "sell_phase1_good": trial.suggest_float("sell_phase1_good", 0.1, 1.8, step=0.05),
            "sell_phase1_loss_protection": trial.suggest_float("sell_phase1_loss_protection", -5.0, -0.5, step=0.1),
            
            # ================================================================================
            # ⏰ FAZ 2 PARAMETRELERİ (60-120 dakika)
            # ================================================================================
            "sell_phase2_excellent": trial.suggest_float("sell_phase2_excellent", 0.3, 2.5, step=0.05),
            "sell_phase2_good": trial.suggest_float("sell_phase2_good", 0.2, 1.8, step=0.05),
            "sell_phase2_decent": trial.suggest_float("sell_phase2_decent", 0.1, 1.2, step=0.05),
            "sell_phase2_loss_protection": trial.suggest_float("sell_phase2_loss_protection", -4.0, -0.3, step=0.1),
            
            # ================================================================================
            # ⏰ FAZ 3 PARAMETRELERİ (120-180 dakika)
            # ================================================================================
            "sell_phase3_excellent": trial.suggest_float("sell_phase3_excellent", 0.2, 2.0, step=0.05),
            "sell_phase3_good": trial.suggest_float("sell_phase3_good", 0.1, 1.5, step=0.05),
            "sell_phase3_decent": trial.suggest_float("sell_phase3_decent", 0.05, 1.0, step=0.025),
            "sell_phase3_breakeven_min": trial.suggest_float("sell_phase3_breakeven_min", -0.6, -0.05, step=0.02),
            "sell_phase3_breakeven_max": trial.suggest_float("sell_phase3_breakeven_max", 0.05, 0.6, step=0.02),
            "sell_phase3_loss_protection": trial.suggest_float("sell_phase3_loss_protection", -3.0, -0.2, step=0.1),
            
            # ================================================================================
            # ⏰ FAZ 4 PARAMETRELERİ (180+ dakika)
            # ================================================================================
            "sell_phase4_excellent": trial.suggest_float("sell_phase4_excellent", 0.1, 1.2, step=0.025),
            "sell_phase4_good": trial.suggest_float("sell_phase4_good", 0.05, 0.8, step=0.025),
            "sell_phase4_minimal": trial.suggest_float("sell_phase4_minimal", 0.02, 0.5, step=0.02),
            "sell_phase4_breakeven_min": trial.suggest_float("sell_phase4_breakeven_min", -0.8, -0.1, step=0.02),
            "sell_phase4_breakeven_max": trial.suggest_float("sell_phase4_breakeven_max", 0.1, 0.8, step=0.02),
            "sell_phase4_force_exit_minutes": trial.suggest_int("sell_phase4_force_exit_minutes", 90, 720, step=30),
            
            # ================================================================================
            # ⚠️ RİSK VE TEKNİK SATIŞ PARAMETRELERİ
            # ================================================================================
            "sell_loss_multiplier": trial.suggest_float("sell_loss_multiplier", 1.5, 8.0, step=0.2),
            "sell_tech_min_minutes": trial.suggest_int("sell_tech_min_minutes", 30, 200, step=5),
            "sell_tech_min_loss": trial.suggest_float("sell_tech_min_loss", -5.0, -0.2, step=0.1),
            "sell_tech_rsi_extreme": trial.suggest_float("sell_tech_rsi_extreme", 2.0, 25.0, step=0.5),
            
            # ================================================================================
            # ⏳ BEKLEME SÜRELERİ PARAMETRELERİ
            # ================================================================================
            "wait_profit_5pct": trial.suggest_int("wait_profit_5pct", 120, 900, step=15),
            "wait_profit_2pct": trial.suggest_int("wait_profit_2pct", 200, 1200, step=30),
            "wait_breakeven": trial.suggest_int("wait_breakeven", 300, 1800, step=30),
            "wait_loss": trial.suggest_int("wait_loss", 400, 2400, step=60),
            
            # ================================================================================
            # 🤖 AI PARAMETRELERİ - TEMEL AYARLAR
            # ================================================================================
            "ai_confidence_threshold": trial.suggest_float("ai_confidence_threshold", 0.05, 0.8, step=0.02),
            "ai_momentum_confidence_override": trial.suggest_float("ai_momentum_confidence_override", 0.1, 0.6, step=0.02),
            
            # ================================================================================
            # 🧠 AI TEKNİK ANALİZ AĞIRLIKLARI
            # ================================================================================
            "ai_weight_trend_main": trial.suggest_float("ai_weight_trend_main", 0.1, 0.8, step=0.02),
            "ai_weight_trend_long": trial.suggest_float("ai_weight_trend_long", 0.05, 0.6, step=0.02),
            "ai_weight_volume": trial.suggest_float("ai_weight_volume", 0.05, 0.5, step=0.02),
            "ai_weight_divergence": trial.suggest_float("ai_weight_divergence", 0.02, 0.3, step=0.01),
            
            # ================================================================================
            # 🎯 AI EŞİKLERİ - STANDALONE SINYALLER
            # ================================================================================
            "ai_standalone_thresh_strong_buy": trial.suggest_float("ai_standalone_thresh_strong_buy", 0.4, 0.95, step=0.02),
            "ai_standalone_thresh_buy": trial.suggest_float("ai_standalone_thresh_buy", 0.1, 0.7, step=0.02),
            "ai_standalone_thresh_sell": trial.suggest_float("ai_standalone_thresh_sell", -0.7, -0.1, step=0.02),
            "ai_standalone_thresh_strong_sell": trial.suggest_float("ai_standalone_thresh_strong_sell", -0.95, -0.4, step=0.02),
            
            # ================================================================================
            # ✅ AI ONAY PARAMETRELERİ - DETAYLI KONTROLLER
            # ================================================================================
            "ai_confirm_min_ta_score": trial.suggest_float("ai_confirm_min_ta_score", 0.05, 0.7, step=0.02),
            "ai_confirm_min_quality_score": trial.suggest_int("ai_confirm_min_quality_score", 1, 12, step=1),
            "ai_confirm_min_ema_spread_1": trial.suggest_float("ai_confirm_min_ema_spread_1", 0.00005, 0.002, step=0.00005),
            "ai_confirm_min_ema_spread_2": trial.suggest_float("ai_confirm_min_ema_spread_2", 0.0001, 0.003, step=0.00005),
            "ai_confirm_min_volume_ratio": trial.suggest_float("ai_confirm_min_volume_ratio", 0.3, 3.0, step=0.05),
            "ai_confirm_min_price_momentum": trial.suggest_float("ai_confirm_min_price_momentum", 0.00005, 0.002, step=0.00005),
            "ai_confirm_min_ema_momentum": trial.suggest_float("ai_confirm_min_ema_momentum", 0.0001, 0.003, step=0.00005),
            "ai_confirm_min_adx": trial.suggest_float("ai_confirm_min_adx", 3.0, 30.0, step=0.5),
            
            # ================================================================================
            # 📊 AI ZARAR/KAR DURUMLARINDA TA SCORE EŞİKLERİ
            # ================================================================================
            "ai_confirm_loss_5pct_ta_score": trial.suggest_float("ai_confirm_loss_5pct_ta_score", 0.1, 0.8, step=0.02),
            "ai_confirm_loss_2pct_ta_score": trial.suggest_float("ai_confirm_loss_2pct_ta_score", 0.08, 0.7, step=0.02),
            "ai_confirm_profit_ta_score": trial.suggest_float("ai_confirm_profit_ta_score", 0.05, 0.6, step=0.02),
            
            # ================================================================================
            # ⚠️ AI RİSK DEĞERLENDİRME PARAMETRELERİ
            # ================================================================================
            "ai_risk_volatility_threshold": trial.suggest_float("ai_risk_volatility_threshold", 0.005, 0.08, step=0.002),
            "ai_risk_volume_spike_threshold": trial.suggest_float("ai_risk_volume_spike_threshold", 1.2, 5.0, step=0.05),
            
            # ================================================================================
            # 🌍 GLOBAL RİSK YÖNETİMİ PARAMETRELERİ
            # ================================================================================
            "global_max_position_size_pct": trial.suggest_float("global_max_position_size_pct", 10.0, 40.0, step=1.0),
            "global_max_open_positions": trial.suggest_int("global_max_open_positions", 3, 15, step=1),
            "global_max_portfolio_drawdown_pct": trial.suggest_float("global_max_portfolio_drawdown_pct", 0.05, 0.35, step=0.01),
            "global_max_daily_loss_pct": trial.suggest_float("global_max_daily_loss_pct", 0.01, 0.12, step=0.005),
            
            # ================================================================================
            # ⚙️ SİSTEM PARAMETRELERİ - ZAMAN VE MİKTAR KONTROLLERI
            # ================================================================================
            "min_time_between_trades_sec": trial.suggest_int("min_time_between_trades_sec", 5, 180, step=5),
            "min_trade_amount_usdt": trial.suggest_float("min_trade_amount_usdt", 10.0, 150.0, step=5.0),
        }

        # ================================================================================
        # 🔍 VALIDATION KONTROLLARI - GELİŞTİRİLMİŞ MANTIK KONTROLLERI
        # ================================================================================
        
        # EMA periyotlarının sıralı olduğundan emin ol
        if not (strategy_params["ema_short"] < strategy_params["ema_medium"] < strategy_params["ema_long"]):
            raise optuna.exceptions.TrialPruned("Invalid EMA period order.")
        
        # RSI aralıklarının mantıklı olduğundan emin ol
        if strategy_params["buy_rsi_excellent_min"] >= strategy_params["buy_rsi_excellent_max"]:
            raise optuna.exceptions.TrialPruned("Invalid RSI excellent range.")
        
        if strategy_params["buy_rsi_good_min"] >= strategy_params["buy_rsi_good_max"]:
            raise optuna.exceptions.TrialPruned("Invalid RSI good range.")
        
        # Pozisyon boyutu aralıklarının mantıklı olduğundan emin ol
        if strategy_params["min_position_usdt"] >= strategy_params["max_position_usdt"]:
            raise optuna.exceptions.TrialPruned("Invalid position size range.")
        
        # Faz satış parametrelerinin mantıklı olduğundan emin ol
        if strategy_params["sell_phase3_breakeven_min"] >= strategy_params["sell_phase3_breakeven_max"]:
            raise optuna.exceptions.TrialPruned("Invalid phase3 breakeven range.")
        
        if strategy_params["sell_phase4_breakeven_min"] >= strategy_params["sell_phase4_breakeven_max"]:
            raise optuna.exceptions.TrialPruned("Invalid phase4 breakeven range.")
        
        # AI threshold'larının mantıklı olduğundan emin ol
        if strategy_params["ai_standalone_thresh_sell"] >= strategy_params["ai_standalone_thresh_buy"]:
            raise optuna.exceptions.TrialPruned("Invalid AI standalone thresholds.")
        
        # ================================================================================
        # 🚀 BACKTEST ÇALIŞTIRMA VE RİSK-ODAKLI SONUÇ DEĞERLENDİRME
        # ================================================================================
        
        backtest_results = asyncio.run(run_single_backtest_async(strategy_params))
        
        # Adım 1'de hesapladığımız yeni metrikleri sonuçlardan alalım
        profit_pct = backtest_results.get('total_profit_pct', -100.0)
        num_trades = backtest_results.get('total_trades', 0)
        max_drawdown = backtest_results.get('max_drawdown_pct', 100.0)
        sortino_ratio = backtest_results.get('sortino_ratio', -10.0)

        # Anlamlı bir sonuç için minimum işlem sayısı kontrolü yapalım
        if num_trades < 30:
            return -1000.0  # Çok az işlem varsa bu deneme kötü bir puandır, ele.

        # Yüksek sermaye düşüşünü (drawdown) ağır şekilde cezalandıralım
        if max_drawdown > 25.0: # %25'ten fazla düşüşü kabul etme, ele.
            return -500.0

        # Optimizasyon hedefimiz artık Sortino Oranı!
        # Sortino Oranı, riske göre ayarlanmış getiriyi ölçtüğü için çok daha güvenilir bir metriktir.
        metric_value = sortino_ratio
        
        # Eğer Sortino oranı çok benzerse, daha az riskli (düşük drawdown) olanı tercih et
        # Bu, optimizasyona ince bir ayar katmanı ekler
        if metric_value > 0:
            metric_value = metric_value * (1 - (max_drawdown / 100))
        
        # Loglama mesajını yeni ve anlamlı metrikleri içerecek şekilde güncelleyelim
        logger.info(
            f"TRIAL {trial.number} COMPLETED. "
            f"Metric (Risk-Adjusted Score): {metric_value:.3f}, "
            f"Sortino: {sortino_ratio:.2f}, "
            f"Profit: {profit_pct:.2f}%, Drawdown: {max_drawdown:.2f}%, Trades: {num_trades}"
        )
        return metric_value
    
    except optuna.exceptions.TrialPruned as e:
        logger.debug(f"TRIAL {trial.number} PRUNED: {e}")
        raise 
    except Exception as e:
        logger.error(f"TRIAL {trial.number} FAILED with unexpected error: {e}", exc_info=True)
        return -100.0  # Başarısız denemeler için çok kötü bir skor

async def run_single_backtest_async(strategy_params: Dict) -> Dict:
    """Tek bir backtest denemesini asenkron olarak çalıştıran ve sonuçları döndüren yardımcı fonksiyon."""
    portfolio = Portfolio(initial_capital_usdt=INITIAL_CAPITAL)
    
    # Stratejiyi Optuna'nın önerdiği parametrelerle başlat
    strategy = MomentumStrategy(portfolio=portfolio, **strategy_params)
    
    backtester = MomentumBacktester(
        csv_path=str(DATA_FILE_PATH), 
        initial_capital=INITIAL_CAPITAL,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        symbol=settings.SYMBOL,
        portfolio_instance=portfolio,
        strategy_instance=strategy
    )
    return await backtester.run_backtest()

async def run_final_backtest_with_best_params(best_params: Dict, study_name: str):
    """En iyi parametrelerle son bir backtest çalıştırır ve temiz bir CSV'ye loglar."""
    logger.info("\n" + "="*80)
    logger.info(f"🚀 RUNNING FINAL BACKTEST with best parameters for study: {study_name}")
    logger.info(f"Optimal Parameters: {json.dumps(best_params, indent=2)}")
    
    original_csv_logging_status = settings.ENABLE_CSV_LOGGING
    original_trades_csv_path = settings.TRADES_CSV_LOG_PATH
    
    # Final backtest için CSV loglamasını aç ve özel bir dosya kullan
    settings.ENABLE_CSV_LOGGING = True
    final_csv_path_str = str(Path("logs") / f"trades_final_optimized_{study_name}.csv")
    ensure_csv_header(final_csv_path_str) 
    settings.TRADES_CSV_LOG_PATH = final_csv_path_str
    
    logger.info(f"Final backtest trade results will be logged to: {final_csv_path_str}")
        
    try:
        results = await run_single_backtest_async(best_params)
        
        logger.info("--- FINAL BACKTEST RESULTS (with best_params) ---")
        for key, value in results.items():
            if isinstance(value, float):
                if "pct" in key.lower() or "rate" in key.lower() or "factor" in key.lower():
                    logger.info(f"  {key.replace('_', ' ').title():<25}: {value:,.2f}%")
                else:
                    logger.info(f"  {key.replace('_', ' ').title():<25}: ${value:,.2f}")
            else:
                logger.info(f"  {key.replace('_', ' ').title():<25}: {value}")
        logger.info("-" * 60)
    finally:
        # Orijinal ayarları geri yükle
        settings.TRADES_CSV_LOG_PATH = original_trades_csv_path
        settings.ENABLE_CSV_LOGGING = original_csv_logging_status

def run_optimization(start_date: str, end_date: str):
    """Ana optimizasyon sürecini yönetir - Dışarıdan tarih alabilen ve buluta hazır versiyon."""
    study_name = f"momentum_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # Bulut için PostgreSQL veya yerel için SQLite veritabanı yolunu ortam değişkeninden al
    storage_path = os.getenv("OPTUNA_DB_URL")
    if not storage_path:
        print("OPTUNA_DB_URL bulunamadı, yerel SQLite veritabanı kullanılacak.")
        storage_path = f"sqlite:///logs/{study_name}.db"
    else:
        print(f"Bulut veritabanı kullanılacak.")
    
    logger.info(f"Optuna study '{study_name}' will be stored in: {storage_path}")
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Walk-Forward için backtest tarihlerini dışarıdan gelen parametrelere ayarla
    global BACKTEST_START_DATE, BACKTEST_END_DATE
    BACKTEST_START_DATE = start_date
    BACKTEST_END_DATE = end_date

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_path,
        load_if_exists=True
    )
    
    # Hedeflenen deneme sayısını 2000 olarak ayarladık
    n_trials = 2000
    # Timeout'u çok uzun bir süreye ayarlayalım ki sorun çıkarmasın
    timeout_seconds = 30 * 24 * 3600 # 30 gün

    logger.info(f"🚀 Starting {n_trials}-TRIAL optimization for study: '{study_name}'")
    logger.info(f"   Backtest Period: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
    
    original_csv_logging_status = settings.ENABLE_CSV_LOGGING
    settings.ENABLE_CSV_LOGGING = False
    
    try:
        # n_jobs=1, çünkü her sanal makine zaten tek bir işlem yürütecek.
        # Paralelliği makine sayısıyla sağlıyoruz.
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, n_jobs=1, catch=(Exception,))
    except KeyboardInterrupt:
        logger.info("🛑 Optimization process interrupted by user.")
    finally:
        settings.ENABLE_CSV_LOGGING = original_csv_logging_status
        logger.info(f"\n🏁 Optimization for period {start_date}-{end_date} Finished or Interrupted!")
        
        try:
            best_trial = study.best_trial
            logger.info("🏆 Best trial found for this period:")
            logger.info(f"  Value: {best_trial.value:.4f}")
            # Parametreleri loglamak çok uzun sürebilir, bu yüzden özet bilgi daha iyi olabilir
            # logger.info(f"  Params: \n{json.dumps(best_trial.params, indent=4)}")
        except ValueError:
            logger.warning("⚠️ No best trial found for this period. All trials may have failed or were pruned.")
        
        # Bu kısımda final backtest çalıştırmıyoruz, çünkü bu sadece bir optimizasyon penceresi.
        # logger.info(f"📊 To visualize results, run: optuna-dashboard {storage_path}")

if __name__ == "__main__":
    # Komut satırından --start ve --end argümanlarını almak için parser oluştur
    parser = argparse.ArgumentParser(description="Run a Walk-Forward optimization window.")
    parser.add_argument("--start", required=True, help="Backtest start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, help="Backtest end date in YYYY-MM-DD format")
    args = parser.parse_args()

    # Ortam değişkeninden veya varsayılan yoldan veri dosyasını bul
    default_path = Path("historical_data") / "BTCUSDT_15m_20210101_20241231.csv"
    DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", default_path)
    print(f"Veri dosyası şu yoldan okunacak: {DATA_FILE_PATH}")

    if not Path(DATA_FILE_PATH).exists():
        logger.error(f"ERROR: Data file not found at {DATA_FILE_PATH}")
    else:
        logger.info(f"Using data file: {DATA_FILE_PATH} for optimization.")
        # run_optimization fonksiyonunu, alınan tarih argümanlarıyla çalıştır
        run_optimization(start_date=args.start, end_date=args.end)