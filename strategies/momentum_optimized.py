# strategies/momentum_optimized.py - Enhanced Standalone Momentum Strategy

import pandas as pd
import pandas_ta as ta
from typing import Optional, Dict, Tuple, Any
from datetime import datetime, timezone
import asyncio

from utils.portfolio import Portfolio, Position
from utils.config import settings
from utils.logger import logger
from utils.ai_signal_provider import AiSignalProvider # AiSignalProvider import edildi

class MomentumStrategy:
    """🚀 Advanced Standalone Momentum Strategy with AI Enhancement"""
    
    def __init__(
        self, 
        portfolio: Portfolio, 
        symbol: str = "BTC/USDT",
        # --- Teknik İndikatör Parametreleri ---
        ema_short: Optional[int] = None,
        ema_medium: Optional[int] = None,
        ema_long: Optional[int] = None,
        rsi_period: Optional[int] = None,
        adx_period: Optional[int] = None,
        atr_period: Optional[int] = None,
        volume_sma_period: Optional[int] = None,
        
        # --- Pozisyon Yönetimi Parametreleri ---
        max_positions: Optional[int] = None,
        base_position_size_pct: Optional[float] = None,
        min_position_usdt: Optional[float] = None,
        max_position_usdt: Optional[float] = None,
        
        # --- Performansa Dayalı Pozisyon Boyutlandırma ---
        size_high_profit_pct: Optional[float] = None,
        size_good_profit_pct: Optional[float] = None,
        size_normal_profit_pct: Optional[float] = None,
        size_breakeven_pct: Optional[float] = None,
        size_loss_pct: Optional[float] = None,
        size_max_balance_pct: Optional[float] = None,
        
        # --- Performans Eşikleri ---
        perf_high_profit_threshold: Optional[float] = None,
        perf_good_profit_threshold: Optional[float] = None,
        perf_normal_profit_threshold: Optional[float] = None,
        perf_breakeven_threshold: Optional[float] = None,
        
        # --- Risk Yönetimi Parametreleri ---
        max_loss_pct: Optional[float] = None,
        min_profit_target_usdt: Optional[float] = None,
        quick_profit_threshold_usdt: Optional[float] = None,
        max_hold_minutes: Optional[int] = None,
        breakeven_minutes: Optional[int] = None,
        
        # --- Alım Koşulları - Temel ---
        buy_min_quality_score: Optional[int] = None,
        buy_min_ema_spread_1: Optional[float] = None,
        buy_min_ema_spread_2: Optional[float] = None,
        
        # --- EMA Momentum Parametreleri ---
        buy_ema_mom_excellent: Optional[float] = None,
        buy_ema_mom_good: Optional[float] = None,
        buy_ema_mom_decent: Optional[float] = None,
        buy_ema_mom_min: Optional[float] = None,
        
        # --- RSI Parametreleri ---
        buy_rsi_excellent_min: Optional[float] = None,
        buy_rsi_excellent_max: Optional[float] = None,
        buy_rsi_good_min: Optional[float] = None,
        buy_rsi_good_max: Optional[float] = None,
        buy_rsi_extreme_min: Optional[float] = None,
        buy_rsi_extreme_max: Optional[float] = None,
        
        # --- ADX Parametreleri ---
        buy_adx_excellent: Optional[float] = None,
        buy_adx_good: Optional[float] = None,
        buy_adx_decent: Optional[float] = None,
        
        # --- Volume Parametreleri ---
        buy_volume_excellent: Optional[float] = None,
        buy_volume_good: Optional[float] = None,
        buy_volume_decent: Optional[float] = None,
        
        # --- Price Momentum Parametreleri ---
        buy_price_mom_excellent: Optional[float] = None,
        buy_price_mom_good: Optional[float] = None,
        buy_price_mom_decent: Optional[float] = None,
        
        # --- Satış Koşulları - Temel ---
        sell_min_hold_minutes: Optional[int] = None,
        sell_catastrophic_loss_pct: Optional[float] = None,
        
        # --- Premium Kar Seviyeleri ---
        sell_premium_excellent: Optional[float] = None,
        sell_premium_great: Optional[float] = None,
        sell_premium_good: Optional[float] = None,
        
        # --- Faz 1 Parametreleri ---
        sell_phase1_excellent: Optional[float] = None,
        sell_phase1_good: Optional[float] = None,
        sell_phase1_loss_protection: Optional[float] = None,
        
        # --- Faz 2 Parametreleri ---
        sell_phase2_excellent: Optional[float] = None,
        sell_phase2_good: Optional[float] = None,
        sell_phase2_decent: Optional[float] = None,
        sell_phase2_loss_protection: Optional[float] = None,
        
        # --- Faz 3 Parametreleri ---
        sell_phase3_excellent: Optional[float] = None,
        sell_phase3_good: Optional[float] = None,
        sell_phase3_decent: Optional[float] = None,
        sell_phase3_breakeven_min: Optional[float] = None,
        sell_phase3_breakeven_max: Optional[float] = None,
        sell_phase3_loss_protection: Optional[float] = None,
        
        # --- Faz 4 Parametreleri ---
        sell_phase4_excellent: Optional[float] = None,
        sell_phase4_good: Optional[float] = None,
        sell_phase4_minimal: Optional[float] = None,
        sell_phase4_breakeven_min: Optional[float] = None,
        sell_phase4_breakeven_max: Optional[float] = None,
        sell_phase4_force_exit_minutes: Optional[int] = None,
        
        # --- Risk ve Teknik Satış Parametreleri ---
        sell_loss_multiplier: Optional[float] = None,
        sell_tech_min_minutes: Optional[int] = None,
        sell_tech_min_loss: Optional[float] = None,
        sell_tech_rsi_extreme: Optional[float] = None,
        
        # --- Bekleme Süreleri ---
        wait_profit_5pct: Optional[int] = None,
        wait_profit_2pct: Optional[int] = None,
        wait_breakeven: Optional[int] = None,
        wait_loss: Optional[int] = None,
        
        # --- AI Parametreleri ---
        ai_confidence_threshold: Optional[float] = None,
        ai_momentum_confidence_override: Optional[float] = None,
        
        # --- AI Teknik Analiz Ağırlıkları ---
        ai_weight_trend_main: Optional[float] = None,
        ai_weight_trend_long: Optional[float] = None,
        ai_weight_volume: Optional[float] = None,
        ai_weight_divergence: Optional[float] = None,
        
        # --- AI Eşikleri ---
        ai_standalone_thresh_strong_buy: Optional[float] = None,
        ai_standalone_thresh_buy: Optional[float] = None,
        ai_standalone_thresh_sell: Optional[float] = None,
        ai_standalone_thresh_strong_sell: Optional[float] = None,
        
        # --- AI Onay Parametreleri ---
        ai_confirm_min_ta_score: Optional[float] = None,
        ai_confirm_min_quality_score: Optional[int] = None,
        ai_confirm_min_ema_spread_1: Optional[float] = None,
        ai_confirm_min_ema_spread_2: Optional[float] = None,
        ai_confirm_min_volume_ratio: Optional[float] = None,
        ai_confirm_min_price_momentum: Optional[float] = None,
        ai_confirm_min_ema_momentum: Optional[float] = None,
        ai_confirm_min_adx: Optional[float] = None,
        
        # --- AI Zarar/Kar Durumlarında TA Score Eşikleri ---
        ai_confirm_loss_5pct_ta_score: Optional[float] = None,
        ai_confirm_loss_2pct_ta_score: Optional[float] = None,
        ai_confirm_profit_ta_score: Optional[float] = None,
        
        # --- AI Risk Değerlendirme ---
        ai_risk_volatility_threshold: Optional[float] = None,
        ai_risk_volume_spike_threshold: Optional[float] = None,
        
        # --- Global Risk Parametreleri ---
        global_max_position_size_pct: Optional[float] = None,
        global_max_open_positions: Optional[int] = None,
        global_max_portfolio_drawdown_pct: Optional[float] = None,
        global_max_daily_loss_pct: Optional[float] = None,
        
        # --- Sistem Parametreleri ---
        min_time_between_trades_sec: Optional[int] = None,
        min_trade_amount_usdt: Optional[float] = None,
        
        # --- AI Provider ---
        ai_provider_instance: Optional[AiSignalProvider] = None
    ):
        self.strategy_name = "Momentum"
        self.portfolio = portfolio
        self.symbol = symbol if symbol else settings.SYMBOL
        
        # Teknik İndikatör Parametreleri
        self.ema_short = ema_short if ema_short is not None else settings.MOMENTUM_EMA_SHORT
        self.ema_medium = ema_medium if ema_medium is not None else settings.MOMENTUM_EMA_MEDIUM
        self.ema_long = ema_long if ema_long is not None else settings.MOMENTUM_EMA_LONG
        self.rsi_period = rsi_period if rsi_period is not None else settings.MOMENTUM_RSI_PERIOD
        self.adx_period = adx_period if adx_period is not None else settings.MOMENTUM_ADX_PERIOD
        self.atr_period = atr_period if atr_period is not None else settings.MOMENTUM_ATR_PERIOD
        self.volume_sma_period = volume_sma_period if volume_sma_period is not None else settings.MOMENTUM_VOLUME_SMA_PERIOD
        
        # Pozisyon Yönetimi
        self.max_positions = max_positions if max_positions is not None else settings.MOMENTUM_MAX_POSITIONS
        self.base_position_pct = base_position_size_pct if base_position_size_pct is not None else settings.MOMENTUM_BASE_POSITION_SIZE_PCT
        self.min_position_usdt = min_position_usdt if min_position_usdt is not None else settings.MOMENTUM_MIN_POSITION_USDT
        self.max_position_usdt = max_position_usdt if max_position_usdt is not None else settings.MOMENTUM_MAX_POSITION_USDT
        
        # Performansa Dayalı Pozisyon Boyutlandırma
        self.size_high_profit_pct = size_high_profit_pct if size_high_profit_pct is not None else settings.MOMENTUM_SIZE_HIGH_PROFIT_PCT
        self.size_good_profit_pct = size_good_profit_pct if size_good_profit_pct is not None else settings.MOMENTUM_SIZE_GOOD_PROFIT_PCT
        self.size_normal_profit_pct = size_normal_profit_pct if size_normal_profit_pct is not None else settings.MOMENTUM_SIZE_NORMAL_PROFIT_PCT
        self.size_breakeven_pct = size_breakeven_pct if size_breakeven_pct is not None else settings.MOMENTUM_SIZE_BREAKEVEN_PCT
        self.size_loss_pct = size_loss_pct if size_loss_pct is not None else settings.MOMENTUM_SIZE_LOSS_PCT
        self.size_max_balance_pct = size_max_balance_pct if size_max_balance_pct is not None else settings.MOMENTUM_SIZE_MAX_BALANCE_PCT
        
        # Performans Eşikleri
        self.perf_high_profit_threshold = perf_high_profit_threshold if perf_high_profit_threshold is not None else settings.MOMENTUM_PERF_HIGH_PROFIT_THRESHOLD
        self.perf_good_profit_threshold = perf_good_profit_threshold if perf_good_profit_threshold is not None else settings.MOMENTUM_PERF_GOOD_PROFIT_THRESHOLD
        self.perf_normal_profit_threshold = perf_normal_profit_threshold if perf_normal_profit_threshold is not None else settings.MOMENTUM_PERF_NORMAL_PROFIT_THRESHOLD
        self.perf_breakeven_threshold = perf_breakeven_threshold if perf_breakeven_threshold is not None else settings.MOMENTUM_PERF_BREAKEVEN_THRESHOLD
        
        # Risk Yönetimi
        self.max_loss_pct = max_loss_pct if max_loss_pct is not None else settings.MOMENTUM_MAX_LOSS_PCT
        self.min_profit_target_usdt = min_profit_target_usdt if min_profit_target_usdt is not None else settings.MOMENTUM_MIN_PROFIT_TARGET_USDT
        self.quick_profit_threshold_usdt = quick_profit_threshold_usdt if quick_profit_threshold_usdt is not None else settings.MOMENTUM_QUICK_PROFIT_THRESHOLD_USDT
        self.max_hold_minutes = max_hold_minutes if max_hold_minutes is not None else settings.MOMENTUM_MAX_HOLD_MINUTES
        self.breakeven_minutes = breakeven_minutes if breakeven_minutes is not None else settings.MOMENTUM_BREAKEVEN_MINUTES
        
        # Alım Koşulları
        self.buy_min_quality_score = buy_min_quality_score if buy_min_quality_score is not None else settings.MOMENTUM_BUY_MIN_QUALITY_SCORE
        self.buy_min_ema_spread_1 = buy_min_ema_spread_1 if buy_min_ema_spread_1 is not None else settings.MOMENTUM_BUY_MIN_EMA_SPREAD_1
        self.buy_min_ema_spread_2 = buy_min_ema_spread_2 if buy_min_ema_spread_2 is not None else settings.MOMENTUM_BUY_MIN_EMA_SPREAD_2
        
        # EMA Momentum
        self.buy_ema_mom_excellent = buy_ema_mom_excellent if buy_ema_mom_excellent is not None else settings.MOMENTUM_BUY_EMA_MOM_EXCELLENT
        self.buy_ema_mom_good = buy_ema_mom_good if buy_ema_mom_good is not None else settings.MOMENTUM_BUY_EMA_MOM_GOOD
        self.buy_ema_mom_decent = buy_ema_mom_decent if buy_ema_mom_decent is not None else settings.MOMENTUM_BUY_EMA_MOM_DECENT
        self.buy_ema_mom_min = buy_ema_mom_min if buy_ema_mom_min is not None else settings.MOMENTUM_BUY_EMA_MOM_MIN
        
        # RSI Parametreleri
        self.buy_rsi_excellent_min = buy_rsi_excellent_min if buy_rsi_excellent_min is not None else settings.MOMENTUM_BUY_RSI_EXCELLENT_MIN
        self.buy_rsi_excellent_max = buy_rsi_excellent_max if buy_rsi_excellent_max is not None else settings.MOMENTUM_BUY_RSI_EXCELLENT_MAX
        self.buy_rsi_good_min = buy_rsi_good_min if buy_rsi_good_min is not None else settings.MOMENTUM_BUY_RSI_GOOD_MIN
        self.buy_rsi_good_max = buy_rsi_good_max if buy_rsi_good_max is not None else settings.MOMENTUM_BUY_RSI_GOOD_MAX
        self.buy_rsi_extreme_min = buy_rsi_extreme_min if buy_rsi_extreme_min is not None else settings.MOMENTUM_BUY_RSI_EXTREME_MIN
        self.buy_rsi_extreme_max = buy_rsi_extreme_max if buy_rsi_extreme_max is not None else settings.MOMENTUM_BUY_RSI_EXTREME_MAX
        
        # ADX Parametreleri
        self.buy_adx_excellent = buy_adx_excellent if buy_adx_excellent is not None else settings.MOMENTUM_BUY_ADX_EXCELLENT
        self.buy_adx_good = buy_adx_good if buy_adx_good is not None else settings.MOMENTUM_BUY_ADX_GOOD
        self.buy_adx_decent = buy_adx_decent if buy_adx_decent is not None else settings.MOMENTUM_BUY_ADX_DECENT
        
        # Volume Parametreleri
        self.buy_volume_excellent = buy_volume_excellent if buy_volume_excellent is not None else settings.MOMENTUM_BUY_VOLUME_EXCELLENT
        self.buy_volume_good = buy_volume_good if buy_volume_good is not None else settings.MOMENTUM_BUY_VOLUME_GOOD
        self.buy_volume_decent = buy_volume_decent if buy_volume_decent is not None else settings.MOMENTUM_BUY_VOLUME_DECENT
        
        # Price Momentum
        self.buy_price_mom_excellent = buy_price_mom_excellent if buy_price_mom_excellent is not None else settings.MOMENTUM_BUY_PRICE_MOM_EXCELLENT
        self.buy_price_mom_good = buy_price_mom_good if buy_price_mom_good is not None else settings.MOMENTUM_BUY_PRICE_MOM_GOOD
        self.buy_price_mom_decent = buy_price_mom_decent if buy_price_mom_decent is not None else settings.MOMENTUM_BUY_PRICE_MOM_DECENT
        
        # Satış Koşulları
        self.sell_min_hold_minutes = sell_min_hold_minutes if sell_min_hold_minutes is not None else settings.MOMENTUM_SELL_MIN_HOLD_MINUTES
        self.sell_catastrophic_loss_pct = sell_catastrophic_loss_pct if sell_catastrophic_loss_pct is not None else settings.MOMENTUM_SELL_CATASTROPHIC_LOSS_PCT
        
        # Premium Kar Seviyeleri
        self.sell_premium_excellent = sell_premium_excellent if sell_premium_excellent is not None else settings.MOMENTUM_SELL_PREMIUM_EXCELLENT
        self.sell_premium_great = sell_premium_great if sell_premium_great is not None else settings.MOMENTUM_SELL_PREMIUM_GREAT
        self.sell_premium_good = sell_premium_good if sell_premium_good is not None else settings.MOMENTUM_SELL_PREMIUM_GOOD
        
        # Faz Parametreleri
        self.sell_phase1_excellent = sell_phase1_excellent if sell_phase1_excellent is not None else settings.MOMENTUM_SELL_PHASE1_EXCELLENT
        self.sell_phase1_good = sell_phase1_good if sell_phase1_good is not None else settings.MOMENTUM_SELL_PHASE1_GOOD
        self.sell_phase1_loss_protection = sell_phase1_loss_protection if sell_phase1_loss_protection is not None else settings.MOMENTUM_SELL_PHASE1_LOSS_PROTECTION
        
        self.sell_phase2_excellent = sell_phase2_excellent if sell_phase2_excellent is not None else settings.MOMENTUM_SELL_PHASE2_EXCELLENT
        self.sell_phase2_good = sell_phase2_good if sell_phase2_good is not None else settings.MOMENTUM_SELL_PHASE2_GOOD
        self.sell_phase2_decent = sell_phase2_decent if sell_phase2_decent is not None else settings.MOMENTUM_SELL_PHASE2_DECENT
        self.sell_phase2_loss_protection = sell_phase2_loss_protection if sell_phase2_loss_protection is not None else settings.MOMENTUM_SELL_PHASE2_LOSS_PROTECTION
        
        self.sell_phase3_excellent = sell_phase3_excellent if sell_phase3_excellent is not None else settings.MOMENTUM_SELL_PHASE3_EXCELLENT
        self.sell_phase3_good = sell_phase3_good if sell_phase3_good is not None else settings.MOMENTUM_SELL_PHASE3_GOOD
        self.sell_phase3_decent = sell_phase3_decent if sell_phase3_decent is not None else settings.MOMENTUM_SELL_PHASE3_DECENT
        self.sell_phase3_breakeven_min = sell_phase3_breakeven_min if sell_phase3_breakeven_min is not None else settings.MOMENTUM_SELL_PHASE3_BREAKEVEN_MIN
        self.sell_phase3_breakeven_max = sell_phase3_breakeven_max if sell_phase3_breakeven_max is not None else settings.MOMENTUM_SELL_PHASE3_BREAKEVEN_MAX
        self.sell_phase3_loss_protection = sell_phase3_loss_protection if sell_phase3_loss_protection is not None else settings.MOMENTUM_SELL_PHASE3_LOSS_PROTECTION
        
        self.sell_phase4_excellent = sell_phase4_excellent if sell_phase4_excellent is not None else settings.MOMENTUM_SELL_PHASE4_EXCELLENT
        self.sell_phase4_good = sell_phase4_good if sell_phase4_good is not None else settings.MOMENTUM_SELL_PHASE4_GOOD
        self.sell_phase4_minimal = sell_phase4_minimal if sell_phase4_minimal is not None else settings.MOMENTUM_SELL_PHASE4_MINIMAL
        self.sell_phase4_breakeven_min = sell_phase4_breakeven_min if sell_phase4_breakeven_min is not None else settings.MOMENTUM_SELL_PHASE4_BREAKEVEN_MIN
        self.sell_phase4_breakeven_max = sell_phase4_breakeven_max if sell_phase4_breakeven_max is not None else settings.MOMENTUM_SELL_PHASE4_BREAKEVEN_MAX
        self.sell_phase4_force_exit_minutes = sell_phase4_force_exit_minutes if sell_phase4_force_exit_minutes is not None else settings.MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES
        
        # Risk ve Teknik Satış
        self.sell_loss_multiplier = sell_loss_multiplier if sell_loss_multiplier is not None else settings.MOMENTUM_SELL_LOSS_MULTIPLIER
        self.sell_tech_min_minutes = sell_tech_min_minutes if sell_tech_min_minutes is not None else settings.MOMENTUM_SELL_TECH_MIN_MINUTES
        self.sell_tech_min_loss = sell_tech_min_loss if sell_tech_min_loss is not None else settings.MOMENTUM_SELL_TECH_MIN_LOSS
        self.sell_tech_rsi_extreme = sell_tech_rsi_extreme if sell_tech_rsi_extreme is not None else settings.MOMENTUM_SELL_TECH_RSI_EXTREME
        
        # Bekleme Süreleri
        self.wait_profit_5pct = wait_profit_5pct if wait_profit_5pct is not None else settings.MOMENTUM_WAIT_PROFIT_5PCT
        self.wait_profit_2pct = wait_profit_2pct if wait_profit_2pct is not None else settings.MOMENTUM_WAIT_PROFIT_2PCT
        self.wait_breakeven = wait_breakeven if wait_breakeven is not None else settings.MOMENTUM_WAIT_BREAKEVEN
        self.wait_loss = wait_loss if wait_loss is not None else settings.MOMENTUM_WAIT_LOSS
        
        # AI Parametreleri
        self.ai_confidence_threshold = ai_confidence_threshold if ai_confidence_threshold is not None else settings.AI_CONFIDENCE_THRESHOLD
        self.ai_momentum_confidence_override = ai_momentum_confidence_override if ai_momentum_confidence_override is not None else settings.AI_MOMENTUM_CONFIDENCE_OVERRIDE
        
        # AI Ağırlıkları
        self.ai_weight_trend_main = ai_weight_trend_main if ai_weight_trend_main is not None else settings.AI_TA_WEIGHT_TREND_MAIN
        self.ai_weight_trend_long = ai_weight_trend_long if ai_weight_trend_long is not None else settings.AI_TA_WEIGHT_TREND_LONG
        self.ai_weight_volume = ai_weight_volume if ai_weight_volume is not None else settings.AI_TA_WEIGHT_VOLUME
        self.ai_weight_divergence = ai_weight_divergence if ai_weight_divergence is not None else settings.AI_TA_WEIGHT_DIVERGENCE
        
        # AI Eşikleri
        self.ai_standalone_thresh_strong_buy = ai_standalone_thresh_strong_buy if ai_standalone_thresh_strong_buy is not None else settings.AI_TA_STANDALONE_THRESH_STRONG_BUY
        self.ai_standalone_thresh_buy = ai_standalone_thresh_buy if ai_standalone_thresh_buy is not None else settings.AI_TA_STANDALONE_THRESH_BUY
        self.ai_standalone_thresh_sell = ai_standalone_thresh_sell if ai_standalone_thresh_sell is not None else settings.AI_TA_STANDALONE_THRESH_SELL
        self.ai_standalone_thresh_strong_sell = ai_standalone_thresh_strong_sell if ai_standalone_thresh_strong_sell is not None else settings.AI_TA_STANDALONE_THRESH_STRONG_SELL
        
        # AI Onay Parametreleri
        self.ai_confirm_min_ta_score = ai_confirm_min_ta_score if ai_confirm_min_ta_score is not None else settings.AI_CONFIRM_MIN_TA_SCORE
        self.ai_confirm_min_quality_score = ai_confirm_min_quality_score if ai_confirm_min_quality_score is not None else settings.AI_CONFIRM_MIN_QUALITY_SCORE
        self.ai_confirm_min_ema_spread_1 = ai_confirm_min_ema_spread_1 if ai_confirm_min_ema_spread_1 is not None else settings.AI_CONFIRM_MIN_EMA_SPREAD_1
        self.ai_confirm_min_ema_spread_2 = ai_confirm_min_ema_spread_2 if ai_confirm_min_ema_spread_2 is not None else settings.AI_CONFIRM_MIN_EMA_SPREAD_2
        self.ai_confirm_min_volume_ratio = ai_confirm_min_volume_ratio if ai_confirm_min_volume_ratio is not None else settings.AI_CONFIRM_MIN_VOLUME_RATIO
        self.ai_confirm_min_price_momentum = ai_confirm_min_price_momentum if ai_confirm_min_price_momentum is not None else settings.AI_CONFIRM_MIN_PRICE_MOMENTUM
        self.ai_confirm_min_ema_momentum = ai_confirm_min_ema_momentum if ai_confirm_min_ema_momentum is not None else settings.AI_CONFIRM_MIN_EMA_MOMENTUM
        self.ai_confirm_min_adx = ai_confirm_min_adx if ai_confirm_min_adx is not None else settings.AI_CONFIRM_MIN_ADX
        
        # AI Zarar/Kar Eşikleri
        self.ai_confirm_loss_5pct_ta_score = ai_confirm_loss_5pct_ta_score if ai_confirm_loss_5pct_ta_score is not None else settings.AI_CONFIRM_LOSS_5PCT_TA_SCORE
        self.ai_confirm_loss_2pct_ta_score = ai_confirm_loss_2pct_ta_score if ai_confirm_loss_2pct_ta_score is not None else settings.AI_CONFIRM_LOSS_2PCT_TA_SCORE
        self.ai_confirm_profit_ta_score = ai_confirm_profit_ta_score if ai_confirm_profit_ta_score is not None else settings.AI_CONFIRM_PROFIT_TA_SCORE
        
        # AI Risk Değerlendirme
        self.ai_risk_volatility_threshold = ai_risk_volatility_threshold if ai_risk_volatility_threshold is not None else settings.AI_RISK_VOLATILITY_THRESHOLD
        self.ai_risk_volume_spike_threshold = ai_risk_volume_spike_threshold if ai_risk_volume_spike_threshold is not None else settings.AI_RISK_VOLUME_SPIKE_THRESHOLD
        
        # Global Risk
        self.global_max_position_size_pct = global_max_position_size_pct if global_max_position_size_pct is not None else settings.GLOBAL_MAX_POSITION_SIZE_PCT
        self.global_max_open_positions = global_max_open_positions if global_max_open_positions is not None else settings.GLOBAL_MAX_OPEN_POSITIONS
        self.global_max_portfolio_drawdown_pct = global_max_portfolio_drawdown_pct if global_max_portfolio_drawdown_pct is not None else settings.GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT
        self.global_max_daily_loss_pct = global_max_daily_loss_pct if global_max_daily_loss_pct is not None else settings.GLOBAL_MAX_DAILY_LOSS_PCT
        
        # Sistem Parametreleri
        self.min_time_between_trades_sec = min_time_between_trades_sec if min_time_between_trades_sec is not None else settings.MOMENTUM_MIN_TIME_BETWEEN_TRADES_SEC
        self.min_trade_amount_usdt = min_trade_amount_usdt if min_trade_amount_usdt is not None else settings.MIN_TRADE_AMOUNT_USDT
        
        self.last_trade_time = None
        self.position_entry_reasons = {}

        # --- AI Provider Kurulumu ---
        # Optimizasyondan gelen AI parametrelerini bir sözlükte topla
        ai_param_overrides = {
            "ai_assistance_enabled": settings.AI_ASSISTANCE_ENABLED, # Optimizasyon sırasında AI hep açık olsun
            "ai_confidence_threshold": self.ai_confidence_threshold,
            "ai_momentum_confidence_override": self.ai_momentum_confidence_override,
            "ai_weight_trend_main": self.ai_weight_trend_main,
            "ai_weight_trend_long": self.ai_weight_trend_long,
            "ai_weight_volume": self.ai_weight_volume,
            "ai_weight_divergence": self.ai_weight_divergence,
            "ai_standalone_thresh_strong_buy": self.ai_standalone_thresh_strong_buy,
            "ai_standalone_thresh_buy": self.ai_standalone_thresh_buy,
            "ai_standalone_thresh_sell": self.ai_standalone_thresh_sell,
            "ai_standalone_thresh_strong_sell": self.ai_standalone_thresh_strong_sell,
            "ai_confirm_min_ta_score": self.ai_confirm_min_ta_score,
            "ai_confirm_min_quality_score": self.ai_confirm_min_quality_score,
            "ai_confirm_min_ema_spread_1": self.ai_confirm_min_ema_spread_1,
            "ai_confirm_min_ema_spread_2": self.ai_confirm_min_ema_spread_2,
            "ai_confirm_min_volume_ratio": self.ai_confirm_min_volume_ratio,
            "ai_confirm_min_price_momentum": self.ai_confirm_min_price_momentum,
            "ai_confirm_min_ema_momentum": self.ai_confirm_min_ema_momentum,
            "ai_confirm_min_adx": self.ai_confirm_min_adx,
            "ai_confirm_loss_5pct_ta_score": self.ai_confirm_loss_5pct_ta_score,
            "ai_confirm_loss_2pct_ta_score": self.ai_confirm_loss_2pct_ta_score,
            "ai_confirm_profit_ta_score": self.ai_confirm_profit_ta_score,
            "ai_risk_volatility_threshold": self.ai_risk_volatility_threshold,
            "ai_risk_volume_spike_threshold": self.ai_risk_volume_spike_threshold
            # Not: config.py'de olan ancak optimizasyonda olmayan diğer AI parametreleri
            # (örn: AI_TA_EMA_PERIODS) varsayılan değerlerini alacaktır.
        }
        
        if ai_provider_instance:
            self.ai_provider = ai_provider_instance
        else:
            self.ai_provider = AiSignalProvider(overrides=ai_param_overrides) if settings.AI_ASSISTANCE_ENABLED else None
        
        logger.info(f"✅ {self.strategy_name} Strategy initialized for {self.symbol}")
        logger.info(f"   Technical: EMA({self.ema_short},{self.ema_medium},{self.ema_long}), RSI({self.rsi_period}), ADX({self.adx_period})")
        logger.info(f"   Position: {self.base_position_pct}% base, ${self.min_position_usdt}-${self.max_position_usdt}, Max Pos: {self.max_positions}")
        logger.info(f"   Buy Quality Min: {self.buy_min_quality_score}, AI: {'ON' if self.ai_provider and self.ai_provider.is_enabled else 'OFF'}"
                    )
        
    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Daha güçlü ve doğrulamalı indikatör hesaplama metodu."""
        
        # --- Parametre Doğrulama ---
        # Optimizasyon sırasında None gelme ihtimaline karşı parametreleri kontrol et
        required_params = {
            "ema_long": self.ema_long,
            "rsi_period": self.rsi_period,
            "adx_period": self.adx_period,
            "volume_sma_period": self.volume_sma_period,
            "atr_period": self.atr_period,
            "ema_short": self.ema_short,
            "ema_medium": self.ema_medium
        }
        for name, value in required_params.items():
            if value is None or not isinstance(value, (int, float)) or value <= 0:
                # Eğer bir parametre None ise veya geçersizse, hatayı logla ve None dön.
                # Bu, programın çökmesini engeller ve sorunun kaynağını netleştirir.
                logger.error(f"[{self.strategy_name}] Geçersiz indikatör parametresi: '{name}' değeri '{value}'. Optimizasyon ayarlarını kontrol edin.")
                return None
        
        # Gerekli veri uzunluğunu doğrulanmış parametrelerle güvenli bir şekilde hesapla
        min_data_length = max(self.ema_long, self.rsi_period, self.adx_period, self.volume_sma_period, 50)
        if len(df) < min_data_length:
            logger.debug(f"[{self.strategy_name}] İndikatörler için yeterli veri yok: {len(df)} bar, gerekli: {min_data_length} bar")
            return None
            
        try:
            df_copy = df.copy()
            indicators = pd.DataFrame(index=df_copy.index)
            
            indicators['close'] = df_copy['close']
            indicators['volume'] = df_copy['volume']
            indicators['high'] = df_copy['high'] 
            indicators['low'] = df_copy['low']
            
            # Parametrelerin geçerli olduğu artık bilindiği için güvenle kullanabiliriz.
            indicators['ema_short'] = ta.ema(df_copy['close'], length=self.ema_short)
            indicators['ema_medium'] = ta.ema(df_copy['close'], length=self.ema_medium)
            indicators['ema_long'] = ta.ema(df_copy['close'], length=self.ema_long)
            indicators['rsi'] = ta.rsi(df_copy['close'], length=self.rsi_period)
            
            adx_result = ta.adx(df_copy['high'], df_copy['low'], df_copy['close'], length=self.adx_period)
            if adx_result is not None and not adx_result.empty:
                indicators['adx'] = adx_result.iloc[:, 0]
            else:
                indicators['adx'] = 20.0 
            
            macd_result = ta.macd(df_copy['close'])
            if macd_result is not None and not macd_result.empty:
                indicators['macd'] = macd_result.iloc[:, 0]
                indicators['macd_signal'] = macd_result.iloc[:, 1]
                indicators['macd_hist'] = macd_result.iloc[:, 2]
            else:
                indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = 0.0, 0.0, 0.0
            
            indicators['volume_sma'] = ta.sma(df_copy['volume'], length=self.volume_sma_period)
            indicators['volume_ratio'] = (indicators['volume'] / indicators['volume_sma'].replace(0, 1e-9)).fillna(1.0)
            
            indicators['atr'] = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], length=self.atr_period)
            indicators['resistance'] = df_copy['high'].rolling(window=20).max()
            indicators['support'] = df_copy['low'].rolling(window=20).min()
            
            return indicators.tail(2)
            
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"🛑 [{self.strategy_name}] İndikatör hesaplaması kullanıcı tarafından durduruldu.")
            raise 
        except Exception as e:
            logger.error(f"[{self.strategy_name}] İndikatör hesaplama hatası: {e}", exc_info=True)
            return None
        
    def calculate_position_size(self, current_price: float) -> float:
        """Optimized position sizing with performance-based adjustments"""
        available_usdt = self.portfolio.get_available_usdt()
        initial_capital = self.portfolio.initial_capital_usdt
        current_portfolio_value = self.portfolio.get_total_portfolio_value_usdt(current_price)
        
        profit_pct = 0.0
        if initial_capital > 0:
            profit_pct = (current_portfolio_value - initial_capital) / initial_capital
        
        # Performansa dayalı yüzde seçimi - artık self parametrelerini kullanıyor
        if profit_pct > self.perf_high_profit_threshold:
            base_size_pct = self.size_high_profit_pct
        elif profit_pct > self.perf_good_profit_threshold:
            base_size_pct = self.size_good_profit_pct
        elif profit_pct > self.perf_normal_profit_threshold:
            base_size_pct = self.size_normal_profit_pct
        elif profit_pct > self.perf_breakeven_threshold:
            base_size_pct = self.size_breakeven_pct
        else:
            base_size_pct = self.size_loss_pct
        
        base_amount = available_usdt * (base_size_pct / 100.0)
        position_amount = max(self.min_position_usdt, min(base_amount, self.max_position_usdt))
        
        max_safe_amount = available_usdt * (self.size_max_balance_pct / 100.0)
        position_amount = min(position_amount, max_safe_amount)
        
        if position_amount > available_usdt * 0.95:
            position_amount = available_usdt * 0.95
            
        return position_amount

    def calculate_stop_loss(self, entry_price: float, indicators: Optional[pd.DataFrame]) -> float:
        """Calculate dynamic stop loss using optimized parameters"""
        base_sl_pct = self.max_loss_pct
        
        if indicators is not None and 'atr' in indicators.columns:
            current_atr = indicators.iloc[-1].get('atr')
            if pd.notna(current_atr) and entry_price > 0:
                volatility_ratio = current_atr / entry_price
                if volatility_ratio > 0.025:
                    base_sl_pct = self.max_loss_pct * 1.5
                elif volatility_ratio < 0.008:
                    base_sl_pct = self.max_loss_pct * 0.75
        
        sl_price = entry_price * (1 - base_sl_pct)
        return sl_price

    async def should_sell(self, position: Position, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """Enhanced sell logic with optimized parameters"""
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        
        sell_context: Dict[str, Any] = {
            "current_price": current_price, "position_id": position.position_id,
            "entry_price": position.entry_price, "indicators": {}
        }
        
        current_time = getattr(self, '_current_backtest_time', datetime.now(timezone.utc))
        
        position_age_minutes = 0.0
        if position.timestamp:
            try:
                entry_time = datetime.fromisoformat(position.timestamp.replace('Z', '+00:00'))
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)
                position_age_minutes = (current_time - entry_time).total_seconds() / 60.0
            except Exception as e:
                logger.warning(f"Error calculating position age for {position.position_id}: {e}")

        potential_gross = abs(position.quantity_btc) * current_price
        potential_fee = potential_gross * settings.FEE_SELL
        potential_net = potential_gross - potential_fee
        
        profit_usd = potential_net - position.entry_cost_usdt_total 
        profit_pct = (profit_usd / position.entry_cost_usdt_total) * 100 if position.entry_cost_usdt_total > 0 else 0

        sell_context.update({
            "position_age_minutes": position_age_minutes,
            "profit_usd": profit_usd,
            "profit_pct": profit_pct,
            "price_change_pct": (current_price - position.entry_price) / position.entry_price * 100 if position.entry_price > 0 else 0
        })

        # İndikatörleri hesapla
        indicators = await self.calculate_indicators(df)
        if indicators is not None and not indicators.empty:
            current_indicators = indicators.iloc[-1]
            sell_context["indicators"].update({
                "rsi": current_indicators.get('rsi', 50),
                "adx": current_indicators.get('adx', 20),
                "volume_ratio": current_indicators.get('volume_ratio', 1.0)
            })

        # Minimum tutma süresi kontrolü
        if position_age_minutes < self.sell_min_hold_minutes:
            if profit_pct < self.sell_catastrophic_loss_pct * 100:
                return True, f"EMERGENCY_EXIT_CATASTROPHIC_LOSS_{profit_pct:.1f}%", sell_context
            else:
                return False, f"FORCED_HOLD_{position_age_minutes:.0f}m_of_{self.sell_min_hold_minutes}m", sell_context
        
        # Premium kar seviyeleri
        if profit_usd >= self.sell_premium_excellent:
            return True, f"PREMIUM_EXCELLENT_PROFIT_${profit_usd:.2f}", sell_context
        elif profit_usd >= self.sell_premium_great:
            return True, f"PREMIUM_GREAT_PROFIT_${profit_usd:.2f}", sell_context
        elif profit_usd >= self.sell_premium_good:
            return True, f"PREMIUM_GOOD_PROFIT_${profit_usd:.2f}", sell_context

        # Fazlı satış mantığı
        if self.sell_min_hold_minutes <= position_age_minutes <= 60:
            if profit_usd >= self.sell_phase1_excellent: 
                return True, f"P1_EXC_PROFIT_${profit_usd:.2f}", sell_context
            if profit_usd >= self.sell_phase1_good: 
                return True, f"P1_GOOD_PROFIT_${profit_usd:.2f}", sell_context
            if profit_usd <= self.sell_phase1_loss_protection: 
                return True, f"P1_LOSS_PROT_${profit_usd:.2f}", sell_context
        elif 60 < position_age_minutes <= 120:
            if profit_usd >= self.sell_phase2_excellent: 
                return True, f"P2_EXC_PROFIT_${profit_usd:.2f}", sell_context
            if profit_usd >= self.sell_phase2_good: 
                return True, f"P2_GOOD_PROFIT_${profit_usd:.2f}", sell_context
            if profit_usd <= self.sell_phase2_loss_protection: 
                return True, f"P2_LOSS_PROT_${profit_usd:.2f}", sell_context
        elif 120 < position_age_minutes <= 180:
            if profit_usd >= self.sell_phase3_excellent: 
                return True, f"P3_EXC_PROFIT_${profit_usd:.2f}", sell_context
            if profit_usd >= self.sell_phase3_good: 
                return True, f"P3_GOOD_PROFIT_${profit_usd:.2f}", sell_context
            if self.sell_phase3_breakeven_min <= profit_usd <= self.sell_phase3_breakeven_max: 
                return True, f"P3_BREAKEVEN_${profit_usd:.2f}", sell_context
            if profit_usd <= self.sell_phase3_loss_protection: 
                return True, f"P3_LOSS_PROT_${profit_usd:.2f}", sell_context
        elif position_age_minutes >= self.sell_phase4_force_exit_minutes:
            return True, f"P4_FORCE_EXIT_{position_age_minutes:.0f}m_PROFIT_${profit_usd:.2f}", sell_context
        
        # Mutlak zarar limiti
        position_entry_cost = position.entry_cost_usdt_total
        total_trading_cost_estimate = (position_entry_cost * settings.FEE_BUY) + (potential_gross * settings.FEE_SELL)
        max_acceptable_loss_abs = total_trading_cost_estimate * self.sell_loss_multiplier
        if profit_usd <= -max_acceptable_loss_abs:
            return True, f"MAX_LOSS_LIMIT_${profit_usd:.2f}_vs_-${max_acceptable_loss_abs:.2f}", sell_context
        
        return False, f"HOLD_AGE_{position_age_minutes:.0f}m_PNL_${profit_usd:.2f}", sell_context

    async def should_buy(self, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """Enhanced buy logic with optimized parameters"""
        indicators = await self.calculate_indicators(df)
        if indicators is None or len(indicators) < 2:
            return False, "No or insufficient indicators", {}
            
        current = indicators.iloc[-1]
        previous = indicators.iloc[-2]
        current_price = current['close']
        
        buy_context: Dict[str, Any] = {
            "current_price": current_price, "quality_score": 0, "indicators": {}, 
            "reason": "", "strategy_name": self.strategy_name
        }
        
        open_positions = self.portfolio.get_open_positions(self.symbol, strategy_name=self.strategy_name)
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions ({len(open_positions)}/{self.max_positions})", buy_context
        
        required_amount = self.calculate_position_size(current_price)
        available_usdt = self.portfolio.get_available_usdt()
        if available_usdt < required_amount:
            return False, f"Insufficient balance (need ${required_amount:.2f}, have ${available_usdt:.2f})", buy_context
        
        # Bekleme süresi kontrolü
        if self.last_trade_time:
            current_time = getattr(self, '_current_backtest_time', datetime.now(timezone.utc))
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            seconds_since_last = (current_time - self.last_trade_time).total_seconds()
            if seconds_since_last < self.min_time_between_trades_sec:
                return False, f"Wait time {seconds_since_last:.0f}s of {self.min_time_between_trades_sec}s", buy_context
        
        quality_score = 0
        ema_short, ema_medium, ema_long = current.get('ema_short'), current.get('ema_medium'), current.get('ema_long')
        buy_context["indicators"].update({"ema_short": ema_short, "ema_medium": ema_medium, "ema_long": ema_long})

        # EMA trend kontrolü
        if not (pd.notna(ema_short) and pd.notna(ema_medium) and pd.notna(ema_long) and ema_short > ema_medium > ema_long):
            return False, "EMA_TREND_FAIL", buy_context
        
        # EMA spread kontrolü
        ema_spread_1 = (ema_short - ema_medium) / ema_medium if ema_medium else 0
        ema_spread_2 = (ema_medium - ema_long) / ema_long if ema_long else 0
        buy_context.update({"ema_spread_1": ema_spread_1, "ema_spread_2": ema_spread_2})
        
        if ema_spread_1 < self.buy_min_ema_spread_1 or ema_spread_2 < self.buy_min_ema_spread_2:
            return False, f"EMA_SPREAD_NARROW_{ema_spread_1*100:.3f}%_{ema_spread_2*100:.3f}%", buy_context

        # EMA momentum değerlendirmesi
        prev_ema_short = previous.get('ema_short')
        if pd.notna(prev_ema_short) and prev_ema_short > 0:
            ema_momentum = (ema_short - prev_ema_short) / prev_ema_short
            buy_context["ema_momentum"] = ema_momentum
            if ema_momentum > self.buy_ema_mom_excellent:
                quality_score += 4
            elif ema_momentum > self.buy_ema_mom_good:
                quality_score += 3
            elif ema_momentum > self.buy_ema_mom_decent:
                quality_score += 2
            elif ema_momentum > self.buy_ema_mom_min:
                quality_score += 1
            else:
                return False, f"EMA_MOM_LOW_{ema_momentum*100:.3f}%", buy_context
        else:
            quality_score += 1
        
        # RSI değerlendirmesi
        rsi = current.get('rsi')
        buy_context["indicators"]["rsi"] = rsi
        if pd.notna(rsi):
            if self.buy_rsi_extreme_min < rsi < self.buy_rsi_extreme_max:
                if self.buy_rsi_excellent_min <= rsi <= self.buy_rsi_excellent_max:
                    quality_score += 3
                elif self.buy_rsi_good_min <= rsi <= self.buy_rsi_good_max:
                    quality_score += 2
                else:
                    quality_score += 1
            else:
                return False, f"RSI_EXTREME_{rsi:.1f}", buy_context
        
        # ADX değerlendirmesi
        adx = current.get('adx')
        buy_context["indicators"]["adx"] = adx
        if pd.notna(adx):
            if adx > self.buy_adx_excellent:
                quality_score += 3
            elif adx > self.buy_adx_good:
                quality_score += 2
            elif adx > self.buy_adx_decent:
                quality_score += 1
        
        # Volume değerlendirmesi
        volume_ratio = current.get('volume_ratio', 1.0)
        buy_context["indicators"]["volume_ratio"] = volume_ratio
        if volume_ratio > self.buy_volume_excellent:
            quality_score += 3
        elif volume_ratio > self.buy_volume_good:
            quality_score += 2
        elif volume_ratio > self.buy_volume_decent:
            quality_score += 1
        
        # Price momentum değerlendirmesi
        prev_close = previous.get('close')
        if pd.notna(prev_close) and prev_close > 0:
            price_momentum = (current_price - prev_close) / prev_close
            buy_context["price_momentum"] = price_momentum
            if price_momentum > self.buy_price_mom_excellent:
                quality_score += 3
            elif price_momentum > self.buy_price_mom_good:
                quality_score += 2
            elif price_momentum > self.buy_price_mom_decent:
                quality_score += 1
        
        buy_context["quality_score"] = quality_score
        buy_context["min_quality_required"] = self.buy_min_quality_score
        
        if quality_score < self.buy_min_quality_score:
            return False, f"QUALITY_LOW_Q{quality_score} (MinReq:{self.buy_min_quality_score})", buy_context
        
        # AI onayı
        buy_context["ai_approved"] = "N/A"
        if self.ai_provider:
            ai_confirmation = await self.ai_provider.get_ai_confirmation(
                current_signal_type="BUY", ohlcv_df=df, context=buy_context
            )
            buy_context["ai_approved"] = str(ai_confirmation)
            if not ai_confirmation:
                return False, f"AI_REJECT_Q{quality_score}", buy_context
        
        buy_reason_final = f"MOM_BUY_Q{quality_score}"
        if self.ai_provider and buy_context["ai_approved"] == "True":
            buy_reason_final += "_AI_OK"
        
        buy_context["reason"] = buy_reason_final
        return True, buy_reason_final, buy_context

    async def process_data(self, df: pd.DataFrame) -> None:
        """Main strategy processing logic"""
        try:
            if df.empty:
                return
                
            current_bar = df.iloc[-1]
            current_price = current_bar['close']
            
            current_time_for_process = getattr(self, '_current_backtest_time', datetime.now(timezone.utc))
            current_time_iso = current_time_for_process.isoformat()
            
            # Mevcut pozisyonları kontrol et ve satış kararları ver
            open_positions = self.portfolio.get_open_positions(self.symbol, strategy_name=self.strategy_name)
            
            for position in list(open_positions):
                should_sell_flag, sell_reason, sell_context_dict = await self.should_sell(position, df)
                if should_sell_flag:
                    await self.portfolio.execute_sell(
                        position_to_close=position, current_price=current_price,
                        timestamp=current_time_iso, reason=sell_reason, sell_context=sell_context_dict
                    )
            
            # Satıştan sonra pozisyonları yeniden kontrol et
            open_positions_after_sell = self.portfolio.get_open_positions(self.symbol, strategy_name=self.strategy_name)

            # Yeni alım fırsatlarını değerlendir
            if len(open_positions_after_sell) < self.max_positions:
                should_buy_flag, buy_reason_str, buy_context_dict = await self.should_buy(df)
                if should_buy_flag:
                    position_amount = self.calculate_position_size(current_price)
                    indicators_for_sl = await self.calculate_indicators(df)
                    stop_loss_price = self.calculate_stop_loss(current_price, indicators_for_sl)
                    
                    new_position = await self.portfolio.execute_buy(
                        strategy_name=self.strategy_name, symbol=self.symbol,
                        current_price=current_price, timestamp=current_time_iso,
                        reason=buy_reason_str, 
                        amount_usdt_override=position_amount,
                        stop_loss_price_from_strategy=stop_loss_price,
                        buy_context=buy_context_dict
                    )
                    if new_position:
                        self.position_entry_reasons[new_position.position_id] = buy_reason_str
                        self.last_trade_time = current_time_for_process
                
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"🛑 [{self.strategy_name}] Strategy processing interrupted by user")
            raise
        except Exception as e:
            logger.error(f"[{self.strategy_name}] Process data error: {e}", exc_info=True)