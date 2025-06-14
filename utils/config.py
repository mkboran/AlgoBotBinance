# utils/config.py - Advanced Trading Bot Configuration

import os
from pathlib import Path
from typing import Optional, Final, Tuple # Tuple import edildiğinden emin olun
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field # Field import edildiğinden emin olun
from dotenv import load_dotenv

# Load environment variables
if not globals().get("_CONFIG_INITIALIZED", False):
    env_path = Path(".") / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ .env file loaded: {env_path}")
    else:
        print(f"ℹ️  .env file not found at {env_path}. Using environment variables or defaults.")
    _CONFIG_INITIALIZED = True

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ .env file loaded: {env_path}")
else:
    print(f"ℹ️  .env file not found at {env_path}. Using environment variables or defaults.")

def parse_bool_env(env_var: str, default: str) -> bool:
    """Parse boolean from environment variable safely"""
    value = os.getenv(env_var, default).lower()
    return value in ('true', '1', 'yes', 'on', 'enabled')

class Settings(BaseSettings):
    """🚀 Advanced Trading Bot Configuration - Optimized for Performance"""
    
    ENABLE_CSV_LOGGING: bool = Field(default=True, env="ENABLE_CSV_LOGGING") # Yeni ayar

    # ================================================================================
    # 🔐 API CREDENTIALS (Optional - for live trading)
    # ================================================================================
    BINANCE_API_KEY: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    BINANCE_API_SECRET: Optional[str] = Field(default=None, env="BINANCE_API_SECRET")
    
    # ================================================================================
    # 📊 CORE TRADING SETTINGS
    # ================================================================================
    INITIAL_CAPITAL_USDT: float = Field(default=1000.0, env="INITIAL_CAPITAL_USDT")
    SYMBOL: str = Field(default="BTC/USDT", env="SYMBOL")
    TIMEFRAME: str = Field(default="15m", env="TIMEFRAME")
    
    FEE_BUY: float = Field(default=0.001, env="FEE_BUY")
    FEE_SELL: float = Field(default=0.001, env="FEE_SELL")
    
    OHLCV_LIMIT: int = Field(default=250, env="OHLCV_LIMIT")
    
    MIN_TRADE_AMOUNT_USDT: float = Field(default=25.0, env="MIN_TRADE_AMOUNT_USDT") 
    
    PRICE_PRECISION: int = Field(default=2, env="PRICE_PRECISION") # YENİ EKLENDİ
    ASSET_PRECISION: int = Field(default=6, env="ASSET_PRECISION") # YENİ EKLENDİ
    
    # ================================================================================
    # 📊 DATA FETCHING CONFIGURATION
    # ================================================================================
    DATA_FETCHER_RETRY_ATTEMPTS: int = Field(default=3, env="DATA_FETCHER_RETRY_ATTEMPTS")
    DATA_FETCHER_RETRY_MULTIPLIER: float = Field(default=1.0, env="DATA_FETCHER_RETRY_MULTIPLIER")
    DATA_FETCHER_RETRY_MIN_WAIT: float = Field(default=1.0, env="DATA_FETCHER_RETRY_MIN_WAIT")
    DATA_FETCHER_RETRY_MAX_WAIT: float = Field(default=10.0, env="DATA_FETCHER_RETRY_MAX_WAIT")
    DATA_FETCHER_TIMEOUT_SECONDS: int = Field(default=30, env="DATA_FETCHER_TIMEOUT_SECONDS")
    LOOP_SLEEP_SECONDS: int = Field(default=5, env="LOOP_SLEEP_SECONDS")
    LOOP_SLEEP_SECONDS_ON_DATA_ERROR: int = Field(default=15, env="LOOP_SLEEP_SECONDS_ON_DATA_ERROR")
    
    # ================================================================================
    # 📝 LOGGING CONFIGURATION
    # ================================================================================
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_TO_FILE: bool = Field(default=True, env="LOG_TO_FILE") 
    TRADES_CSV_LOG_PATH: str = Field(default="logs/trades.csv", env="TRADES_CSV_LOG_PATH")
    # TRADES_JSONL_LOG_PATH: str = Field(default="logs/trades.jsonl", env="TRADES_JSONL_LOG_PATH") # İhtiyaç yoksa kaldırılabilir
    
    # ================================================================================
    # 🚀 MOMENTUM STRATEGY CONFIGURATION
    # ================================================================================
    MOMENTUM_EMA_SHORT: int = Field(default=13, env="MOMENTUM_EMA_SHORT")
    MOMENTUM_EMA_MEDIUM: int = Field(default=21, env="MOMENTUM_EMA_MEDIUM")
    MOMENTUM_EMA_LONG: int = Field(default=56, env="MOMENTUM_EMA_LONG")
    MOMENTUM_RSI_PERIOD: int = Field(default=13, env="MOMENTUM_RSI_PERIOD")
    MOMENTUM_ADX_PERIOD: int = Field(default=25, env="MOMENTUM_ADX_PERIOD")
    MOMENTUM_ATR_PERIOD: int = Field(default=18, env="MOMENTUM_ATR_PERIOD")
    MOMENTUM_VOLUME_SMA_PERIOD: int = Field(default=29, env="MOMENTUM_VOLUME_SMA_PERIOD")
    
    MOMENTUM_BASE_POSITION_SIZE_PCT: float = Field(default=45.0, env="MOMENTUM_BASE_POSITION_SIZE_PCT")
    MOMENTUM_MIN_POSITION_USDT: float = Field(default=400.0, env="MOMENTUM_MIN_POSITION_USDT")
    MOMENTUM_MAX_POSITION_USDT: float = Field(default=800.0, env="MOMENTUM_MAX_POSITION_USDT") # Düşürülmüştü
    MOMENTUM_MAX_POSITIONS: int = Field(default=6, env="MOMENTUM_MAX_POSITIONS")
    MOMENTUM_MAX_TOTAL_EXPOSURE_PCT: float = Field(default=60.0, env="MOMENTUM_MAX_TOTAL_EXPOSURE_PCT") # Düşürülmüştü
    
    MOMENTUM_SIZE_HIGH_PROFIT_PCT: float = Field(default=50.0, env="MOMENTUM_SIZE_HIGH_PROFIT_PCT")  # 23.0 → 50.0
    MOMENTUM_SIZE_GOOD_PROFIT_PCT: float = Field(default=35.0, env="MOMENTUM_SIZE_GOOD_PROFIT_PCT")  # 12.0 → 35.0
    MOMENTUM_SIZE_NORMAL_PROFIT_PCT: float = Field(default=25.0, env="MOMENTUM_SIZE_NORMAL_PROFIT_PCT")  # 12.0 → 25.0
    MOMENTUM_SIZE_BREAKEVEN_PCT: float = Field(default=15.0, env="MOMENTUM_SIZE_BREAKEVEN_PCT")  # 16.0 → 15.0
    MOMENTUM_SIZE_LOSS_PCT: float = Field(default=8.0, env="MOMENTUM_SIZE_LOSS_PCT")  # 6.0 → 8.0
    MOMENTUM_SIZE_MAX_BALANCE_PCT: float = Field(default=45.0, env="MOMENTUM_SIZE_MAX_BALANCE_PCT")  # 18.0 → 45.0

    MOMENTUM_PERF_HIGH_PROFIT_THRESHOLD: float = Field(default=0.060000000000000005, env="MOMENTUM_PERF_HIGH_PROFIT_THRESHOLD")
    MOMENTUM_PERF_GOOD_PROFIT_THRESHOLD: float = Field(default=0.06, env="MOMENTUM_PERF_GOOD_PROFIT_THRESHOLD")
    MOMENTUM_PERF_NORMAL_PROFIT_THRESHOLD: float = Field(default=0.05, env="MOMENTUM_PERF_NORMAL_PROFIT_THRESHOLD")
    MOMENTUM_PERF_BREAKEVEN_THRESHOLD: float = Field(default=-0.020000000000000004, env="MOMENTUM_PERF_BREAKEVEN_THRESHOLD")
    
    MOMENTUM_MAX_LOSS_PCT: float = Field(default=0.025, env="MOMENTUM_MAX_LOSS_PCT")  # 0.007 → 0.025
    MOMENTUM_MIN_PROFIT_TARGET_USDT: float = Field(default=5.0, env="MOMENTUM_MIN_PROFIT_TARGET_USDT")  # 0.75 → 5.0
    MOMENTUM_QUICK_PROFIT_THRESHOLD_USDT: float = Field(default=3.0, env="MOMENTUM_QUICK_PROFIT_THRESHOLD_USDT")  # 0.50 → 3.0

    MOMENTUM_MAX_HOLD_MINUTES: int = Field(default=60, env="MOMENTUM_MAX_HOLD_MINUTES")
    MOMENTUM_BREAKEVEN_MINUTES: int = Field(default=5, env="MOMENTUM_BREAKEVEN_MINUTES")
    MOMENTUM_MIN_TIME_BETWEEN_TRADES_SEC: int = Field(default=15, env="MOMENTUM_MIN_TIME_BETWEEN_TRADES_SEC")

    MOMENTUM_BUY_MIN_QUALITY_SCORE: int = Field(default=8, env="MOMENTUM_BUY_MIN_QUALITY_SCORE")  # 15 → 8
    MOMENTUM_BUY_MIN_EMA_SPREAD_1: float = Field(default=0.0001, env="MOMENTUM_BUY_MIN_EMA_SPREAD_1")  # 0.00026 → 0.0001
    MOMENTUM_BUY_MIN_EMA_SPREAD_2: float = Field(default=0.00008, env="MOMENTUM_BUY_MIN_EMA_SPREAD_2")  # 0.00014 → 0.00008
    MOMENTUM_BUY_EMA_MOM_EXCELLENT: float = Field(default=0.0014000000000000002, env="MOMENTUM_BUY_EMA_MOM_EXCELLENT")
    MOMENTUM_BUY_EMA_MOM_GOOD: float = Field(default=0.0005, env="MOMENTUM_BUY_EMA_MOM_GOOD")
    MOMENTUM_BUY_EMA_MOM_DECENT: float = Field(default=0.0004, env="MOMENTUM_BUY_EMA_MOM_DECENT")
    MOMENTUM_BUY_EMA_MOM_MIN: float = Field(default=1.3395490493134142e-05, env="MOMENTUM_BUY_EMA_MOM_MIN")
    MOMENTUM_BUY_RSI_EXCELLENT_MIN: float = Field(default=17.5, env="MOMENTUM_BUY_RSI_EXCELLENT_MIN")
    MOMENTUM_BUY_RSI_EXCELLENT_MAX: float = Field(default=75.0, env="MOMENTUM_BUY_RSI_EXCELLENT_MAX")
    MOMENTUM_BUY_RSI_GOOD_MIN: float = Field(default=12.5, env="MOMENTUM_BUY_RSI_GOOD_MIN")
    MOMENTUM_BUY_RSI_GOOD_MAX: float = Field(default=85.0, env="MOMENTUM_BUY_RSI_GOOD_MAX")
    MOMENTUM_BUY_RSI_EXTREME_MIN: float = Field(default=6.0, env="MOMENTUM_BUY_RSI_EXTREME_MIN")
    MOMENTUM_BUY_RSI_EXTREME_MAX: float = Field(default=90.0, env="MOMENTUM_BUY_RSI_EXTREME_MAX")
    MOMENTUM_BUY_ADX_EXCELLENT: float = Field(default=20.0, env="MOMENTUM_BUY_ADX_EXCELLENT")
    MOMENTUM_BUY_ADX_GOOD: float = Field(default=21.0, env="MOMENTUM_BUY_ADX_GOOD")
    MOMENTUM_BUY_ADX_DECENT: float = Field(default=18.0, env="MOMENTUM_BUY_ADX_DECENT")
    MOMENTUM_BUY_VOLUME_EXCELLENT: float = Field(default=2.6, env="MOMENTUM_BUY_VOLUME_EXCELLENT")
    MOMENTUM_BUY_VOLUME_GOOD: float = Field(default=1.1, env="MOMENTUM_BUY_VOLUME_GOOD")
    MOMENTUM_BUY_VOLUME_DECENT: float = Field(default=1.5, env="MOMENTUM_BUY_VOLUME_DECENT")
    MOMENTUM_BUY_PRICE_MOM_EXCELLENT: float = Field(default=0.001, env="MOMENTUM_BUY_PRICE_MOM_EXCELLENT")
    MOMENTUM_BUY_PRICE_MOM_GOOD: float = Field(default=0.0001, env="MOMENTUM_BUY_PRICE_MOM_GOOD")
    MOMENTUM_BUY_PRICE_MOM_DECENT: float = Field(default=-0.001, env="MOMENTUM_BUY_PRICE_MOM_DECENT")
    
    MOMENTUM_SELL_MIN_HOLD_MINUTES: int = Field(default=15, env="MOMENTUM_SELL_MIN_HOLD_MINUTES")
    MOMENTUM_SELL_CATASTROPHIC_LOSS_PCT: float = Field(default=-0.035, env="MOMENTUM_SELL_CATASTROPHIC_LOSS_PCT")
    MOMENTUM_SELL_PREMIUM_EXCELLENT: float = Field(default=6.5, env="MOMENTUM_SELL_PREMIUM_EXCELLENT")
    MOMENTUM_SELL_PREMIUM_GREAT: float = Field(default=4.0, env="MOMENTUM_SELL_PREMIUM_GREAT")
    MOMENTUM_SELL_PREMIUM_GOOD: float = Field(default=2.75, env="MOMENTUM_SELL_PREMIUM_GOOD")
    MOMENTUM_SELL_PHASE1_EXCELLENT: float = Field(default=1.0, env="MOMENTUM_SELL_PHASE1_EXCELLENT")
    MOMENTUM_SELL_PHASE1_GOOD: float = Field(default=1.25, env="MOMENTUM_SELL_PHASE1_GOOD")
    MOMENTUM_SELL_PHASE1_LOSS_PROTECTION: float = Field(default=-1.5, env="MOMENTUM_SELL_PHASE1_LOSS_PROTECTION")
    MOMENTUM_SELL_PHASE2_EXCELLENT: float = Field(default=1.25, env="MOMENTUM_SELL_PHASE2_EXCELLENT")
    MOMENTUM_SELL_PHASE2_GOOD: float = Field(default=1.0, env="MOMENTUM_SELL_PHASE2_GOOD")
    MOMENTUM_SELL_PHASE2_DECENT: float = Field(default=1.00, env="MOMENTUM_SELL_PHASE2_DECENT")
    MOMENTUM_SELL_PHASE2_LOSS_PROTECTION: float = Field(default=-2.5, env="MOMENTUM_SELL_PHASE2_LOSS_PROTECTION")
    MOMENTUM_SELL_PHASE3_EXCELLENT: float = Field(default=1.25, env="MOMENTUM_SELL_PHASE3_EXCELLENT")
    MOMENTUM_SELL_PHASE3_GOOD: float = Field(default=1.0, env="MOMENTUM_SELL_PHASE3_GOOD")
    MOMENTUM_SELL_PHASE3_DECENT: float = Field(default=0.5, env="MOMENTUM_SELL_PHASE3_DECENT")
    MOMENTUM_SELL_PHASE3_BREAKEVEN_MIN: float = Field(default=-0.15000000000000002, env="MOMENTUM_SELL_PHASE3_BREAKEVEN_MIN")
    MOMENTUM_SELL_PHASE3_BREAKEVEN_MAX: float = Field(default=0.25, env="MOMENTUM_SELL_PHASE3_BREAKEVEN_MAX")
    MOMENTUM_SELL_PHASE3_LOSS_PROTECTION: float = Field(default=-1.0, env="MOMENTUM_SELL_PHASE3_LOSS_PROTECTION")
    MOMENTUM_SELL_PHASE4_EXCELLENT: float = Field(default=0.4, env="MOMENTUM_SELL_PHASE4_EXCELLENT")
    MOMENTUM_SELL_PHASE4_GOOD: float = Field(default=0.2, env="MOMENTUM_SELL_PHASE4_GOOD")
    MOMENTUM_SELL_PHASE4_MINIMAL: float = Field(default=0.15000000000000002, env="MOMENTUM_SELL_PHASE4_MINIMAL")
    MOMENTUM_SELL_PHASE4_BREAKEVEN_MIN: float = Field(default=-0.2, env="MOMENTUM_SELL_PHASE4_BREAKEVEN_MIN")
    MOMENTUM_SELL_PHASE4_BREAKEVEN_MAX: float = Field(default=0.35000000000000003, env="MOMENTUM_SELL_PHASE4_BREAKEVEN_MAX")
    MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES: int = Field(default=240, env="MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES")
    MOMENTUM_SELL_LOSS_MULTIPLIER: float = Field(default=6.0, env="MOMENTUM_SELL_LOSS_MULTIPLIER")
    MOMENTUM_SELL_TECH_MIN_MINUTES: int = Field(default=75, env="MOMENTUM_SELL_TECH_MIN_MINUTES")
    MOMENTUM_SELL_TECH_MIN_LOSS: float = Field(default=-3.0, env="MOMENTUM_SELL_TECH_MIN_LOSS")
    MOMENTUM_SELL_TECH_RSI_EXTREME: float = Field(default=19.0, env="MOMENTUM_SELL_TECH_RSI_EXTREME")
    
    MOMENTUM_WAIT_PROFIT_5PCT: int = Field(default=180, env="MOMENTUM_WAIT_PROFIT_5PCT")
    MOMENTUM_WAIT_PROFIT_2PCT: int = Field(default=660, env="MOMENTUM_WAIT_PROFIT_2PCT")
    MOMENTUM_WAIT_BREAKEVEN: int = Field(default=810, env="MOMENTUM_WAIT_BREAKEVEN")
    MOMENTUM_WAIT_LOSS: int = Field(default=720, env="MOMENTUM_WAIT_LOSS")

    # ================================================================================
    # 🎯 BOLLINGER RSI STRATEGY CONFIGURATION
    # ================================================================================
    BOLLINGER_RSI_BB_PERIOD: int = Field(default=20, env="BOLLINGER_RSI_BB_PERIOD")
    BOLLINGER_RSI_BB_STD_DEV: float = Field(default=2.0, env="BOLLINGER_RSI_BB_STD_DEV")
    BOLLINGER_RSI_RSI_PERIOD: int = Field(default=14, env="BOLLINGER_RSI_RSI_PERIOD")
    BOLLINGER_RSI_VOLUME_SMA_PERIOD: int = Field(default=20, env="BOLLINGER_RSI_VOLUME_SMA_PERIOD")
    BOLLINGER_RSI_BASE_POSITION_SIZE_PCT: float = Field(default=6.0, env="BOLLINGER_RSI_BASE_POSITION_SIZE_PCT")
    BOLLINGER_RSI_MAX_POSITION_USDT: float = Field(default=150.0, env="BOLLINGER_RSI_MAX_POSITION_USDT")
    BOLLINGER_RSI_MIN_POSITION_USDT: float = Field(default=100.0, env="BOLLINGER_RSI_MIN_POSITION_USDT")
    BOLLINGER_RSI_MAX_POSITIONS: int = Field(default=2, env="BOLLINGER_RSI_MAX_POSITIONS")
    BOLLINGER_RSI_MAX_TOTAL_EXPOSURE_PCT: float = Field(default=15.0, env="BOLLINGER_RSI_MAX_TOTAL_EXPOSURE_PCT")
    BOLLINGER_RSI_MAX_LOSS_PCT: float = Field(default=0.006, env="BOLLINGER_RSI_MAX_LOSS_PCT")
    BOLLINGER_RSI_MIN_PROFIT_TARGET_USDT: float = Field(default=1.20, env="BOLLINGER_RSI_MIN_PROFIT_TARGET_USDT")
    BOLLINGER_RSI_QUICK_PROFIT_THRESHOLD_USDT: float = Field(default=0.60, env="BOLLINGER_RSI_QUICK_PROFIT_THRESHOLD_USDT")
    BOLLINGER_RSI_MAX_HOLD_MINUTES: int = Field(default=45, env="BOLLINGER_RSI_MAX_HOLD_MINUTES")
    BOLLINGER_RSI_BREAKEVEN_MINUTES: int = Field(default=5, env="BOLLINGER_RSI_BREAKEVEN_MINUTES")
    BOLLINGER_RSI_MIN_TIME_BETWEEN_TRADES_SEC: int = Field(default=45, env="BOLLINGER_RSI_MIN_TIME_BETWEEN_TRADES_SEC")
    
    # ================================================================================
    # 🛡️ GLOBAL RISK MANAGEMENT CONFIGURATION
    # ================================================================================
    GLOBAL_MAX_POSITION_SIZE_PCT: float = Field(default=15.0, env="GLOBAL_MAX_POSITION_SIZE_PCT")
    GLOBAL_MAX_OPEN_POSITIONS: int = Field(default=6, env="GLOBAL_MAX_OPEN_POSITIONS")
    GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT: float = Field(default=0.17500000000000002, env="GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT")
    GLOBAL_MAX_DAILY_LOSS_PCT: float = Field(default=0.02, env="GLOBAL_MAX_DAILY_LOSS_PCT")
    DRAWDOWN_LIMIT_HIGH_VOL_REGIME_PCT: Optional[float] = Field(default=0.15, env="DRAWDOWN_LIMIT_HIGH_VOL_REGIME_PCT")
    
    # ================================================================================
    # 🤖 ADVANCED AI ASSISTANCE CONFIGURATION - ENHANCED
    # ================================================================================
    AI_ASSISTANCE_ENABLED: bool = Field(default=parse_bool_env("AI_ASSISTANCE_ENABLED", "true"), env="AI_ASSISTANCE_ENABLED")
    AI_OPERATION_MODE: str = Field(default="technical_analysis", env="AI_OPERATION_MODE")
    AI_CONFIDENCE_THRESHOLD: float = Field(default=0.15000000000000002, env="AI_CONFIDENCE_THRESHOLD")
    AI_MODEL_PATH: Optional[str] = Field(default=None, env="AI_MODEL_PATH")
    
    AI_TA_EMA_PERIODS_MAIN_TF: Tuple[int, int, int] = Field(default=tuple(map(int, os.getenv("AI_TA_EMA_PERIODS_MAIN_TF", "9,21,50").split(','))), env="AI_TA_EMA_PERIODS_MAIN_TF")
    AI_TA_EMA_PERIODS_LONG_TF: Tuple[int, int, int] = Field(default=tuple(map(int, os.getenv("AI_TA_EMA_PERIODS_LONG_TF", "18,63,150").split(','))), env="AI_TA_EMA_PERIODS_LONG_TF")
    AI_TA_RSI_PERIOD: int = Field(default=int(os.getenv("AI_TA_RSI_PERIOD", "14")), env="AI_TA_RSI_PERIOD")
    AI_TA_DIVERGENCE_LOOKBACK: int = Field(default=int(os.getenv("AI_TA_DIVERGENCE_LOOKBACK", "10")), env="AI_TA_DIVERGENCE_LOOKBACK")
    AI_TA_LONG_TIMEFRAME_STR: str = Field(default=os.getenv("AI_TA_LONG_TIMEFRAME_STR", "1h"), env="AI_TA_LONG_TIMEFRAME_STR") # GÜNCELLENDİ
    
    AI_TA_WEIGHT_TREND_MAIN: float = Field(default=float(os.getenv("AI_TA_WEIGHT_TREND_MAIN", "0.6")), env="AI_TA_WEIGHT_TREND_MAIN")
    AI_TA_WEIGHT_TREND_LONG: float = Field(default=float(os.getenv("AI_TA_WEIGHT_TREND_LONG", "0.25")), env="AI_TA_WEIGHT_TREND_LONG")
    AI_TA_WEIGHT_VOLUME: float = Field(default=float(os.getenv("AI_TA_WEIGHT_VOLUME", "0.3")), env="AI_TA_WEIGHT_VOLUME")
    AI_TA_WEIGHT_DIVERGENCE: float = Field(default=float(os.getenv("AI_TA_WEIGHT_DIVERGENCE", "0.15000000000000002")), env="AI_TA_WEIGHT_DIVERGENCE")
    
    AI_TA_STANDALONE_THRESH_STRONG_BUY: float = Field(default=float(os.getenv("AI_TA_STANDALONE_THRESH_STRONG_BUY", "0.65")), env="AI_TA_STANDALONE_THRESH_STRONG_BUY")
    AI_TA_STANDALONE_THRESH_BUY: float = Field(default=float(os.getenv("AI_TA_STANDALONE_THRESH_BUY", "0.30000000000000004")), env="AI_TA_STANDALONE_THRESH_BUY")
    AI_TA_STANDALONE_THRESH_SELL: float = Field(default=float(os.getenv("AI_TA_STANDALONE_THRESH_SELL", "-0.39999999999999997")), env="AI_TA_STANDALONE_THRESH_SELL")
    AI_TA_STANDALONE_THRESH_STRONG_SELL: float = Field(default=float(os.getenv("AI_TA_STANDALONE_THRESH_STRONG_SELL", "-0.6")), env="AI_TA_STANDALONE_THRESH_STRONG_SELL")
    
    AI_CONFIRM_MIN_TA_SCORE: float = Field(default=float(os.getenv("AI_CONFIRM_MIN_TA_SCORE", "0.25")), env="AI_CONFIRM_MIN_TA_SCORE")
    AI_CONFIRM_MIN_QUALITY_SCORE: int = Field(default=int(os.getenv("AI_CONFIRM_MIN_QUALITY_SCORE", "2")), env="AI_CONFIRM_MIN_QUALITY_SCORE")
    AI_CONFIRM_MIN_EMA_SPREAD_1: float = Field(default=float(os.getenv("AI_CONFIRM_MIN_EMA_SPREAD_1", "0.0006000000000000001")), env="AI_CONFIRM_MIN_EMA_SPREAD_1")
    AI_CONFIRM_MIN_EMA_SPREAD_2: float = Field(default=float(os.getenv("AI_CONFIRM_MIN_EMA_SPREAD_2", "0.001")), env="AI_CONFIRM_MIN_EMA_SPREAD_2")
    AI_CONFIRM_MIN_VOLUME_RATIO: float = Field(default=float(os.getenv("AI_CONFIRM_MIN_VOLUME_RATIO", "2.0")), env="AI_CONFIRM_MIN_VOLUME_RATIO")
    AI_CONFIRM_MIN_PRICE_MOMENTUM: float = Field(default=float(os.getenv("AI_CONFIRM_MIN_PRICE_MOMENTUM", "0.0009000000000000001")), env="AI_CONFIRM_MIN_PRICE_MOMENTUM")
    AI_CONFIRM_MIN_EMA_MOMENTUM: float = Field(default=float(os.getenv("AI_CONFIRM_MIN_EMA_MOMENTUM", "0.0008")), env="AI_CONFIRM_MIN_EMA_MOMENTUM")
    AI_CONFIRM_MIN_ADX: float = Field(default=float(os.getenv("AI_CONFIRM_MIN_ADX", "8.0")), env="AI_CONFIRM_MIN_ADX")
    
    AI_CONFIRM_LOSS_5PCT_TA_SCORE: float = Field(default=float(os.getenv("AI_CONFIRM_LOSS_5PCT_TA_SCORE", "0.25")), env="AI_CONFIRM_LOSS_5PCT_TA_SCORE")
    AI_CONFIRM_LOSS_2PCT_TA_SCORE: float = Field(default=float(os.getenv("AI_CONFIRM_LOSS_2PCT_TA_SCORE", "0.2")), env="AI_CONFIRM_LOSS_2PCT_TA_SCORE")
    AI_CONFIRM_PROFIT_TA_SCORE: float = Field(default=float(os.getenv("AI_CONFIRM_PROFIT_TA_SCORE", "0.1")), env="AI_CONFIRM_PROFIT_TA_SCORE")
    
    AI_RISK_ASSESSMENT_ENABLED: bool = Field(default=parse_bool_env("AI_RISK_ASSESSMENT_ENABLED", "true"), env="AI_RISK_ASSESSMENT_ENABLED")
    AI_RISK_VOLATILITY_THRESHOLD: float = Field(default=float(os.getenv("AI_RISK_VOLATILITY_THRESHOLD", "0.015")), env="AI_RISK_VOLATILITY_THRESHOLD")
    AI_RISK_VOLUME_SPIKE_THRESHOLD: float = Field(default=float(os.getenv("AI_RISK_VOLUME_SPIKE_THRESHOLD", "2.9000000000000004")), env="AI_RISK_VOLUME_SPIKE_THRESHOLD")
    
    AI_MOMENTUM_CONFIDENCE_OVERRIDE: float = Field(default=float(os.getenv("AI_MOMENTUM_CONFIDENCE_OVERRIDE", "0.35")), env="AI_MOMENTUM_CONFIDENCE_OVERRIDE")
    AI_BOLLINGER_CONFIDENCE_OVERRIDE: float = Field(default=float(os.getenv("AI_BOLLINGER_CONFIDENCE_OVERRIDE", "0.35")), env="AI_BOLLINGER_CONFIDENCE_OVERRIDE")
    
    AI_TRACK_PERFORMANCE: bool = Field(default=parse_bool_env("AI_TRACK_PERFORMANCE", "true"), env="AI_TRACK_PERFORMANCE")
    AI_PERFORMANCE_LOG_PATH: str = Field(default="logs/ai_performance.jsonl", env="AI_PERFORMANCE_LOG_PATH")
    
    # ================================================================================
    # 🛠️ SYSTEM SETTINGS
    # ================================================================================
    RUNNING_IN_DOCKER: bool = Field(default=parse_bool_env('RUNNING_IN_DOCKER', 'false'), env='RUNNING_IN_DOCKER')
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, 
        extra="ignore"
    )
    
    # ================================================================================
    # 🔧 COMPUTED PROPERTIES (Kullanılmıyorsa veya stratejiler içinde hesaplanıyorsa kaldırılabilir)
    # ================================================================================
    # @property
    # def default_trade_amount_usdt(self) -> float:
    #     return self.INITIAL_CAPITAL_USDT * (self.MOMENTUM_BASE_POSITION_SIZE_PCT / 100.0)
    
    # @property
    # def timeframe_in_seconds(self) -> int:
    #     timeframe_map = {
    #         "1s": 1, "5s": 5, "15s": 15, "30s": 30,
    #         "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    #         "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    #         "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400
    #     }
    #     return timeframe_map.get(self.TIMEFRAME.lower(), 900)


# Global settings instance
settings: Final[Settings] = Settings()

# Dosya sonundaki os.getenv ile yapılan atamalar Settings sınıfı içine taşındığı için kaldırıldı.

if __name__ == "__main__":
    print("🔧 Configuration loaded via Pydantic model.")
    print(f"Symbol: {settings.SYMBOL}, Timeframe: {settings.TIMEFRAME}")
    print(f"AI Assistance Enabled: {settings.AI_ASSISTANCE_ENABLED}")
    print(f"AI Long Timeframe for TA: {settings.AI_TA_LONG_TIMEFRAME_STR}") # Güncellenmiş değeri gösterir
    print(f"Price Precision: {settings.PRICE_PRECISION}, Asset Precision: {settings.ASSET_PRECISION}") # Yeni eklenenler
    # Eğer Settings sınıfında print_summary veya validate_settings gibi metodlar varsa
    # ve bunlar logger kullanmıyorsa, burada çağırabilirsiniz.
    # Örneğin, bir önceki versiyondaki print_summary ve validate_settings metodları `print` kullandığı için
    # buraya eklenebilir veya doğrudan çağrılabilir.
    # if hasattr(settings, "validate_settings") and callable(getattr(settings, "validate_settings")):
    #     settings.validate_settings()
    # if hasattr(settings, "print_summary") and callable(getattr(settings, "print_summary")):
    #     settings.print_summary()

# else:
    # print("DEBUG: Config settings loaded via import.") # logger yerine print kullanılabilir veya bu satır tamamen kaldırılabilir.
    # logger importu kaldırıldığı için logger.debug() burada kullanılamaz.