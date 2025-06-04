# utils/config.py - Advanced Trading Bot Configuration

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Final
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables
env_path = Path(".") / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ .env file loaded: {env_path}")
else:
    print(f"ℹ️  .env file not found at {env_path}. Using environment variables or defaults.")

class Settings(BaseSettings):
    """🚀 Advanced Trading Bot Configuration - Optimized for Performance"""
    
    # ================================================================================
    # 🔐 API CREDENTIALS (Optional - for live trading)
    # ================================================================================
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None
    
    # ================================================================================
    # 📊 CORE TRADING SETTINGS
    # ================================================================================
    # Portfolio Management
    INITIAL_CAPITAL_USDT: float = 1000.0
    SYMBOL: str = "BTC/USDT"
    TIMEFRAME: str = "15m"
    
    # Fee Structure (Binance Spot)
    FEE_BUY: float = 0.001   # 0.1% buy fee
    FEE_SELL: float = 0.001  # 0.1% sell fee
    
    # Data Fetching
    OHLCV_LIMIT: int = 250  # Number of candles to fetch
    LOOP_SLEEP_SECONDS: int = 60  # Main loop delay
    LOOP_SLEEP_SECONDS_ON_DATA_ERROR: int = 120  # Error retry delay
    
    # Trading Limits
    MIN_TRADE_AMOUNT_USDT: float = 50.0  # Minimum trade amount
    
    # ================================================================================
    # 📊 DATA FETCHING CONFIGURATION
    # ================================================================================
    DATA_FETCHER_RETRY_ATTEMPTS: int = 3
    DATA_FETCHER_RETRY_MULTIPLIER: float = 1.0  # Exponential backoff multiplier
    DATA_FETCHER_RETRY_MIN_WAIT: float = 1.0    # Min wait time in seconds
    DATA_FETCHER_RETRY_MAX_WAIT: float = 10.0   # Max wait time in seconds
    DATA_FETCHER_TIMEOUT_SECONDS: int = 30
    LOOP_SLEEP_SECONDS: int = 5
    LOOP_SLEEP_SECONDS_ON_DATA_ERROR: int = 15
    
    # ================================================================================
    # 📝 LOGGING CONFIGURATION
    # ================================================================================
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = False
    TRADES_CSV_LOG_PATH: str = "logs/trades.csv"
    TRADES_JSONL_LOG_PATH: str = "logs/trades.jsonl"
    
    # ================================================================================
    # 🚀 MOMENTUM STRATEGY CONFIGURATION - COMPLETE
    # ================================================================================
    
    # === Core Technical Parameters ===
    MOMENTUM_EMA_SHORT: int = 9
    MOMENTUM_EMA_MEDIUM: int = 21
    MOMENTUM_EMA_LONG: int = 50
    MOMENTUM_RSI_PERIOD: int = 14
    MOMENTUM_ADX_PERIOD: int = 14
    MOMENTUM_ATR_PERIOD: int = 14
    MOMENTUM_VOLUME_SMA_PERIOD: int = 20
    
    # === Position Sizing - Dynamic ===
    MOMENTUM_BASE_POSITION_SIZE_PCT: float = 15.0
    MOMENTUM_MIN_POSITION_USDT: float = 200.0
    MOMENTUM_MAX_POSITION_USDT: float = 200.0  # Reduced from 500 to 200
    MOMENTUM_MAX_POSITIONS: int = 2
    MOMENTUM_MAX_TOTAL_EXPOSURE_PCT: float = 25.0  # Reduced from 40 to 25
    
    # === Position Sizing - Performance Based ===
    MOMENTUM_SIZE_HIGH_PROFIT_PCT: float = 18.0      # Reduced from 22.0
    MOMENTUM_SIZE_GOOD_PROFIT_PCT: float = 16.0      # Reduced from 18.0
    MOMENTUM_SIZE_NORMAL_PROFIT_PCT: float = 15.0    # Reduced from 16.0
    MOMENTUM_SIZE_BREAKEVEN_PCT: float = 14.0        # Around breakeven
    MOMENTUM_SIZE_LOSS_PCT: float = 12.0             # In loss
    MOMENTUM_SIZE_MIN_USD: float = 200.0
    MOMENTUM_SIZE_MAX_USD: float = 200.0             # Reduced from 500 to 200
    MOMENTUM_SIZE_MAX_BALANCE_PCT: float = 19.0
    
    # === Performance Thresholds ===
    MOMENTUM_PERF_HIGH_PROFIT_THRESHOLD: float = 0.10    # 10%
    MOMENTUM_PERF_GOOD_PROFIT_THRESHOLD: float = 0.05    # 5%
    MOMENTUM_PERF_NORMAL_PROFIT_THRESHOLD: float = 0.02  # 2%
    MOMENTUM_PERF_BREAKEVEN_THRESHOLD: float = -0.02     # -2%
    
    # === Risk Management ===
    MOMENTUM_MAX_LOSS_PCT: float = 0.008                 # 0.8% max loss per trade
    MOMENTUM_MIN_PROFIT_TARGET_USDT: float = 1.50        # $1.50 minimum profit target
    MOMENTUM_QUICK_PROFIT_THRESHOLD_USDT: float = 0.75   # $0.75 quick profit
    
    # === Timing Controls ===
    MOMENTUM_MAX_HOLD_MINUTES: int = 60                  # 60 minutes max hold time
    MOMENTUM_BREAKEVEN_MINUTES: int = 3                  # Move to breakeven after 3min
    MOMENTUM_MIN_TIME_BETWEEN_TRADES_SEC: int = 30       # 30 seconds between trades

    # === Buy Conditions ===
    MOMENTUM_BUY_MIN_QUALITY_SCORE: int = 3
    
    # EMA Spreads
    MOMENTUM_BUY_MIN_EMA_SPREAD_1: float = 0.0001
    MOMENTUM_BUY_MIN_EMA_SPREAD_2: float = 0.0002
    
    # EMA Momentum Levels
    MOMENTUM_BUY_EMA_MOM_EXCELLENT: float = 0.001    # 4 points
    MOMENTUM_BUY_EMA_MOM_GOOD: float = 0.0005        # 3 points
    MOMENTUM_BUY_EMA_MOM_DECENT: float = 0.0002      # 2 points
    MOMENTUM_BUY_EMA_MOM_MIN: float = 0.0001         # 1 point
    
    # RSI Ranges
    MOMENTUM_BUY_RSI_EXCELLENT_MIN: float = 20.0     # 3 points
    MOMENTUM_BUY_RSI_EXCELLENT_MAX: float = 80.0     # 3 points
    MOMENTUM_BUY_RSI_GOOD_MIN: float = 10.0          # 2 points
    MOMENTUM_BUY_RSI_GOOD_MAX: float = 90.0          # 2 points
    MOMENTUM_BUY_RSI_EXTREME_MIN: float = 5.0        # Rejection threshold
    MOMENTUM_BUY_RSI_EXTREME_MAX: float = 95.0       # Rejection threshold
    
    # ADX Levels
    MOMENTUM_BUY_ADX_EXCELLENT: float = 15.0         # 3 points
    MOMENTUM_BUY_ADX_GOOD: float = 10.0              # 2 points
    MOMENTUM_BUY_ADX_DECENT: float = 5.0             # 1 point
    
    # Volume Levels
    MOMENTUM_BUY_VOLUME_EXCELLENT: float = 1.2       # 3 points
    MOMENTUM_BUY_VOLUME_GOOD: float = 1.0            # 2 points
    MOMENTUM_BUY_VOLUME_DECENT: float = 0.8          # 1 point
    
    # Price Momentum Levels
    MOMENTUM_BUY_PRICE_MOM_EXCELLENT: float = 0.0005 # 3 points
    MOMENTUM_BUY_PRICE_MOM_GOOD: float = 0.0001      # 2 points
    MOMENTUM_BUY_PRICE_MOM_DECENT: float = -0.0005   # 1 point (even negative!)
    
    # === Sell Conditions - Phase System ===
    MOMENTUM_SELL_MIN_HOLD_MINUTES: int = 25
    MOMENTUM_SELL_CATASTROPHIC_LOSS_PCT: float = -0.03  # -3% emergency exit
    
    # Premium Profit Levels (time independent)
    MOMENTUM_SELL_PREMIUM_EXCELLENT: float = 5.0    # $5+ profit
    MOMENTUM_SELL_PREMIUM_GREAT: float = 3.0        # $3+ profit
    MOMENTUM_SELL_PREMIUM_GOOD: float = 2.0         # $2+ profit
    
    # Phase 1 (25-60 minutes)
    MOMENTUM_SELL_PHASE1_EXCELLENT: float = 1.5     # $1.5 profit
    MOMENTUM_SELL_PHASE1_GOOD: float = 1.0          # $1.0 profit
    MOMENTUM_SELL_PHASE1_LOSS_PROTECTION: float = -2.0  # -$2 loss
    
    # Phase 2 (60-120 minutes)
    MOMENTUM_SELL_PHASE2_EXCELLENT: float = 1.0     # $1.0 profit
    MOMENTUM_SELL_PHASE2_GOOD: float = 0.75         # $0.75 profit
    MOMENTUM_SELL_PHASE2_DECENT: float = 0.50       # $0.50 profit (90min+)
    MOMENTUM_SELL_PHASE2_LOSS_PROTECTION: float = -1.5  # -$1.5 loss
    
    # Phase 3 (120-180 minutes)
    MOMENTUM_SELL_PHASE3_EXCELLENT: float = 0.75    # $0.75 profit
    MOMENTUM_SELL_PHASE3_GOOD: float = 0.50         # $0.50 profit
    MOMENTUM_SELL_PHASE3_DECENT: float = 0.25       # $0.25 profit
    MOMENTUM_SELL_PHASE3_BREAKEVEN_MIN: float = -0.20   # Breakeven protection
    MOMENTUM_SELL_PHASE3_BREAKEVEN_MAX: float = 0.20    # Breakeven protection
    MOMENTUM_SELL_PHASE3_LOSS_PROTECTION: float = -1.0  # -$1 loss
    
    # Phase 4 (180+ minutes)
    MOMENTUM_SELL_PHASE4_EXCELLENT: float = 0.50    # $0.50 profit
    MOMENTUM_SELL_PHASE4_GOOD: float = 0.25         # $0.25 profit
    MOMENTUM_SELL_PHASE4_MINIMAL: float = 0.10      # $0.10 profit
    MOMENTUM_SELL_PHASE4_BREAKEVEN_MIN: float = -0.30   # Wide breakeven
    MOMENTUM_SELL_PHASE4_BREAKEVEN_MAX: float = 0.30    # Wide breakeven
    MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES: int = 240  # 4 hours force exit
    
    # Loss Protection
    MOMENTUM_SELL_LOSS_MULTIPLIER: float = 4.0      # Trading cost multiplier
    
    # Technical Exit Conditions
    MOMENTUM_SELL_TECH_MIN_MINUTES: int = 90        # Technical exit min time
    MOMENTUM_SELL_TECH_MIN_LOSS: float = -1.5       # Technical exit min loss
    MOMENTUM_SELL_TECH_RSI_EXTREME: float = 12.0    # RSI oversold extreme
    
    # === Timing Controls - Wait Times ===
    MOMENTUM_WAIT_PROFIT_5PCT: int = 300             # 5min wait when 5%+ profit
    MOMENTUM_WAIT_PROFIT_2PCT: int = 450             # 7.5min wait when 2%+ profit
    MOMENTUM_WAIT_BREAKEVEN: int = 600               # 10min wait around breakeven
    MOMENTUM_WAIT_LOSS: int = 900                    # 15min wait when in loss

    # ================================================================================
    # 🎯 BOLLINGER RSI STRATEGY CONFIGURATION
    # ================================================================================
    
    # === Core Technical Parameters ===
    BOLLINGER_RSI_BB_PERIOD: int = 20                       # Bollinger Bands period
    BOLLINGER_RSI_BB_STD_DEV: float = 2.0                   # Bollinger Bands std deviation
    BOLLINGER_RSI_RSI_PERIOD: int = 14                      # RSI calculation period
    BOLLINGER_RSI_VOLUME_SMA_PERIOD: int = 20               # Volume SMA period
    
    # === Position Sizing ===
    BOLLINGER_RSI_BASE_POSITION_SIZE_PCT: float = 6.0       # 6% of portfolio per trade
    BOLLINGER_RSI_MAX_POSITION_USDT: float = 150.0          # Reduced from 200 to 150
    BOLLINGER_RSI_MIN_POSITION_USDT: float = 100.0          # Min $100 per position
    BOLLINGER_RSI_MAX_POSITIONS: int = 2                    # Max 2 simultaneous positions
    BOLLINGER_RSI_MAX_TOTAL_EXPOSURE_PCT: float = 15.0      # Max 15% total exposure
    
    # === Risk Management ===
    BOLLINGER_RSI_MAX_LOSS_PCT: float = 0.006               # 0.6% max loss per trade
    BOLLINGER_RSI_MIN_PROFIT_TARGET_USDT: float = 1.20      # $1.20 minimum profit target
    BOLLINGER_RSI_QUICK_PROFIT_THRESHOLD_USDT: float = 0.60  # $0.60 quick profit
    
    # === Timing Controls ===
    BOLLINGER_RSI_MAX_HOLD_MINUTES: int = 45                # 45 minutes max hold time
    BOLLINGER_RSI_BREAKEVEN_MINUTES: int = 5                # Move to breakeven after 5min
    BOLLINGER_RSI_MIN_TIME_BETWEEN_TRADES_SEC: int = 45     # 45 seconds between trades
    
    # ================================================================================
    # 🛡️ GLOBAL RISK MANAGEMENT CONFIGURATION
    # ================================================================================
    GLOBAL_MAX_POSITION_SIZE_PCT: float = 20.0              # Max 20% per single position
    GLOBAL_MAX_OPEN_POSITIONS: int = 8                      # Max 8 total open positions
    GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT: float = 0.15         # Max 15% portfolio drawdown
    GLOBAL_MAX_DAILY_LOSS_PCT: float = 0.05                 # Max 5% daily loss
    
    # Additional risk settings
    DRAWDOWN_LIMIT_HIGH_VOL_REGIME_PCT: Optional[float] = 0.15  # 15% in high volatility
    
    # ================================================================================
    # 🤖 ADVANCED AI ASSISTANCE CONFIGURATION - ENHANCED
    # ================================================================================
    AI_ASSISTANCE_ENABLED: bool = True
    AI_OPERATION_MODE: str = "technical_analysis"  # "technical_analysis" or "ml_model"
    AI_CONFIDENCE_THRESHOLD: float = 0.3
    
    # AI Model Settings
    AI_MODEL_PATH: Optional[str] = None
    
    # AI Technical Analysis Settings - ENHANCED
    AI_TA_EMA_PERIODS_MAIN_TF: tuple = (9, 21, 50)
    AI_TA_EMA_PERIODS_LONG_TF: tuple = (18, 63, 150)
    AI_TA_RSI_PERIOD: int = 14
    AI_TA_DIVERGENCE_LOOKBACK: int = 10
    AI_TA_LONG_TIMEFRAME_STR: str = "1m"
    
    # AI Technical Analysis Weights (must sum to 1.0)
    AI_TA_WEIGHT_TREND_MAIN: float = 0.4
    AI_TA_WEIGHT_TREND_LONG: float = 0.3
    AI_TA_WEIGHT_VOLUME: float = 0.2
    AI_TA_WEIGHT_DIVERGENCE: float = 0.1
    
    # AI Standalone Signal Thresholds
    AI_TA_STANDALONE_THRESH_STRONG_BUY: float = 0.7
    AI_TA_STANDALONE_THRESH_BUY: float = 0.4
    AI_TA_STANDALONE_THRESH_SELL: float = -0.4
    AI_TA_STANDALONE_THRESH_STRONG_SELL: float = -0.7
    
    # AI Confirmation Thresholds
    AI_CONFIRM_MIN_TA_SCORE: float = 0.3
    AI_CONFIRM_MIN_QUALITY_SCORE: int = 3
    AI_CONFIRM_MIN_EMA_SPREAD_1: float = 0.0003
    AI_CONFIRM_MIN_EMA_SPREAD_2: float = 0.0005
    AI_CONFIRM_MIN_VOLUME_RATIO: float = 1.0
    AI_CONFIRM_MIN_PRICE_MOMENTUM: float = 0.0005
    AI_CONFIRM_MIN_EMA_MOMENTUM: float = 0.0008
    AI_CONFIRM_MIN_ADX: float = 12.0
    
    # Portfolio-based dynamic thresholds
    AI_CONFIRM_LOSS_5PCT_TA_SCORE: float = 0.4
    AI_CONFIRM_LOSS_2PCT_TA_SCORE: float = 0.35
    AI_CONFIRM_PROFIT_TA_SCORE: float = 0.3
    
    # AI Risk Assessment
    AI_RISK_ASSESSMENT_ENABLED: bool = True
    AI_RISK_VOLATILITY_THRESHOLD: float = 0.02
    AI_RISK_VOLUME_SPIKE_THRESHOLD: float = 2.0
    
    # Strategy-Specific AI Settings
    AI_MOMENTUM_CONFIDENCE_OVERRIDE: float = 0.25
    AI_BOLLINGER_CONFIDENCE_OVERRIDE: float = 0.35
    
    # AI Performance Tracking
    AI_TRACK_PERFORMANCE: bool = True
    AI_PERFORMANCE_LOG_PATH: str = "logs/ai_performance.jsonl"
    
    # ================================================================================
    # 🛠️ SYSTEM SETTINGS
    # ================================================================================
    # Environment Detection
    RUNNING_IN_DOCKER: bool = os.getenv('RUNNING_IN_DOCKER', 'false').lower() == 'true'
    
    # Pydantic Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore unknown environment variables
    )
    
    # ================================================================================
    # 🔧 COMPUTED PROPERTIES
    # ================================================================================
    
    @property
    def trade_amount_usdt(self) -> float:
        """Calculate default trade amount based on portfolio size"""
        return self.INITIAL_CAPITAL_USDT * (self.MOMENTUM_BASE_POSITION_SIZE_PCT / 100.0)
    
    @property
    def min_trade_amount_usdt(self) -> float:
        """Minimum trade amount"""
        return self.MIN_TRADE_AMOUNT_USDT

    @property
    def timeframe_seconds(self) -> int:
        """Convert timeframe to seconds"""
        timeframe_map = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
            "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400
        }
        return timeframe_map.get(self.TIMEFRAME, 900)  # Default to 15m
    
    @property
    def momentum_stop_loss_pct(self) -> float:
        """Dynamic stop-loss percentage based on position size"""
        # Tighter stop-loss for larger positions
        if self.trade_amount_usdt >= 200:
            return 0.006  # 0.6% for large positions
        elif self.trade_amount_usdt >= 100:
            return 0.008  # 0.8% for medium positions  
        else:
            return 0.010  # 1.0% for small positions
    
    @property
    def bollinger_rsi_stop_loss_pct(self) -> float:
        """Dynamic stop-loss percentage for Bollinger RSI strategy"""
        # Tighter stop-loss for mean reversion (smaller positions)
        trade_amount = self.INITIAL_CAPITAL_USDT * (self.BOLLINGER_RSI_BASE_POSITION_SIZE_PCT / 100.0)
        if trade_amount >= 150:
            return 0.004  # 0.4% for larger positions
        elif trade_amount >= 100:
            return 0.006  # 0.6% for medium positions  
        else:
            return 0.008  # 0.8% for smaller positions
    
    @property
    def bollinger_rsi_trade_amount_usdt(self) -> float:
        """Calculate Bollinger RSI trade amount based on portfolio size"""
        return self.INITIAL_CAPITAL_USDT * (self.BOLLINGER_RSI_BASE_POSITION_SIZE_PCT / 100.0)
    
    # ================================================================================
    # 🚀 PERFORMANCE OPTIMIZATIONS
    # ================================================================================
    
    # Ultra-Fast Profit Taking (for larger positions)
    MOMENTUM_ULTRA_FAST_PROFIT_SECONDS: int = 30      # Take profit after 30 seconds
    MOMENTUM_ULTRA_FAST_PROFIT_USD: float = 0.50      # If profit >= $0.50
    
    MOMENTUM_QUICK_PROFIT_MINUTES: int = 1             # Take profit after 1 minute
    MOMENTUM_QUICK_PROFIT_USD: float = 0.75            # If profit >= $0.75
    
    MOMENTUM_TARGET_PROFIT_MINUTES: int = 5            # Take profit after 5 minutes
    MOMENTUM_TARGET_PROFIT_USD: float = 1.50           # If profit >= $1.50
    
    # Enhanced Loss Protection
    MOMENTUM_EARLY_PROTECTION_MINUTES: int = 2        # Early protection after 2 min
    MOMENTUM_EARLY_PROTECTION_FACTOR: float = 0.5     # 50% of max loss
    
    MOMENTUM_PROGRESSIVE_PROTECTION_MINUTES: int = 10 # Progressive protection after 10 min
    MOMENTUM_PROGRESSIVE_PROTECTION_FACTOR: float = 0.75  # 75% of max loss
    
    # Breakeven Protection
    MOMENTUM_BREAKEVEN_BUFFER_PCT: float = 0.0005     # 0.05% buffer below breakeven
    MOMENTUM_PROFIT_LOCK_THRESHOLD_USD: float = 0.75  # Lock profit if >= $0.75
    MOMENTUM_PROFIT_LOCK_AMOUNT_PCT: float = 0.0005   # Lock 0.05% profit
    
    # ================================================================================
    # 📊 VALIDATION & HEALTH CHECKS
    # ================================================================================
    
    def validate_settings(self) -> bool:
        """Validate configuration for consistency"""
        errors = []
        
        # Position sizing validation
        if self.MOMENTUM_MIN_POSITION_USDT > self.MOMENTUM_MAX_POSITION_USDT:
            errors.append("Momentum: Min position size cannot be greater than max position size")
            
        if self.BOLLINGER_RSI_MIN_POSITION_USDT > self.BOLLINGER_RSI_MAX_POSITION_USDT:
            errors.append("BollingerRSI: Min position size cannot be greater than max position size")
        
        # Exposure validation for Momentum
        max_momentum_exposure = (self.MOMENTUM_MAX_POSITION_USDT / self.INITIAL_CAPITAL_USDT) * 100
        if max_momentum_exposure > self.GLOBAL_MAX_POSITION_SIZE_PCT:
            errors.append(f"Momentum max position size ({max_momentum_exposure:.1f}%) exceeds global limit ({self.GLOBAL_MAX_POSITION_SIZE_PCT}%)")
        
        # Exposure validation for BollingerRSI
        max_bollinger_exposure = (self.BOLLINGER_RSI_MAX_POSITION_USDT / self.INITIAL_CAPITAL_USDT) * 100
        if max_bollinger_exposure > self.GLOBAL_MAX_POSITION_SIZE_PCT:
            errors.append(f"BollingerRSI max position size ({max_bollinger_exposure:.1f}%) exceeds global limit ({self.GLOBAL_MAX_POSITION_SIZE_PCT}%)")
        
        # Total exposure validation
        total_max_exposure = self.MOMENTUM_MAX_TOTAL_EXPOSURE_PCT + self.BOLLINGER_RSI_MAX_TOTAL_EXPOSURE_PCT
        if total_max_exposure > 50.0:  # Reasonable total limit
            errors.append(f"Total max exposure ({total_max_exposure:.1f}%) too high (>50%)")
        
        # Risk validation - FIX: Use correct attribute names
        if self.MOMENTUM_MAX_LOSS_PCT > 0.02:  # 2%
            errors.append("Momentum max loss per trade too high (>2%)")
            
        if self.BOLLINGER_RSI_MAX_LOSS_PCT > 0.02:  # 2%
            errors.append("BollingerRSI max loss per trade too high (>2%)")
        
        # Timing validation - FIX: Use correct attribute names
        if self.MOMENTUM_BREAKEVEN_MINUTES >= self.MOMENTUM_MAX_HOLD_MINUTES:
            errors.append("Momentum: Breakeven time cannot be >= max hold time")
            
        if self.BOLLINGER_RSI_BREAKEVEN_MINUTES >= self.BOLLINGER_RSI_MAX_HOLD_MINUTES:
            errors.append("BollingerRSI: Breakeven time cannot be >= max hold time")
        
        # Position limits validation
        total_max_positions = self.MOMENTUM_MAX_POSITIONS + self.BOLLINGER_RSI_MAX_POSITIONS
        if total_max_positions > self.GLOBAL_MAX_OPEN_POSITIONS:
            errors.append(f"Total max positions ({total_max_positions}) exceeds global limit ({self.GLOBAL_MAX_OPEN_POSITIONS})")
        
        if errors:
            for error in errors:
                print(f"❌ Config Error: {error}")
            return False
        
        print("✅ Configuration validation passed")
        return True

    def print_summary(self) -> None:
        """Print configuration summary"""
        print("\n" + "="*60)
        print("🚀 DUAL-STRATEGY TRADING BOT CONFIGURATION")
        print("="*60)
        print(f"📊 Symbol: {self.SYMBOL} | Timeframe: {self.TIMEFRAME}")
        print(f"💰 Initial Capital: ${self.INITIAL_CAPITAL_USDT:,.2f}")
        print("")
        print("🎯 MOMENTUM STRATEGY:")
        print(f"   Position Size: {self.MOMENTUM_BASE_POSITION_SIZE_PCT}% (${self.trade_amount_usdt:.2f})")
        print(f"   Position Range: ${self.MOMENTUM_MIN_POSITION_USDT} - ${self.MOMENTUM_MAX_POSITION_USDT}")
        print(f"   Max Positions: {self.MOMENTUM_MAX_POSITIONS} | Max Exposure: {self.MOMENTUM_MAX_TOTAL_EXPOSURE_PCT}%")
        print(f"   Max Loss: {self.MOMENTUM_MAX_LOSS_PCT*100:.2f}% | Hold Time: {self.MOMENTUM_MAX_HOLD_MINUTES}min")
        print("")
        print("📊 BOLLINGER RSI STRATEGY:")
        print(f"   Position Size: {self.BOLLINGER_RSI_BASE_POSITION_SIZE_PCT}% (${self.bollinger_rsi_trade_amount_usdt:.2f})")
        print(f"   Position Range: ${self.BOLLINGER_RSI_MIN_POSITION_USDT} - ${self.BOLLINGER_RSI_MAX_POSITION_USDT}")
        print(f"   Max Positions: {self.BOLLINGER_RSI_MAX_POSITIONS} | Max Exposure: {self.BOLLINGER_RSI_MAX_TOTAL_EXPOSURE_PCT}%")
        print(f"   Max Loss: {self.BOLLINGER_RSI_MAX_LOSS_PCT*100:.2f}% | Hold Time: {self.BOLLINGER_RSI_MAX_HOLD_MINUTES}min")
        print("")
        print("🛡️  GLOBAL RISK LIMITS:")
        print(f"   Max Single Position: {self.GLOBAL_MAX_POSITION_SIZE_PCT}%")
        print(f"   Max Open Positions: {self.GLOBAL_MAX_OPEN_POSITIONS}")
        print(f"   Max Portfolio Drawdown: {self.GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT*100:.1f}%")
        print(f"   Max Daily Loss: {self.GLOBAL_MAX_DAILY_LOSS_PCT*100:.2f}%")
        print("")
        print("🤖 AI ASSISTANT:")
        if self.AI_ASSISTANCE_ENABLED:
            print(f"   Status: ENABLED | Mode: {self.AI_OPERATION_MODE.upper()}")
            print(f"   Confidence Threshold: {self.AI_CONFIDENCE_THRESHOLD}")
            print(f"   Momentum Override: {self.AI_MOMENTUM_CONFIDENCE_OVERRIDE}")
            print(f"   BollingerRSI Override: {self.AI_BOLLINGER_CONFIDENCE_OVERRIDE}")
            print(f"   Risk Assessment: {'Enabled' if self.AI_RISK_ASSESSMENT_ENABLED else 'Disabled'}")
            print(f"   Performance Tracking: {'Enabled' if self.AI_TRACK_PERFORMANCE else 'Disabled'}")
        else:
            print(f"   Status: DISABLED")
        print("")
        print(f"📝 Log Level: {self.LOG_LEVEL}")
        print("="*60)
        
        # Summary calculations
        total_exposure = self.MOMENTUM_MAX_TOTAL_EXPOSURE_PCT + self.BOLLINGER_RSI_MAX_TOTAL_EXPOSURE_PCT
        total_positions = self.MOMENTUM_MAX_POSITIONS + self.BOLLINGER_RSI_MAX_POSITIONS
        print(f"\n📈 COMBINED STRATEGY LIMITS:")
        print(f"   Total Max Exposure: {total_exposure}%")
        print(f"   Total Max Positions: {total_positions}")
        print(f"   Portfolio Utilization: {total_exposure}/{self.GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT*100:.1f}% = {(total_exposure/(self.GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT*100))*100:.1f}%")
        if self.AI_ASSISTANCE_ENABLED:
            print(f"   AI Enhancement: Active ({self.AI_OPERATION_MODE})")

# Create global settings instance
settings: Final[Settings] = Settings()

# Auto-validate on import
if __name__ == "__main__":
    print("🔧 Configuration loaded and validated!")
    settings.print_summary()
    settings.validate_settings()
else:
    # Quick validation on import
    settings.validate_settings()

# ==============================================
# 🤖 AI SIGNAL PROVIDER SETTINGS - HEPSİ BURADAN!
# ==============================================

# AI Ana Ayarları - FIXED BOOLEAN PARSING
def parse_bool_env(env_var: str, default: str) -> bool:
    """Parse boolean from environment variable safely"""
    value = os.getenv(env_var, default).lower()
    return value in ('true', '1', 'yes', 'on', 'enabled')

AI_ASSISTANCE_ENABLED = parse_bool_env("AI_ASSISTANCE_ENABLED", "true")  # Fixed parsing
AI_OPERATION_MODE = os.getenv("AI_OPERATION_MODE", "technical_analysis")  # "technical_analysis" veya "ml_model"
AI_MODEL_PATH = os.getenv("AI_MODEL_PATH")  # ML model dosya yolu (opsiyonel)
AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.3"))  # 0.6 → 0.3 (KOLAY!)

# AI Risk Assessment - FIXED BOOLEAN PARSING
AI_RISK_ASSESSMENT_ENABLED = parse_bool_env("AI_RISK_ASSESSMENT_ENABLED", "true")
AI_RISK_VOLATILITY_THRESHOLD = float(os.getenv("AI_RISK_VOLATILITY_THRESHOLD", "0.02"))
AI_RISK_VOLUME_SPIKE_THRESHOLD = float(os.getenv("AI_RISK_VOLUME_SPIKE_THRESHOLD", "2.0"))

# AI Performance Tracking - FIXED BOOLEAN PARSING
AI_TRACK_PERFORMANCE = parse_bool_env("AI_TRACK_PERFORMANCE", "true")
AI_PERFORMANCE_LOG_PATH = os.getenv("AI_PERFORMANCE_LOG_PATH", "logs/ai_performance.jsonl")

# AI Strategy Confidence Overrides (Strateji bazlı AI eşikleri)
AI_MOMENTUM_CONFIDENCE_OVERRIDE = float(os.getenv("AI_MOMENTUM_CONFIDENCE_OVERRIDE", "0.25"))  # Momentum için çok kolay
AI_BOLLINGER_CONFIDENCE_OVERRIDE = float(os.getenv("AI_BOLLINGER_CONFIDENCE_OVERRIDE", "0.35"))  # Bollinger için biraz sıkı

# ==============================================
# 🔧 AI TECHNICAL ANALYSIS PARAMETERS - FİNE TUNING!
# ==============================================

# AI TA - EMA Periods (Çoklu zaman dilimi)
AI_TA_EMA_PERIODS_MAIN_TF = tuple(map(int, os.getenv("AI_TA_EMA_PERIODS_MAIN", "9,21,50").split(",")))  # Ana zaman dilimi
AI_TA_EMA_PERIODS_LONG_TF = tuple(map(int, os.getenv("AI_TA_EMA_PERIODS_LONG", "18,63,150").split(",")))  # Uzun zaman dilimi

# AI TA - Diğer indikatör ayarları  
AI_TA_RSI_PERIOD = int(os.getenv("AI_TA_RSI_PERIOD", "14"))
AI_TA_DIVERGENCE_LOOKBACK = int(os.getenv("AI_TA_DIVERGENCE_LOOKBACK", "10"))
AI_TA_LONG_TIMEFRAME_STR = os.getenv("AI_TA_LONG_TIMEFRAME", "1m")  # 5s → 1m resample

# AI TA - Ağırlıklar (toplam 1.0 olmalı)
AI_TA_WEIGHT_TREND_MAIN = float(os.getenv("AI_TA_WEIGHT_TREND_MAIN", "0.4"))     # Ana trend ağırlığı artırıldı
AI_TA_WEIGHT_TREND_LONG = float(os.getenv("AI_TA_WEIGHT_TREND_LONG", "0.3"))     # Uzun trend önemli
AI_TA_WEIGHT_VOLUME = float(os.getenv("AI_TA_WEIGHT_VOLUME", "0.2"))             # Volume azaltıldı
AI_TA_WEIGHT_DIVERGENCE = float(os.getenv("AI_TA_WEIGHT_DIVERGENCE", "0.1"))     # Divergence azaltıldı

# AI TA - Standalone Signal Thresholds
AI_TA_STANDALONE_THRESH_STRONG_BUY = float(os.getenv("AI_TA_STANDALONE_STRONG_BUY", "0.7"))
AI_TA_STANDALONE_THRESH_BUY = float(os.getenv("AI_TA_STANDALONE_BUY", "0.4"))
AI_TA_STANDALONE_THRESH_SELL = float(os.getenv("AI_TA_STANDALONE_SELL", "-0.4"))
AI_TA_STANDALONE_THRESH_STRONG_SELL = float(os.getenv("AI_TA_STANDALONE_STRONG_SELL", "-0.7"))

# ==============================================
# 🎯 AI CONFIRMATION THRESHOLDS - MAKUL DEĞERLER!
# ==============================================

# AI onay için gerekli minimum değerler
AI_CONFIRM_MIN_TA_SCORE = float(os.getenv("AI_CONFIRM_MIN_TA_SCORE", "0.3"))                # TA skoru minimum
AI_CONFIRM_MIN_QUALITY_SCORE = int(os.getenv("AI_CONFIRM_MIN_QUALITY_SCORE", "3"))          # Quality minimum  
AI_CONFIRM_MIN_EMA_SPREAD_1 = float(os.getenv("AI_CONFIRM_MIN_EMA_SPREAD_1", "0.0003"))     # EMA spread 1
AI_CONFIRM_MIN_EMA_SPREAD_2 = float(os.getenv("AI_CONFIRM_MIN_EMA_SPREAD_2", "0.0005"))     # EMA spread 2
AI_CONFIRM_MIN_VOLUME_RATIO = float(os.getenv("AI_CONFIRM_MIN_VOLUME_RATIO", "1.0"))        # Volume ratio
AI_CONFIRM_MIN_PRICE_MOMENTUM = float(os.getenv("AI_CONFIRM_MIN_PRICE_MOMENTUM", "0.0005")) # Price momentum
AI_CONFIRM_MIN_EMA_MOMENTUM = float(os.getenv("AI_CONFIRM_MIN_EMA_MOMENTUM", "0.0008"))     # EMA momentum
AI_CONFIRM_MIN_ADX = float(os.getenv("AI_CONFIRM_MIN_ADX", "12"))                           # ADX minimum

# Portfolio durumuna göre dinamik eşikler
AI_CONFIRM_LOSS_5PCT_TA_SCORE = float(os.getenv("AI_CONFIRM_LOSS_5PCT_TA_SCORE", "0.4"))   # %5+ zarar varsa
AI_CONFIRM_LOSS_2PCT_TA_SCORE = float(os.getenv("AI_CONFIRM_LOSS_2PCT_TA_SCORE", "0.35"))  # %2+ zarar varsa
AI_CONFIRM_PROFIT_TA_SCORE = float(os.getenv("AI_CONFIRM_PROFIT_TA_SCORE", "0.3"))         # Kar varsa

# ==============================================
# 🚀 MOMENTUM STRATEGY SETTINGS - KOLAY YÖNETİM!
# ==============================================

# Momentum - Technical Parameters  
MOMENTUM_EMA_SHORT = int(os.getenv("MOMENTUM_EMA_SHORT", "9"))      # Kısa EMA
MOMENTUM_EMA_MEDIUM = int(os.getenv("MOMENTUM_EMA_MEDIUM", "21"))   # Orta EMA  
MOMENTUM_EMA_LONG = int(os.getenv("MOMENTUM_EMA_LONG", "50"))       # Uzun EMA
MOMENTUM_RSI_PERIOD = int(os.getenv("MOMENTUM_RSI_PERIOD", "14"))   # RSI periyodu
MOMENTUM_ADX_PERIOD = int(os.getenv("MOMENTUM_ADX_PERIOD", "14"))   # ADX periyodu
MOMENTUM_ATR_PERIOD = int(os.getenv("MOMENTUM_ATR_PERIOD", "14"))   # ATR periyodu
MOMENTUM_VOLUME_SMA_PERIOD = int(os.getenv("MOMENTUM_VOLUME_SMA_PERIOD", "20"))  # Volume SMA

# Momentum - Position Management
MOMENTUM_MAX_POSITIONS = int(os.getenv("MOMENTUM_MAX_POSITIONS", "2"))                    # Max pozisyon sayısı
MOMENTUM_BASE_POSITION_SIZE_PCT = float(os.getenv("MOMENTUM_BASE_POSITION_SIZE_PCT", "15"))  # Base pozisyon %
MOMENTUM_MIN_POSITION_USDT = float(os.getenv("MOMENTUM_MIN_POSITION_USDT", "200"))       # Min pozisyon
MOMENTUM_MAX_POSITION_USDT = float(os.getenv("MOMENTUM_MAX_POSITION_USDT", "500"))       # Max pozisyon
MOMENTUM_MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("MOMENTUM_MAX_TOTAL_EXPOSURE_PCT", "40"))  # Max total exposure

# Momentum - Risk Management  
MOMENTUM_MAX_LOSS_PCT = float(os.getenv("MOMENTUM_MAX_LOSS_PCT", "0.008"))                    # 0.8% (fixed decimal)
MOMENTUM_MIN_PROFIT_TARGET_USDT = float(os.getenv("MOMENTUM_MIN_PROFIT_TARGET_USDT", "1.0"))  # Min kar hedefi
MOMENTUM_QUICK_PROFIT_THRESHOLD_USDT = float(os.getenv("MOMENTUM_QUICK_PROFIT_THRESHOLD_USDT", "2.0"))  # Hızlı kar

# Momentum - Timing Settings
MOMENTUM_MAX_HOLD_MINUTES = int(os.getenv("MOMENTUM_MAX_HOLD_MINUTES", "240"))           # Max tutma süresi (4 saat)
MOMENTUM_BREAKEVEN_MINUTES = int(os.getenv("MOMENTUM_BREAKEVEN_MINUTES", "120"))        # Kırılma noktası süresi

# ==============================================
# 🎯 MOMENTUM BUY CONDITIONS - ULTRA KOLAY AYARLAR!
# ==============================================

# Momentum Buy - Quality Score  
MOMENTUM_BUY_MIN_QUALITY_SCORE = int(os.getenv("MOMENTUM_BUY_MIN_QUALITY_SCORE", "3"))        # 4 → 3 (KOLAY!)

# Momentum Buy - EMA Koşulları (ULTRA KOLAY!)
MOMENTUM_BUY_MIN_EMA_SPREAD_1 = float(os.getenv("MOMENTUM_BUY_MIN_EMA_SPREAD_1", "0.0001"))   # %0.01 (çok kolay)
MOMENTUM_BUY_MIN_EMA_SPREAD_2 = float(os.getenv("MOMENTUM_BUY_MIN_EMA_SPREAD_2", "0.0002"))   # %0.02 (çok kolay)

# Momentum Buy - EMA Momentum Levels
MOMENTUM_BUY_EMA_MOM_EXCELLENT = float(os.getenv("MOMENTUM_BUY_EMA_MOM_EXCELLENT", "0.001"))    # 4 puan
MOMENTUM_BUY_EMA_MOM_GOOD = float(os.getenv("MOMENTUM_BUY_EMA_MOM_GOOD", "0.0005"))            # 3 puan
MOMENTUM_BUY_EMA_MOM_DECENT = float(os.getenv("MOMENTUM_BUY_EMA_MOM_DECENT", "0.0002"))        # 2 puan
MOMENTUM_BUY_EMA_MOM_MIN = float(os.getenv("MOMENTUM_BUY_EMA_MOM_MIN", "0.0001"))              # 1 puan (minimum)

# Momentum Buy - RSI Ranges (ULTRA GENİŞ!)
MOMENTUM_BUY_RSI_EXCELLENT_MIN = float(os.getenv("MOMENTUM_BUY_RSI_EXCELLENT_MIN", "20"))       # 3 puan
MOMENTUM_BUY_RSI_EXCELLENT_MAX = float(os.getenv("MOMENTUM_BUY_RSI_EXCELLENT_MAX", "80"))       # 3 puan
MOMENTUM_BUY_RSI_GOOD_MIN = float(os.getenv("MOMENTUM_BUY_RSI_GOOD_MIN", "10"))                 # 2 puan
MOMENTUM_BUY_RSI_GOOD_MAX = float(os.getenv("MOMENTUM_BUY_RSI_GOOD_MAX", "90"))                 # 2 puan
MOMENTUM_BUY_RSI_EXTREME_MIN = float(os.getenv("MOMENTUM_BUY_RSI_EXTREME_MIN", "5"))            # Reddetme eşiği
MOMENTUM_BUY_RSI_EXTREME_MAX = float(os.getenv("MOMENTUM_BUY_RSI_EXTREME_MAX", "95"))           # Reddetme eşiği

# Momentum Buy - ADX Levels (ULTRA KOLAY!)
MOMENTUM_BUY_ADX_EXCELLENT = float(os.getenv("MOMENTUM_BUY_ADX_EXCELLENT", "15"))               # 3 puan
MOMENTUM_BUY_ADX_GOOD = float(os.getenv("MOMENTUM_BUY_ADX_GOOD", "10"))                         # 2 puan  
MOMENTUM_BUY_ADX_DECENT = float(os.getenv("MOMENTUM_BUY_ADX_DECENT", "5"))                      # 1 puan

# Momentum Buy - Volume Levels
MOMENTUM_BUY_VOLUME_EXCELLENT = float(os.getenv("MOMENTUM_BUY_VOLUME_EXCELLENT", "1.2"))        # 3 puan
MOMENTUM_BUY_VOLUME_GOOD = float(os.getenv("MOMENTUM_BUY_VOLUME_GOOD", "1.0"))                  # 2 puan
MOMENTUM_BUY_VOLUME_DECENT = float(os.getenv("MOMENTUM_BUY_VOLUME_DECENT", "0.8"))              # 1 puan

# Momentum Buy - Price Momentum Levels
MOMENTUM_BUY_PRICE_MOM_EXCELLENT = float(os.getenv("MOMENTUM_BUY_PRICE_MOM_EXCELLENT", "0.0005"))  # 3 puan
MOMENTUM_BUY_PRICE_MOM_GOOD = float(os.getenv("MOMENTUM_BUY_PRICE_MOM_GOOD", "0.0001"))           # 2 puan
MOMENTUM_BUY_PRICE_MOM_DECENT = float(os.getenv("MOMENTUM_BUY_PRICE_MOM_DECENT", "-0.0005"))      # 1 puan (negatif bile kabul)

# ==============================================
# 💎 MOMENTUM SELL CONDITIONS - PHASE SYSTEM!
# ==============================================

# Momentum Sell - Minimum Hold Time
MOMENTUM_SELL_MIN_HOLD_MINUTES = int(os.getenv("MOMENTUM_SELL_MIN_HOLD_MINUTES", "25"))          # Minimum tutma süresi

# Momentum Sell - Emergency Exit
MOMENTUM_SELL_CATASTROPHIC_LOSS_PCT = float(os.getenv("MOMENTUM_SELL_CATASTROPHIC_LOSS_PCT", "-0.03"))  # %3 zarar = acil çıkış

# Momentum Sell - Premium Profit Levels (zaman fark etmez)
MOMENTUM_SELL_PREMIUM_EXCELLENT = float(os.getenv("MOMENTUM_SELL_PREMIUM_EXCELLENT", "5.0"))     # $5+ kar
MOMENTUM_SELL_PREMIUM_GREAT = float(os.getenv("MOMENTUM_SELL_PREMIUM_GREAT", "3.0"))            # $3+ kar
MOMENTUM_SELL_PREMIUM_GOOD = float(os.getenv("MOMENTUM_SELL_PREMIUM_GOOD", "2.0"))              # $2+ kar

# Momentum Sell - Phase 1 (25-60 dakika)
MOMENTUM_SELL_PHASE1_EXCELLENT = float(os.getenv("MOMENTUM_SELL_PHASE1_EXCELLENT", "1.5"))      # $1.5 kar
MOMENTUM_SELL_PHASE1_GOOD = float(os.getenv("MOMENTUM_SELL_PHASE1_GOOD", "1.0"))               # $1.0 kar
MOMENTUM_SELL_PHASE1_LOSS_PROTECTION = float(os.getenv("MOMENTUM_SELL_PHASE1_LOSS_PROTECTION", "-2.0"))  # -$2 zarar

# Momentum Sell - Phase 2 (60-120 dakika)  
MOMENTUM_SELL_PHASE2_EXCELLENT = float(os.getenv("MOMENTUM_SELL_PHASE2_EXCELLENT", "1.0"))      # $1.0 kar
MOMENTUM_SELL_PHASE2_GOOD = float(os.getenv("MOMENTUM_SELL_PHASE2_GOOD", "0.75"))              # $0.75 kar
MOMENTUM_SELL_PHASE2_DECENT = float(os.getenv("MOMENTUM_SELL_PHASE2_DECENT", "0.50"))          # $0.50 kar (90dk+)
MOMENTUM_SELL_PHASE2_LOSS_PROTECTION = float(os.getenv("MOMENTUM_SELL_PHASE2_LOSS_PROTECTION", "-1.5"))  # -$1.5 zarar

# Momentum Sell - Phase 3 (120-180 dakika)
MOMENTUM_SELL_PHASE3_EXCELLENT = float(os.getenv("MOMENTUM_SELL_PHASE3_EXCELLENT", "0.75"))     # $0.75 kar
MOMENTUM_SELL_PHASE3_GOOD = float(os.getenv("MOMENTUM_SELL_PHASE3_GOOD", "0.50"))              # $0.50 kar
MOMENTUM_SELL_PHASE3_DECENT = float(os.getenv("MOMENTUM_SELL_PHASE3_DECENT", "0.25"))          # $0.25 kar
MOMENTUM_SELL_PHASE3_BREAKEVEN_MIN = float(os.getenv("MOMENTUM_SELL_PHASE3_BREAKEVEN_MIN", "-0.20"))  # Kırılma noktası
MOMENTUM_SELL_PHASE3_BREAKEVEN_MAX = float(os.getenv("MOMENTUM_SELL_PHASE3_BREAKEVEN_MAX", "0.20"))   # Kırılma noktası
MOMENTUM_SELL_PHASE3_LOSS_PROTECTION = float(os.getenv("MOMENTUM_SELL_PHASE3_LOSS_PROTECTION", "-1.0"))  # -$1 zarar

# Momentum Sell - Phase 4 (180+ dakika)
MOMENTUM_SELL_PHASE4_EXCELLENT = float(os.getenv("MOMENTUM_SELL_PHASE4_EXCELLENT", "0.50"))     # $0.50 kar
MOMENTUM_SELL_PHASE4_GOOD = float(os.getenv("MOMENTUM_SELL_PHASE4_GOOD", "0.25"))              # $0.25 kar
MOMENTUM_SELL_PHASE4_MINIMAL = float(os.getenv("MOMENTUM_SELL_PHASE4_MINIMAL", "0.10"))        # $0.10 kar
MOMENTUM_SELL_PHASE4_BREAKEVEN_MIN = float(os.getenv("MOMENTUM_SELL_PHASE4_BREAKEVEN_MIN", "-0.30"))  # Geniş kırılma
MOMENTUM_SELL_PHASE4_BREAKEVEN_MAX = float(os.getenv("MOMENTUM_SELL_PHASE4_BREAKEVEN_MAX", "0.30"))   # Geniş kırılma
MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES = int(os.getenv("MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES", "240"))  # 4 saat zorla çıkış

# Momentum Sell - Loss Protection
MOMENTUM_SELL_LOSS_MULTIPLIER = float(os.getenv("MOMENTUM_SELL_LOSS_MULTIPLIER", "4.0"))        # İşlem maliyeti çarpanı

# Momentum Sell - Technical Exit Conditions
MOMENTUM_SELL_TECH_MIN_MINUTES = int(os.getenv("MOMENTUM_SELL_TECH_MIN_MINUTES", "90"))         # Teknik exit min süre
MOMENTUM_SELL_TECH_MIN_LOSS = float(os.getenv("MOMENTUM_SELL_TECH_MIN_LOSS", "-1.5"))          # Teknik exit min zarar
MOMENTUM_SELL_TECH_RSI_EXTREME = float(os.getenv("MOMENTUM_SELL_TECH_RSI_EXTREME", "12"))       # RSI aşırı satım

# ==============================================
# ⏰ MOMENTUM TIMING SETTINGS - BEKLEME SÜRELERİ!
# ==============================================

# Portfolio durumuna göre bekleme süreleri (saniye)
MOMENTUM_WAIT_PROFIT_5PCT = int(os.getenv("MOMENTUM_WAIT_PROFIT_5PCT", "300"))     # %5+ kar varsa 5dk
MOMENTUM_WAIT_PROFIT_2PCT = int(os.getenv("MOMENTUM_WAIT_PROFIT_2PCT", "450"))     # %2+ kar varsa 7.5dk  
MOMENTUM_WAIT_BREAKEVEN = int(os.getenv("MOMENTUM_WAIT_BREAKEVEN", "600"))         # Kırılma noktası 10dk
MOMENTUM_WAIT_LOSS = int(os.getenv("MOMENTUM_WAIT_LOSS", "900"))                   # Zarar durumu 15dk