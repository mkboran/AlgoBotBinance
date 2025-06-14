# utils/ai_signal_provider.py
from enum import Enum
from typing import Dict, Any, Optional, Union, List, Tuple
# import joblib # ML modeli yüklemek için (şu an kullanılmıyor)
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import asyncio
import pandas_ta as ta

from utils.config import settings
from utils.logger import logger

class AiSignal(Enum):
    """Yapay zeka modelinden gelebilecek potansiyel sinyal türleri."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_OPINION = "NO_OPINION"

class MarketRiskLevel(Enum):
    """Market risk seviyeleri"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW" 
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class MarketRiskAssessment:
    """Market risk değerlendirme sonucu"""
    def __init__(self, level: MarketRiskLevel, score: float, factors: Dict[str, float], recommendation: str):
        self.level = level
        self.score = score  # 0-1 arası
        self.factors = factors
        self.recommendation = recommendation

class AiPerformanceTracker:
    """AI performans takip sistemi"""
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True) # log_path None değilse çalışır
        
    def log_prediction(self, signal_type: str, confidence: float, context: Dict, actual_outcome: Optional[str] = None):
        """AI prediction'ını logla"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_type": signal_type,
            "confidence": float(confidence),
            "context": self._sanitize_context(context),
            "actual_outcome": actual_outcome,
            "prediction_id": hashlib.md5(f"{signal_type}_{confidence}_{datetime.now()}".encode()).hexdigest()[:8]
        }
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"AI performance log error: {e}")

    def _sanitize_context(self, context: Dict) -> Dict:
        """Context'i JSON serializable hale getir"""
        sanitized = {}
        for key, value in context.items():
            if isinstance(value, (np.bool_, bool)):
                sanitized[key] = bool(value)
            elif isinstance(value, (np.integer, int)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                sanitized[key] = float(value)
            elif isinstance(value, str):
                sanitized[key] = value
            elif pd.isna(value) or value is None:
                sanitized[key] = None
            else:
                sanitized[key] = str(value)
        return sanitized

class AiSignalProvider:
    """🤖 ULTRA-ADVANCED AI Signal Provider with Market Intelligence"""
    
    # ai_signal_provider.py -> AiSignalProvider sınıfının içine yapıştırılacak

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        """
        Yapıcı metod. Optimizasyondan gelen override parametrelerini kabul eder.
        """
        if overrides is None:
            overrides = {}

        # Her bir ayarı, önce 'overrides' sözlüğünde ara, bulamazsan 'settings'den al.
        self.is_enabled: bool = overrides.get("ai_assistance_enabled", settings.AI_ASSISTANCE_ENABLED)
        self.model_path: Optional[str] = overrides.get("ai_model_path", settings.AI_MODEL_PATH)
        self.operation_mode: str = overrides.get("ai_operation_mode", settings.AI_OPERATION_MODE)
        self.default_confidence_threshold: float = overrides.get("ai_confidence_threshold", settings.AI_CONFIDENCE_THRESHOLD)
        
        self.ta_ema_periods: Dict[str, Tuple[int, ...]] = {
            'main': overrides.get("ai_ta_ema_periods_main_tf", settings.AI_TA_EMA_PERIODS_MAIN_TF),
            'long': overrides.get("ai_ta_ema_periods_long_tf", settings.AI_TA_EMA_PERIODS_LONG_TF)
        }
        self.ta_rsi_period: int = overrides.get("ai_ta_rsi_period", settings.AI_TA_RSI_PERIOD)
        self.ta_divergence_lookback: int = overrides.get("ai_ta_divergence_lookback", settings.AI_TA_DIVERGENCE_LOOKBACK)
        
        self.ta_weights: Dict[str, float] = {
            'trend_main': overrides.get("ai_weight_trend_main", settings.AI_TA_WEIGHT_TREND_MAIN),
            'trend_long': overrides.get("ai_weight_trend_long", settings.AI_TA_WEIGHT_TREND_LONG),
            'volume': overrides.get("ai_weight_volume", settings.AI_TA_WEIGHT_VOLUME),
            'divergence': overrides.get("ai_weight_divergence", settings.AI_TA_WEIGHT_DIVERGENCE),
        }
        
        # Bu kontrol, ağırlıkların toplamının 1 olup olmadığını doğrulamak için yararlıdır.
        if abs(sum(self.ta_weights.values()) - 1.0) > 1e-6 and self.is_enabled and self.operation_mode == 'technical_analysis':
            logger.warning(f"AI TA ağırlıklarının toplamı ({sum(self.ta_weights.values()):.2f}) 1.0 değil! Ayarları kontrol edin.")

        # DÜZELTİLDİ: Bu parametreler de 'overrides' kullanmalı.
        self.risk_assessment_enabled = overrides.get("ai_risk_assessment_enabled", settings.AI_RISK_ASSESSMENT_ENABLED)
        self.volatility_threshold = overrides.get("ai_risk_volatility_threshold", settings.AI_RISK_VOLATILITY_THRESHOLD)
        self.volume_spike_threshold = overrides.get("ai_risk_volume_spike_threshold", settings.AI_RISK_VOLUME_SPIKE_THRESHOLD)
        
        self.performance_tracker = None
        # Bu kısım 'settings'den okunabilir çünkü optimizasyonla ilgili değil.
        if self.is_enabled and settings.AI_TRACK_PERFORMANCE and settings.AI_PERFORMANCE_LOG_PATH:
            self.performance_tracker = AiPerformanceTracker(settings.AI_PERFORMANCE_LOG_PATH)
        
        # DÜZELTİLDİ: Bu parametreler de 'overrides' kullanmalı.
        self.strategy_confidence_overrides = {
            "Momentum": overrides.get("ai_momentum_confidence_override", settings.AI_MOMENTUM_CONFIDENCE_OVERRIDE),
            "BollingerRSI": overrides.get("ai_bollinger_confidence_override", settings.AI_BOLLINGER_CONFIDENCE_OVERRIDE)
        }
        
        self.ml_model: Any = None
        if self.is_enabled:
            if self.operation_mode == 'ml_model' and self.model_path:
                try:
                    # self.ml_model = joblib.load(self.model_path)
                    logger.info(f"ML modeli yüklenecek (placeholder): {self.model_path}")
                except Exception as e:
                    logger.error(f"ML modeli yüklenemedi: {self.model_path}: {e}")

            logger.info(f"🤖 AI Sinyal Sağlayıcı başlatıldı (CONFIG + OVERRIDES)")
            logger.info(f"   Mod: {self.operation_mode.upper()}, Varsayılan AI Eşiği: {self.default_confidence_threshold}")
        else:
            logger.info("🤖 AI Sinyal Sağlayıcı başlatıldı - DEVRE DIŞI")

            
    def _resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            logger.debug(f"[{self.__class__.__name__}] _resample_ohlcv: Input DataFrame is empty or None for timeframe '{timeframe}'.")
            return None
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index(pd.to_datetime(df['timestamp'], utc=True))
                else:
                    return None
            
            if df.index.tz is None:
                df = df.tz_localize('UTC')
            elif str(df.index.tz).upper() != 'UTC':
                df = df.tz_convert('UTC')

            timeframe_mapping = {
                '1s': '1s', '5s': '5s', '15s': '15s', '30s': '30s',
                '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min', 
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
                '1d': 'D', '1w': 'W', '1M': 'ME' 
            }
            
            timeframe_key = timeframe.lower()
            timeframe_fixed = timeframe_mapping.get(timeframe_key)
            
            if timeframe_fixed is None:
                timeframe_fixed = timeframe_key

            logger.debug(f"[{self.__class__.__name__}] _resample_ohlcv: Resampling to '{timeframe_fixed}' from input timeframe (original: '{timeframe}').")
            
            resampled_df = df.resample(timeframe_fixed).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna() 
            
            if resampled_df.empty:
                logger.warning(f"Resampling resulted in empty DataFrame")
            return resampled_df
        except Exception as e:
            tf_fixed_str = locals().get('timeframe_fixed', 'unknown_target_timeframe')
            logger.error(f"[{self.__class__.__name__}] AiSignalProvider: Data resampling error for timeframe '{timeframe}' (target: '{tf_fixed_str}'): {e}", exc_info=True)
            return None
            
    def _calculate_trend_strength(self, df: pd.DataFrame, ema_periods: Tuple[int, ...]) -> float:
        if df is None or len(df) < max(ema_periods, default=1) + 2: return 0.0
        try:
            if len(ema_periods) < 3:
                logger.warning(f"Trend strength calculation requires at least 3 EMA periods")
                return 0.0

            ema_s = ta.ema(df['close'], length=ema_periods[0]) # pandas_ta kullanımı daha iyi olabilir
            ema_m = ta.ema(df['close'], length=ema_periods[1])
            ema_l = ta.ema(df['close'], length=ema_periods[2])

            if ema_s is None or ema_m is None or ema_l is None:
                return 0.0

            current_price = df['close'].iloc[-1]
            last_ema_s, last_ema_m, last_ema_l = ema_s.iloc[-1], ema_m.iloc[-1], ema_l.iloc[-1]

            if pd.isna(current_price) or pd.isna(last_ema_s):
                return 0.0
            
            pos_score = 0.0
            if last_ema_s > last_ema_m > last_ema_l: 
                pos_score = 1.0
            elif last_ema_s < last_ema_m < last_ema_l: 
                pos_score = -1.0
            elif last_ema_s > last_ema_m: 
                pos_score = 0.5
            elif last_ema_s < last_ema_m: 
                pos_score = -0.5
            
            price_vs_ema_score = 0.0
            if current_price > last_ema_s > last_ema_m: 
                price_vs_ema_score = 0.8
            elif current_price < last_ema_s < last_ema_m: 
                price_vs_ema_score = -0.8
            
            slope_score = 0.0
            if len(ema_s) >= 3:
                ema_s_slope = (last_ema_s - ema_s.iloc[-3]) / ema_s.iloc[-3]
                slope_score = np.tanh(ema_s_slope * 100)

            final_score = (pos_score * 0.4) + (price_vs_ema_score * 0.4) + (slope_score * 0.2)
            return np.clip(final_score, -1.0, 1.0)
        except Exception as e:
            logger.warning(f"Trend strength calculation error: {e}", exc_info=True)
            return 0.0

    def _analyze_volume_profile(self, df: pd.DataFrame, lookback: int = 5) -> float:
        if df is None or len(df) < max(lookback, 2): return 0.0
        try:
            recent_volumes = df['volume'].tail(lookback)
            if len(recent_volumes) < 2: 
                return 0.0
            avg_volume = recent_volumes.iloc[:-1].mean() 
            last_volume = recent_volumes.iloc[-1]
            
            if pd.isna(avg_volume) or avg_volume == 0: 
                return 0.0
            if pd.isna(last_volume): 
                return 0.0
            
            volume_change_ratio = (last_volume - avg_volume) / avg_volume
            price_change_pct = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            
            volume_score = 0.0
            if price_change_pct > 0.001 and volume_change_ratio > 0.2: 
                volume_score = 1.0
            elif price_change_pct < -0.001 and volume_change_ratio > 0.2: 
                volume_score = -1.0
            elif volume_change_ratio > 0.5: 
                volume_score = 0.5

            volume_momentum = np.tanh(volume_change_ratio * 2)
            final_score = np.clip((volume_score * 0.7) + (volume_momentum * 0.3), -1.0, 1.0)
            return final_score
        except Exception as e:
            logger.warning(f"Volume analysis error: {e}", exc_info=True)
            return 0.0

    def _calculate_rsi_divergence(self, df: pd.DataFrame, rsi_period: int, lookback: int) -> float:
        if df is None or len(df) < max(rsi_period, lookback) + 1: return 0.0
        try:
            rsi = ta.rsi(df['close'], length=rsi_period)
            if rsi is None or len(rsi) < lookback: 
                return 0.0

            price_end = df['close'].iloc[-1]
            price_start_series = df['close'].iloc[-lookback : -lookback + 1 if -lookback + 1 != 0 else None]
            if price_start_series.empty: 
                return 0.0
            price_start = price_start_series.iloc[0]
            
            rsi_end_series = rsi.iloc[-1:]
            rsi_start_series = rsi.iloc[-lookback : -lookback + 1 if -lookback + 1 != 0 else None]
            if rsi_end_series.empty or rsi_start_series.empty: 
                return 0.0
            rsi_end = rsi_end_series.iloc[0]
            rsi_start = rsi_start_series.iloc[0]

            if pd.isna(price_end) or price_start == 0: 
                return 0.0

            price_change_pct = (price_end - price_start) / price_start
            rsi_change = rsi_end - rsi_start
            
            if price_change_pct > 0.01 and rsi_change < -5: 
                return -0.8
            elif price_change_pct < -0.01 and rsi_change > 5: 
                return 0.8
            elif price_change_pct > 0.005 and rsi_change < -2: 
                return -0.4
            elif price_change_pct < -0.005 and rsi_change > 2: 
                return 0.4
            return 0.0
        except Exception as e:
            logger.warning(f"RSI divergence error: {e}", exc_info=True)
            return 0.0

    def _get_technical_analysis_score(self, ohlcv_df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> float:
        if ohlcv_df is None or ohlcv_df.empty: return 0.0

        trend_main = self._calculate_trend_strength(ohlcv_df, self.ta_ema_periods['main'])
        volume_main = self._analyze_volume_profile(ohlcv_df)
        divergence_main = self._calculate_rsi_divergence(ohlcv_df, self.ta_rsi_period, self.ta_divergence_lookback)

        ohlcv_df_long_tf = self._resample_ohlcv(ohlcv_df, timeframe=settings.AI_TA_LONG_TIMEFRAME_STR)
        trend_long = 0.0
        if ohlcv_df_long_tf is not None and not ohlcv_df_long_tf.empty and len(ohlcv_df_long_tf) > max(self.ta_ema_periods['long'], default=1):
            trend_long = self._calculate_trend_strength(ohlcv_df_long_tf, self.ta_ema_periods['long'])
        
        current_weights = self.ta_weights
        if context and isinstance(context.get("ai_ta_weights"), dict):
            current_weights = context["ai_ta_weights"]

        score = (
            trend_main * current_weights.get('trend_main', 0.4) +
            trend_long * current_weights.get('trend_long', 0.3) +
            volume_main * current_weights.get('volume', 0.2) +
            divergence_main * current_weights.get('divergence', 0.1)
        )
        return np.clip(score, -1.0, 1.0)

    def _calculate_market_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        if df is None or len(df) < window : return 0.0
        try:
            returns = df['close'].pct_change().dropna().tail(window)
            if returns.empty: 
                return 0.0
            volatility = returns.std() 
            return volatility * 100 if pd.notna(volatility) else 0.0
        except Exception as e:
            logger.warning(f"Volatility calculation error: {e}", exc_info=True)
            return 0.0

    def _detect_unusual_patterns(self, df: pd.DataFrame) -> Dict[str, Any]: # Any olarak değiştirildi
        patterns: Dict[str, Any] = {} # Any olarak değiştirildi
        if df is None or df.empty: return patterns
        try:
            if len(df) >= 20:
                patterns["volume_spike_ratio"] = df['volume'].iloc[-1] / df['volume'].tail(20).mean()
            
            if len(df) >= 2:
                patterns["gap_detected"] = abs(df['open'].iloc[-1] - df['close'].iloc[-2]) > 0.01

        except Exception as e:
            logger.warning(f"Pattern detection error: {e}", exc_info=True)
        return patterns

    async def assess_market_risk(self, df: pd.DataFrame, context: Optional[Dict] = None) -> MarketRiskAssessment:
        if not self.risk_assessment_enabled or df is None or df.empty:
            return MarketRiskAssessment(MarketRiskLevel.MODERATE, 0.5, {}, "Risk assessment disabled or no data")
        
        risk_factors: Dict[str, float] = {} # Tip belirtildi
        try:
            volatility_pct = self._calculate_market_volatility(df) 
            vol_thresh_interpreted = self.volatility_threshold * 100 if self.volatility_threshold <=1 else self.volatility_threshold # Config'deki %0.02 gibi değerler için
            vol_risk = min(1.0, volatility_pct / vol_thresh_interpreted if vol_thresh_interpreted > 0 else 1.0)
            risk_factors['volatility_pct'] = round(volatility_pct,2)
            risk_factors['volatility_risk_score'] = round(vol_risk,2)
            
            patterns = self._detect_unusual_patterns(df)
            volume_spike_ratio = patterns.get('volume_spike_ratio', 1.0)
            vol_spike_thresh_interpreted = self.volume_spike_threshold # Config'deki 2.0 gibi değerler için
            volume_risk = min(1.0, max(0.0, (volume_spike_ratio - 1.0) / (vol_spike_thresh_interpreted -1 if vol_spike_thresh_interpreted > 1 else 1))) 
            risk_factors['volume_spike_ratio'] = round(volume_spike_ratio,2)
            risk_factors['volume_anomaly_risk_score'] = round(volume_risk,2)
            
            total_risk = np.clip((vol_risk * 0.6) + (volume_risk * 0.4), 0.0, 1.0) # Basit ağırlıklandırma
            
            level = MarketRiskLevel.MODERATE
            if total_risk < 0.2: level = MarketRiskLevel.VERY_LOW
            elif total_risk < 0.4: level = MarketRiskLevel.LOW
            elif total_risk < 0.7: level = MarketRiskLevel.MODERATE
            elif total_risk < 0.85: level = MarketRiskLevel.HIGH
            else: level = MarketRiskLevel.EXTREME
            
            recommendation = f"Calculated total risk: {total_risk:.2f}. Factors: VolScore={vol_risk:.2f}, VolumeScore={volume_risk:.2f}"
            return MarketRiskAssessment(level, total_risk, risk_factors, recommendation)
            
        except Exception as e:
            logger.error(f"Market risk assessment error: {e}", exc_info=True)
            return MarketRiskAssessment(MarketRiskLevel.MODERATE, 0.5, {}, f"Assessment error: {str(e)[:100]}")

    async def get_ai_confirmation(
        self, current_signal_type: str, ohlcv_df: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        if not self.is_enabled: return True
        if current_signal_type.upper() == "SELL": return True 
            
        if current_signal_type.upper() == "BUY":
            if ohlcv_df is None or ohlcv_df.empty: return False
            ctx = context if context is not None else {} 

            ta_score = self._get_technical_analysis_score(ohlcv_df, ctx)
            
            strategy_name = ctx.get("strategy_name", "UnknownStrategy")
            confidence_threshold = self.strategy_confidence_overrides.get(strategy_name, self.default_confidence_threshold)

            portfolio_profit_pct = ctx.get("portfolio_profit_pct", 0.0)
            if portfolio_profit_pct < -0.05: 
                confidence_threshold *= 1.5
            elif portfolio_profit_pct < -0.02: 
                confidence_threshold *= 1.2
            elif portfolio_profit_pct < 0:
                confidence_threshold *= 1.1

            quality_score = ctx.get("quality_score", 0)
            min_quality_required = ctx.get("min_quality_required", settings.AI_CONFIRM_MIN_QUALITY_SCORE)
            
            # Config'deki detaylı kontrolleri de ekleyelim (opsiyonel, gerekirse açılır)
            # ema_spread_1 = ctx.get("ema_spread_1", 0); ema_spread_2 = ctx.get("ema_spread_2", 0)
            # volume_ratio = ctx.get("volume_ratio", 1.0); price_momentum = ctx.get("price_momentum", 0)
            # ema_momentum = ctx.get("ema_momentum", 0); adx = ctx.get("adx", 0)
            # if not (ema_spread_1 >= settings.AI_CONFIRM_MIN_EMA_SPREAD_1 and ...): return False ...

            is_approved = ta_score >= confidence_threshold and quality_score >= min_quality_required
            log_level = logger.info if is_approved else logger.debug # Redleri debug olarak logla
            
            log_level(f"🤖 AI {'APPROVED' if is_approved else 'REJECTED'} BUY: Strat='{strategy_name}', TA={ta_score:.3f} (Req>={confidence_threshold:.3f}), Q={quality_score} (Req>={min_quality_required})")
            
            if self.performance_tracker:
                self.performance_tracker.log_prediction(
                    f"BUY_{'CONFIRMED' if is_approved else 'REJECTED'}", 
                    ta_score, 
                    ctx
                )
            return is_approved
        return True

    async def get_market_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_enabled: return {"status": "AI disabled"}
        intelligence = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "active"}
        try:
            ta_score = self._get_technical_analysis_score(df)
            intelligence["technical_score"] = round(ta_score, 3)
            if ta_score > settings.AI_TA_STANDALONE_THRESH_BUY: intelligence["technical_bias"] = "BULLISH"
            elif ta_score < settings.AI_TA_STANDALONE_THRESH_SELL: intelligence["technical_bias"] = "BEARISH"
            else: intelligence["technical_bias"] = "NEUTRAL"
            
            risk_assessment = await self.assess_market_risk(df)
            intelligence["risk_level"] = risk_assessment.level.value
            intelligence["risk_score"] = round(risk_assessment.score, 3)
            intelligence["risk_factors"] = {k: round(v,2) for k,v in risk_assessment.factors.items()}
            intelligence["risk_recommendation"] = risk_assessment.recommendation
            
            volatility = self._calculate_market_volatility(df)
            intelligence["current_volatility_pct"] = round(volatility, 2)
            vol_thresh_pct = settings.AI_RISK_VOLATILITY_THRESHOLD * 100 if settings.AI_RISK_VOLATILITY_THRESHOLD <=1 else settings.AI_RISK_VOLATILITY_THRESHOLD
            intelligence["volatility_status"] = "HIGH" if volatility > vol_thresh_pct else ("NORMAL" if volatility > vol_thresh_pct * 0.5 else "LOW")
            
        except Exception as e:
            logger.error(f"Market intelligence error: {e}", exc_info=True)
            intelligence["error"] = str(e)
        return intelligence

    async def get_standalone_signal(self, ohlcv_df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> AiSignal:
        if not self.is_enabled: return AiSignal.NO_OPINION
        ctx = context or {}
        if self.operation_mode == 'technical_analysis':
            ta_score = self._get_technical_analysis_score(ohlcv_df, ctx)
            # logger.debug(f"🧠 Standalone AI Signal (TA Mode): Score = {ta_score:.3f}") # Bu çok sık log üretebilir
            if ta_score > settings.AI_TA_STANDALONE_THRESH_STRONG_BUY: return AiSignal.STRONG_BUY
            if ta_score > settings.AI_TA_STANDALONE_THRESH_BUY: return AiSignal.BUY
            if ta_score < settings.AI_TA_STANDALONE_THRESH_STRONG_SELL: return AiSignal.STRONG_SELL
            if ta_score < settings.AI_TA_STANDALONE_THRESH_SELL: return AiSignal.SELL
            return AiSignal.HOLD
        
        logger.warning(f"Unknown AI operation mode or ML model not loaded: {self.operation_mode}")
        return AiSignal.NO_OPINION

async def main_test():
    if not settings.AI_ASSISTANCE_ENABLED:
        print("AI Assistance is disabled in config. Enable for testing.")
        return

    print(f"AI Provider Test. Mode: {settings.AI_OPERATION_MODE}, Default Threshold: {settings.AI_CONFIDENCE_THRESHOLD}")
    ai_provider = AiSignalProvider()
    
    # Create a more realistic sample OHLCV DataFrame
    data_size = 200
    base_price = 60000
    data = {
        'open': base_price + np.random.randn(data_size).cumsum() + np.random.rand(data_size) * 50 - 25,
        'close': base_price + np.random.randn(data_size).cumsum() + np.random.rand(data_size) * 50 - 25,
        'volume': np.random.rand(data_size) * 100 + 10
    }
    data['high'] = np.maximum(data['open'], data['close']) + np.random.rand(data_size) * 20
    data['low'] = np.minimum(data['open'], data['close']) - np.random.rand(data_size) * 20
    
    index = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=data_size, freq=settings.TIMEFRAME) # Use config timeframe
    sample_df = pd.DataFrame(data, index=index)
    sample_df = sample_df[['open', 'high', 'low', 'close', 'volume']] # Correct column order

    logger.info(f"Sample OHLCV data (last 5 rows):\n{sample_df.tail()}")

    print("\n--- Test: get_ai_confirmation ---")
    # Simulate context from a strategy
    buy_context_momentum = {
        "strategy_name": "Momentum",
        "quality_score": 15, 
        "min_quality_required": settings.MOMENTUM_BUY_MIN_QUALITY_SCORE,
        "portfolio_profit_pct": -0.01, # Slight loss
        # Add other context fields that get_ai_confirmation might expect from MomentumStrategy
        "ema_spread_1": 0.0005, "ema_spread_2": 0.0006, "volume_ratio": 1.5,
        "price_momentum": 0.001, "ema_momentum": 0.0012, "adx": 25.0
    }
    buy_confirmation = await ai_provider.get_ai_confirmation("BUY", sample_df.copy(), context=buy_context_momentum)
    print(f"BUY confirmation for Momentum: {buy_confirmation}")
    
    # Test with a different strategy or context
    buy_context_rsi = {"strategy_name": "RsiStrategy", "quality_score": 5, "portfolio_profit_pct": 0.02}
    buy_confirmation_rsi = await ai_provider.get_ai_confirmation("BUY", sample_df.copy(), context=buy_context_rsi)
    print(f"BUY confirmation for RsiStrategy: {buy_confirmation_rsi}")


    print("\n--- Test: get_standalone_signal ---")
    standalone_signal = await ai_provider.get_standalone_signal(sample_df.copy())
    print(f"Standalone AI Signal: {standalone_signal.name if standalone_signal else 'N/A'}")

    print("\n--- Test: get_market_intelligence ---")
    market_intel = await ai_provider.get_market_intelligence(sample_df.copy())
    print(f"Market Intelligence:")
    for k, v in market_intel.items():
        if isinstance(v, dict):
            print(f"  {k}: {json.dumps(v, indent=2)}")
        else:
            print(f"  {k}: {v}")
    
    print("\n--- Test: _resample_ohlcv (15m to 1h) ---")
    if settings.TIMEFRAME == '15m' and settings.AI_TA_LONG_TIMEFRAME_STR == '1h':
        resampled = ai_provider._resample_ohlcv(sample_df.copy(), '1h')
        if resampled is not None:
            print(f"Resampled data shape: {resampled.shape}")
        else:
            print("Resampling failed or returned None")
    else:
        print("Skipping resample test as TIMEFRAME is not 15m or AI_TA_LONG_TIMEFRAME_STR is not 1h in config.")

if __name__ == "__main__":
    # logger.py içindeki trading_logger_instance'ın doğru başlatıldığından emin olmak için
    # bir kez daha çağırabiliriz, ancak bu normalde logger.py import edildiğinde gerçekleşir.
    try:
        from utils.logger import trading_logger_instance
        logger.info("Main test for AI Provider starting...")
    except ImportError:
        pass # Zaten en üstte import edildi.
    
    asyncio.run(main_test())