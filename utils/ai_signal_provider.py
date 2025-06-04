# utils/ai_signal_provider.py
import random # Test için gerekirse
from enum import Enum
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import asyncio # Test için
import pandas_ta as ta
import joblib # ML modeli yüklemek için
import json
from datetime import datetime, timezone
from pathlib import Path
import hashlib

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
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_prediction(self, signal_type: str, confidence: float, context: Dict, actual_outcome: Optional[str] = None):
        """AI prediction'ını logla"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_type": signal_type,
            "confidence": float(confidence),  # numpy types'ları float'a çevir
            "context": self._sanitize_context(context),  # Context'i sanitize et
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
            elif value is None:
                sanitized[key] = None
            else:
                sanitized[key] = str(value)  # Diğer türleri string'e çevir
        return sanitized

class AiSignalProvider:
    """🤖 ULTRA-ADVANCED AI Signal Provider with Market Intelligence"""
    
    def __init__(self):
        # 🔧 ALL SETTINGS FROM CONFIG NOW!
        self.is_enabled: bool = settings.AI_ASSISTANCE_ENABLED
        self.model_path: Optional[str] = settings.AI_MODEL_PATH
        self.operation_mode: str = settings.AI_OPERATION_MODE
        
        self.default_confidence_threshold: float = settings.AI_CONFIDENCE_THRESHOLD
        
        # Technical Analysis Parameters - ALL FROM CONFIG
        self.ta_ema_periods: Dict[str, Tuple[int, int, int]] = {
            'main': settings.AI_TA_EMA_PERIODS_MAIN_TF,
            'long': settings.AI_TA_EMA_PERIODS_LONG_TF
        }
        self.ta_rsi_period: int = settings.AI_TA_RSI_PERIOD
        self.ta_divergence_lookback: int = settings.AI_TA_DIVERGENCE_LOOKBACK
        
        self.ta_weights: Dict[str, float] = {
            'trend_main': settings.AI_TA_WEIGHT_TREND_MAIN,
            'trend_long': settings.AI_TA_WEIGHT_TREND_LONG,
            'volume': settings.AI_TA_WEIGHT_VOLUME,
            'divergence': settings.AI_TA_WEIGHT_DIVERGENCE,
        }
        
        # Validate weights sum to 1.0
        if abs(sum(self.ta_weights.values()) - 1.0) > 1e-6 and self.is_enabled and self.operation_mode == 'technical_analysis':
            logger.warning(f"AI TA weights sum ({sum(self.ta_weights.values())}) != 1.0! Check config.")

        # Risk Assessment - ALL FROM CONFIG
        self.risk_assessment_enabled = settings.AI_RISK_ASSESSMENT_ENABLED
        self.volatility_threshold = settings.AI_RISK_VOLATILITY_THRESHOLD
        self.volume_spike_threshold = settings.AI_RISK_VOLUME_SPIKE_THRESHOLD
        
        # Performance Tracking - FROM CONFIG
        self.performance_tracker = None
        if settings.AI_TRACK_PERFORMANCE:
            self.performance_tracker = AiPerformanceTracker(settings.AI_PERFORMANCE_LOG_PATH)
        
        # Strategy-specific confidence thresholds - FROM CONFIG
        self.strategy_confidence_overrides = {
            "Momentum": settings.AI_MOMENTUM_CONFIDENCE_OVERRIDE,
            "BollingerRSI": settings.AI_BOLLINGER_CONFIDENCE_OVERRIDE
        }
        
        self.ml_model: Any = None
        if self.is_enabled:
            if self.operation_mode == 'ml_model' and self.model_path:
                try:
                    # ML model loading code here
                    pass
                except Exception as e:
                    logger.error(f"ML model loading failed: {e}")
            
            logger.info(f"🤖 AI Signal Provider initialized from CONFIG")
            logger.info(f"   Mode: {self.operation_mode.upper()}, Threshold: {self.default_confidence_threshold}")
            logger.info(f"   TA Weights: Main={self.ta_weights['trend_main']}, Long={self.ta_weights['trend_long']}, Vol={self.ta_weights['volume']}, Div={self.ta_weights['divergence']}")
            logger.info(f"   Strategy Overrides: Momentum={self.strategy_confidence_overrides['Momentum']}, BollingerRSI={self.strategy_confidence_overrides['BollingerRSI']}")
        else:
            logger.info("🤖 AI Signal Provider initialized - DISABLED")

    def _resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Verilen DataFrame'i belirtilen zaman dilimine göre yeniden örnekler."""
        if df is None or df.empty:
            return None
        try:
            # DataFrame index'inin DatetimeIndex olduğundan emin ol
            if not isinstance(df.index, pd.DatetimeIndex):
                # Eğer değilse ve 'timestamp' sütunu varsa onu index yapmayı dene
                if 'timestamp' in df.columns:
                    df = df.set_index(pd.to_datetime(df['timestamp']))
                else: # Zaman damgası bilgisi yoksa veya index uygun değilse None dön
                    logger.warning(f"AiSignalProvider: Yeniden örnekleme için uygun zaman damgası index'i bulunamadı.")
                    return None
            
            # ⚡ FutureWarning düzeltmesi - pandas-compatible format
            timeframe_mapping = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', 
                '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
            }
            timeframe_fixed = timeframe_mapping.get(timeframe, timeframe.replace('m', 'min'))
            
            resampled_df = df.resample(timeframe_fixed).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna() # Tamamlanmamış son barı ve boşlukları at
            return resampled_df
        except Exception as e:
            logger.error(f"AiSignalProvider: Veri yeniden örnekleme hatası ({timeframe}): {e}", exc_info=True)
            return None

    def _calculate_trend_strength(self, df: pd.DataFrame, ema_periods: Tuple[int, int, int]) -> float:
        """Belirtilen EMA periyotları ile trend gücünü [-1, 1] aralığında hesaplar - GÜÇLENDİRİLMİŞ!"""
        if df is None or len(df) < max(ema_periods) + 2:
            return 0.0
        try:
            ema_s = df['close'].ewm(span=ema_periods[0], adjust=False).mean()
            ema_m = df['close'].ewm(span=ema_periods[1], adjust=False).mean()
            ema_l = df['close'].ewm(span=ema_periods[2], adjust=False).mean()

            # Çok daha agresif trend analizi
            current_price = df['close'].iloc[-1]
            
            # 1. Pozisyonel skor (EMA sıralaması)
            pos_score = 0
            if ema_s.iloc[-1] > ema_m.iloc[-1] > ema_l.iloc[-1]:
                pos_score = 1.0  # Güçlü yükseliş trendi
            elif ema_s.iloc[-1] < ema_m.iloc[-1] < ema_l.iloc[-1]:
                pos_score = -1.0  # Güçlü düşüş trendi
            elif ema_s.iloc[-1] > ema_m.iloc[-1]:
                pos_score = 0.5  # Orta yükseliş
            elif ema_s.iloc[-1] < ema_m.iloc[-1]:
                pos_score = -0.5  # Orta düşüş
            
            # 2. Fiyat vs EMA pozisyonu
            price_vs_ema_score = 0
            if current_price > ema_s.iloc[-1] > ema_m.iloc[-1]:
                price_vs_ema_score = 0.8  # Fiyat EMA'ların üstünde
            elif current_price < ema_s.iloc[-1] < ema_m.iloc[-1]:
                price_vs_ema_score = -0.8  # Fiyat EMA'ların altında
            
            # 3. EMA eğim skoru (son 3 bar)
            if len(ema_s) >= 3:
                ema_s_slope = (ema_s.iloc[-1] - ema_s.iloc[-3]) / ema_s.iloc[-3]
                slope_score = np.tanh(ema_s_slope * 100)  # [-1, 1] aralığına normalize
            else:
                slope_score = 0
            
            # Kombine skor
            final_score = (pos_score * 0.4) + (price_vs_ema_score * 0.4) + (slope_score * 0.2)
            
            logger.debug(f"AI Trend Analizi: pos={pos_score:.2f}, price_vs_ema={price_vs_ema_score:.2f}, slope={slope_score:.2f}, final={final_score:.2f}")
            
            return np.clip(final_score, -1.0, 1.0)
        except Exception as e:
            logger.warning(f"Trend gücü hesaplama hatası: {e}")
            return 0.0

    def _analyze_volume_profile(self, df: pd.DataFrame, lookback: int = 5) -> float:
        """Hacim profilini [-1, 1] aralığında analiz eder."""
        if df is None or len(df) < lookback:
            return 0.0
        try:
            recent_volumes = df['volume'].tail(lookback)
            avg_volume = recent_volumes.iloc[:-1].mean() # Son bar hariç ortalama
            last_volume = recent_volumes.iloc[-1]
            
            if avg_volume == 0: return 0.0 # Sıfıra bölme hatasını önle
            volume_change_ratio = (last_volume - avg_volume) / avg_volume # Momentum gibi
            
            # Fiyat değişimiyle ilişkilendir - GÜÇLENDİRİLMİŞ!
            price_change = df['close'].iloc[-1] - df['close'].iloc[-2] # Son 2 bar arası değişim
            price_change_pct = price_change / df['close'].iloc[-2] if df['close'].iloc[-2] != 0 else 0
            
            volume_score = 0.0
            # Daha net volume-price ilişkisi
            if price_change_pct > 0.001 and volume_change_ratio > 0.2: # %0.1+ fiyat artışı + %20+ hacim artışı
                volume_score = 1.0  # Güçlü boğa sinyali
            elif price_change_pct < -0.001 and volume_change_ratio > 0.2: # %0.1+ fiyat düşüşü + %20+ hacim artışı  
                volume_score = -1.0 # Güçlü ayı sinyali
            elif volume_change_ratio > 0.5: # Çok yüksek hacim artışı
                volume_score = 0.5 * np.sign(price_change_pct) # Fiyat yönünde orta sinyal
            
            # Hacim değişim oranını da skora yansıt - İYİLEŞTİRİLMİŞ!
            volume_momentum = np.tanh(volume_change_ratio * 2) # 2x ile daha hassas
            final_score = np.clip((volume_score * 0.7) + (volume_momentum * 0.3), -1.0, 1.0)
            
            return final_score
        except Exception as e:
            logger.warning(f"Volume analizi hatası: {e}")
            return 0.0

    def _calculate_rsi_divergence(self, df: pd.DataFrame, rsi_period: int, lookback: int) -> float:
        """RSI uyumsuzluğunu [-1, 1] aralığında hesaplar - GÜÇLENDİRİLMİŞ!"""
        if df is None or len(df) < max(rsi_period, lookback) + 1:
            return 0.0
        try:
            rsi = ta.rsi(df['close'], length=rsi_period)
            if rsi is None or rsi.isna().all() or len(rsi) < lookback:
                return 0.0

            # Daha güçlü divergence analizi
            price_end = df['close'].iloc[-1]
            price_start = df['close'].iloc[-lookback]
            rsi_end = rsi.iloc[-1]
            rsi_start = rsi.iloc[-lookback]

            if pd.isna(rsi_end) or pd.isna(rsi_start): return 0.0

            price_change_pct = (price_end - price_start) / price_start if price_start != 0 else 0
            rsi_change = rsi_end - rsi_start
            
            # Güçlü divergence koşulları
            if price_change_pct > 0.01 and rsi_change < -5: # Fiyat %1+ yükseldi ama RSI 5+ düştü
                return -0.8  # Güçlü ayı divergence
            elif price_change_pct < -0.01 and rsi_change > 5: # Fiyat %1+ düştü ama RSI 5+ yükseldi
                return 0.8   # Güçlü boğa divergence
            elif price_change_pct > 0.005 and rsi_change < -2: # Zayıf divergence
                return -0.4
            elif price_change_pct < -0.005 and rsi_change > 2:
                return 0.4
                
            return 0.0
        except Exception as e:
            logger.warning(f"RSI divergence hatası: {e}")
            return 0.0

    def _get_technical_analysis_score(self, ohlcv_df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> float:
        """Çoklu zaman dilimi dahil teknik analiz skorunu [-1, 1] aralığında hesaplar - DAHA SELEKTİF!"""
        if ohlcv_df is None or ohlcv_df.empty:
            return 0.0

        # Ana zaman dilimi analizi
        trend_main = self._calculate_trend_strength(ohlcv_df, self.ta_ema_periods['main'])
        volume_main = self._analyze_volume_profile(ohlcv_df)
        divergence_main = self._calculate_rsi_divergence(ohlcv_df, self.ta_rsi_period, self.ta_divergence_lookback)

        # Uzun zaman dilimi analizi
        ohlcv_df_long_tf = self._resample_ohlcv(ohlcv_df, timeframe=settings.AI_TA_LONG_TIMEFRAME_STR)
        trend_long = 0.0
        if ohlcv_df_long_tf is not None and len(ohlcv_df_long_tf) > max(self.ta_ema_periods['long']):
            trend_long = self._calculate_trend_strength(ohlcv_df_long_tf, self.ta_ema_periods['long'])
        
        # Ağırlıklı toplam skor - DAHA KONSERVATIF! (Pozitif bias kaldırıldı)
        current_weights = self.ta_weights
        if context and context.get("ai_ta_weights"):
            current_weights = context["ai_ta_weights"]

        score = (
            trend_main * current_weights.get('trend_main', 0.3) +
            trend_long * current_weights.get('trend_long', 0.2) +
            volume_main * current_weights.get('volume', 0.3) +
            divergence_main * current_weights.get('divergence', 0.2)
        )
        
        # KONSERVATIF yaklaşım - pozitif bias kaldırıldı!
        # Sadece gerçekten güçlü sinyallerde pozitif skor ver
        
        return np.clip(score, -1.0, 1.0)

    def _preprocess_data_for_ml(self, ohlcv_df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """ML modeli için OHLCV verisinden özellikler çıkarır. Placeholder."""
        # GERÇEK UYGULAMADA: Bu kısım çok daha karmaşık olmalı.
        # Örnek: Lagged returnler, volatilite, RSI, MACD değerleri, hacim değişimleri vb.
        # Modelinizin eğitildiği özelliklerle aynı olmalı.
        if ohlcv_df is None or len(ohlcv_df) < 20: # Örnek min veri
            logger.warning("ML için ön işleme yetersiz veri.")
            return None
        
        try:
            features = pd.DataFrame(index=[ohlcv_df.index[-1]]) # Sadece son bar için özellikler
            # Örnek basit özellikler:
            features['close_pct_change_1'] = ohlcv_df['close'].pct_change(1).iloc[-1]
            features['close_pct_change_5'] = ohlcv_df['close'].pct_change(5).iloc[-1]
            rsi_series = ta.rsi(ohlcv_df['close'], length=14)
            features['rsi_14'] = rsi_series.iloc[-1] if rsi_series is not None else pd.NA
            # ... daha fazla özellik eklenebilir ...
            
            features = features.fillna(0) # Basit NaN yönetimi
            # Modelinize uygun sayıda ve sırada özellik döndürmelisiniz.
            # logger.debug(f"ML için hazırlanan özellikler: {features.to_dict('records')}")
            return features
        except Exception as e:
            logger.error(f"ML için veri ön işleme hatası: {e}", exc_info=True)
            return None

    def _calculate_market_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Market volatility hesapla"""
        if len(df) < window:
            return 0.0
        try:
            returns = df['close'].pct_change().tail(window)
            volatility = returns.std() * np.sqrt(len(returns))  # Annualized-like
            return volatility
        except Exception as e:
            logger.warning(f"Volatility calculation error: {e}")
            return 0.0

    def _detect_unusual_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Unusual market patterns detect et"""
        patterns = {}
        
        try:
            # Volume spike detection
            if len(df) >= 20:
                avg_volume = df['volume'].tail(20).mean()
                current_volume = df['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                patterns['volume_spike'] = min(1.0, max(0.0, (volume_ratio - 1.0) / 2.0))
            
            # Price gap detection
            if len(df) >= 2:
                prev_close = df['close'].iloc[-2]
                current_open = df['open'].iloc[-1]
                gap_pct = abs(current_open - prev_close) / prev_close if prev_close > 0 else 0.0
                patterns['price_gap'] = min(1.0, gap_pct * 10)  # Scale to 0-1
            
            # Consecutive same-direction candles
            if len(df) >= 5:
                recent_closes = df['close'].tail(5)
                consecutive_up = sum(1 for i in range(1, len(recent_closes)) 
                                   if recent_closes.iloc[i] > recent_closes.iloc[i-1])
                consecutive_down = sum(1 for i in range(1, len(recent_closes)) 
                                     if recent_closes.iloc[i] < recent_closes.iloc[i-1])
                patterns['momentum_strength'] = max(consecutive_up, consecutive_down) / 4.0
            
        except Exception as e:
            logger.warning(f"Pattern detection error: {e}")
        
        return patterns

    async def assess_market_risk(self, df: pd.DataFrame, context: Optional[Dict] = None) -> MarketRiskAssessment:
        """🛡️ Advanced market risk assessment"""
        if not self.risk_assessment_enabled or df.empty:
            return MarketRiskAssessment(MarketRiskLevel.MODERATE, 0.5, {}, "Risk assessment disabled")
        
        risk_factors = {}
        
        try:
            # 1. Volatility Risk
            volatility = self._calculate_market_volatility(df)
            vol_risk = min(1.0, volatility / self.volatility_threshold)
            risk_factors['volatility'] = vol_risk
            
            # 2. Volume Analysis Risk
            patterns = self._detect_unusual_patterns(df)
            volume_risk = patterns.get('volume_spike', 0.0)
            risk_factors['volume_anomaly'] = volume_risk
            
            # 3. Price Action Risk
            price_gap_risk = patterns.get('price_gap', 0.0)
            risk_factors['price_gap'] = price_gap_risk
            
            # 4. Momentum Exhaustion Risk
            momentum_risk = patterns.get('momentum_strength', 0.0)
            if momentum_risk > 0.75:  # Too much momentum = risk
                risk_factors['momentum_exhaustion'] = momentum_risk
            else:
                risk_factors['momentum_exhaustion'] = 0.0
            
            # 5. Technical Divergence Risk
            if len(df) >= 20:
                rsi_div = self._calculate_rsi_divergence(df, 14, 10)
                divergence_risk = abs(rsi_div) if abs(rsi_div) > 0.5 else 0.0
                risk_factors['technical_divergence'] = divergence_risk
            
            # Combined Risk Score
            weights = {
                'volatility': 0.3,
                'volume_anomaly': 0.25,
                'price_gap': 0.2,
                'momentum_exhaustion': 0.15,
                'technical_divergence': 0.1
            }
            
            total_risk = sum(risk_factors.get(factor, 0.0) * weight 
                           for factor, weight in weights.items())
            
            # Risk Level Classification
            if total_risk < 0.2:
                level = MarketRiskLevel.VERY_LOW
                recommendation = "Ideal trading conditions"
            elif total_risk < 0.4:
                level = MarketRiskLevel.LOW
                recommendation = "Good trading conditions"
            elif total_risk < 0.6:
                level = MarketRiskLevel.MODERATE
                recommendation = "Normal trading conditions"
            elif total_risk < 0.8:
                level = MarketRiskLevel.HIGH
                recommendation = "Elevated risk - reduce position sizes"
            else:
                level = MarketRiskLevel.EXTREME
                recommendation = "High risk - avoid new positions"
            
            return MarketRiskAssessment(level, total_risk, risk_factors, recommendation)
            
        except Exception as e:
            logger.error(f"Market risk assessment error: {e}")
            return MarketRiskAssessment(MarketRiskLevel.MODERATE, 0.5, {}, "Assessment error")

    async def get_ai_confirmation(
        self,
        current_signal_type: str,
        ohlcv_df: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """🎯 CONFIG BASED AI confirmation - All thresholds from config!"""
        if not self.is_enabled:
            return True
        
        # Auto-approve SELL signals
        if current_signal_type.upper() == "SELL":
            logger.debug("AI: Auto-approving SELL signal")
            return True
            
        # CONFIG BASED BUY signal analysis
        if current_signal_type.upper() == "BUY":
            if ohlcv_df is None or ohlcv_df.empty:
                logger.debug("AI: No OHLCV data for analysis")
                return False
            
            if context is None:
                logger.debug("AI: No context provided")
                return False
            
            # Technical analysis score
            ta_score = self._get_technical_analysis_score(ohlcv_df, context)
            
            # CONFIG thresholds
            required_ta_score = settings.AI_CONFIRM_MIN_TA_SCORE
            
            # Context quality score
            quality_score = context.get("quality_score", 0)
            min_quality_required = context.get("min_quality_required", settings.AI_CONFIRM_MIN_QUALITY_SCORE)
            
            # Portfolio performance adjustment
            portfolio_profit_pct = context.get("portfolio_profit_pct", 0)
            
            # CONFIG dynamic threshold adjustment
            if portfolio_profit_pct < -0.05:
                required_ta_score = settings.AI_CONFIRM_LOSS_5PCT_TA_SCORE
                logger.debug(f"AI: Loss mode (-5%+), TA threshold raised to {required_ta_score}")
            elif portfolio_profit_pct < -0.02:
                required_ta_score = settings.AI_CONFIRM_LOSS_2PCT_TA_SCORE
                logger.debug(f"AI: Loss mode (-2%+), TA threshold raised to {required_ta_score}")
            elif portfolio_profit_pct < 0:
                required_ta_score = settings.AI_CONFIRM_PROFIT_TA_SCORE
            
            # CONFIG main checks
            if ta_score < required_ta_score:
                logger.debug(f"AI REJECTED: TA score {ta_score:.3f} < required {required_ta_score:.3f}")
                return False
            
            if quality_score < min_quality_required:
                logger.debug(f"AI REJECTED: Quality {quality_score} < required {min_quality_required}")
                return False
            
            # CONFIG EMA spread check
            ema_spread_1 = context.get("ema_spread_1", 0)
            ema_spread_2 = context.get("ema_spread_2", 0)
            if (ema_spread_1 < settings.AI_CONFIRM_MIN_EMA_SPREAD_1 or 
                ema_spread_2 < settings.AI_CONFIRM_MIN_EMA_SPREAD_2):
                logger.debug(f"AI REJECTED: EMA spreads too narrow ({ema_spread_1*100:.4f}%, {ema_spread_2*100:.4f}%)")
                return False
            
            # CONFIG volume check
            volume_ratio = context.get("volume_ratio", 1.0)
            if volume_ratio < settings.AI_CONFIRM_MIN_VOLUME_RATIO:
                logger.debug(f"AI REJECTED: Volume ratio {volume_ratio:.2f} < {settings.AI_CONFIRM_MIN_VOLUME_RATIO}")
                return False
            
            # CONFIG price momentum check
            price_momentum = context.get("price_momentum", 0)
            if price_momentum < settings.AI_CONFIRM_MIN_PRICE_MOMENTUM:
                logger.debug(f"AI REJECTED: Price momentum {price_momentum*100:.4f}% < {settings.AI_CONFIRM_MIN_PRICE_MOMENTUM*100:.4f}%")
                return False
            
            # CONFIG EMA momentum check
            ema_momentum = context.get("ema_momentum", 0)
            if ema_momentum < settings.AI_CONFIRM_MIN_EMA_MOMENTUM:
                logger.debug(f"AI REJECTED: EMA momentum {ema_momentum*100:.4f}% < {settings.AI_CONFIRM_MIN_EMA_MOMENTUM*100:.4f}%")
                return False
            
            # CONFIG ADX check
            adx = context.get("adx", 0)
            if adx < settings.AI_CONFIRM_MIN_ADX:
                logger.debug(f"AI REJECTED: ADX {adx:.1f} < {settings.AI_CONFIRM_MIN_ADX}")
                return False
            
            logger.info(f"🤖 AI APPROVED (CONFIG): TA={ta_score:.3f}, Q={quality_score}, Vol={volume_ratio:.1f}x, "
                       f"Price={price_momentum*100:+.3f}%, EMA_Mom={ema_momentum*100:+.3f}%, "
                       f"ADX={adx:.1f}, Portfolio={portfolio_profit_pct*100:+.1f}%")
            return True
        
        return True

    async def get_market_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """🧠 Get comprehensive market intelligence"""
        if not self.is_enabled:
            return {"status": "disabled"}
        
        intelligence = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "active"
        }
        
        try:
            # Teknik analiz
            ta_score = self._get_technical_analysis_score(df)
            intelligence["technical_score"] = ta_score
            intelligence["technical_bias"] = "bullish" if ta_score > 0.1 else "bearish" if ta_score < -0.1 else "neutral"
            
            # Risk değerlendirmesi
            risk_assessment = await self.assess_market_risk(df)
            intelligence["risk_level"] = risk_assessment.level.value
            intelligence["risk_score"] = risk_assessment.score
            intelligence["risk_factors"] = risk_assessment.factors
            intelligence["recommendation"] = risk_assessment.recommendation
            
            # Piyasa modelleri
            patterns = self._detect_unusual_patterns(df)
            intelligence["unusual_patterns"] = patterns
            
            # Volatilite analizi
            volatility = self._calculate_market_volatility(df)
            intelligence["volatility"] = volatility
            intelligence["volatility_status"] = (
                "high" if volatility > self.volatility_threshold else
                "normal" if volatility > self.volatility_threshold * 0.5 else "low"
            )
            
        except Exception as e:
            logger.error(f"Market intelligence error: {e}")
            intelligence["error"] = str(e)
        
        return intelligence

    async def get_standalone_signal(
        self,
        ohlcv_df: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> AiSignal:
        """🧠 Get standalone AI signal based on CONFIG thresholds"""
        if not self.is_enabled:
            return AiSignal.NO_OPINION
        
        if context is None:
            context = {}

        if self.operation_mode == 'technical_analysis':
            ta_score = self._get_technical_analysis_score(ohlcv_df, context)
            
            # CONFIG standalone thresholds
            ta_strong_buy_thresh = settings.AI_TA_STANDALONE_THRESH_STRONG_BUY
            ta_buy_thresh = settings.AI_TA_STANDALONE_THRESH_BUY
            ta_strong_sell_thresh = settings.AI_TA_STANDALONE_THRESH_STRONG_SELL
            ta_sell_thresh = settings.AI_TA_STANDALONE_THRESH_SELL

            logger.debug(f"🧠 Standalone AI Signal (TA Mode): Score = {ta_score:.3f}")

            if ta_score > ta_strong_buy_thresh: 
                return AiSignal.STRONG_BUY
            if ta_score > ta_buy_thresh: 
                return AiSignal.BUY
            if ta_score < ta_strong_sell_thresh: 
                return AiSignal.STRONG_SELL
            if ta_score < ta_sell_thresh: 
                return AiSignal.SELL
            return AiSignal.HOLD
        else:
            logger.warning(f"Unknown AI operation mode: {self.operation_mode}")
            return AiSignal.NO_OPINION

# Test için
async def main_test():
    # Bu testlerin çalışması için config.py'de AI_... ayarlarının olması gerekir.
    # settings.AI_ASSISTANCE_ENABLED = True # Test için
    # settings.AI_OPERATION_MODE = 'technical_analysis' # 'ml_model' veya 'technical_analysis'
    # settings.AI_MODEL_PATH = None # Eğer ml_model ise model yolu
    # settings.AI_CONFIDENCE_THRESHOLD = 0.55
    # # ... diğer AI_TA... ayarları ...

    if not settings.AI_ASSISTANCE_ENABLED:
        print("AI Desteği config'de kapalı. Test için etkinleştirin.")
        return

    print(f"Test başlatılıyor. AI Modu: {settings.AI_OPERATION_MODE}, Güven Eşiği: {settings.AI_CONFIDENCE_THRESHOLD}")

    ai_provider = AiSignalProvider()
    
    # Örnek OHLCV DataFrame oluştur
    data_size = 200 # Yeterli veri olmalı
    data = {
        'open': np.random.rand(data_size) * 10 + 100,
        'high': np.random.rand(data_size) * 2 + 110,
        'low': np.random.rand(data_size) * 2 + 98,
        'close': np.random.rand(data_size) * 10 + 100,
        'volume': np.random.rand(data_size) * 100 + 10
    }
    # Zaman damgalı index oluştur (5 saniyelik)
    index = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=data_size, freq='5S')
    sample_df = pd.DataFrame(data, index=index)
    sample_df['high'] = sample_df[['open', 'close']].max(axis=1) + np.random.rand(data_size) * 2
    sample_df['low'] = sample_df[['open', 'close']].min(axis=1) - np.random.rand(data_size) * 2
    
    print("\n--- get_ai_confirmation Testleri ---")
    buy_confirmation = await ai_provider.get_ai_confirmation("BUY", sample_df, context={"strategy": "TestMomentum"})
    print(f"BUY sinyali için AI Onayı: {buy_confirmation}")
    
    sell_confirmation = await ai_provider.get_ai_confirmation("SELL", sample_df, context={"strategy": "TestMeanRev", "ai_confidence_threshold": 0.7})
    print(f"SELL sinyali için AI Onayı (özel eşik 0.7): {sell_confirmation}")

    print("\n--- get_standalone_signal Testleri ---")
    standalone_signal = await ai_provider.get_standalone_signal(sample_df)
    print(f"Bağımsız AI Sinyali: {standalone_signal.name if standalone_signal else 'Yok'}")

if __name__ == "__main__":
    # Önce ayarların yüklendiğinden emin olun (eğer config.py ayrı çalıştırılmıyorsa)
    # from utils.config import settings # Zaten yukarıda var
    asyncio.run(main_test())