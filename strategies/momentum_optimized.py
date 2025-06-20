# strategies/momentum_optimized.py - PROFIT ENHANCED Momentum Strategy

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional, Dict, Tuple, Any, List
from datetime import datetime, timezone
import asyncio

from utils.portfolio import Portfolio, Position
from utils.config import settings
from utils.logger import logger
from utils.ai_signal_provider import AiSignalProvider
# ML Enhancement imports - Line 12'den sonra ekle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from collections import deque
import joblib
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Using traditional ML only.")



class AdvancedMLPredictor:
    """🧠 Advanced Machine Learning Prediction Engine"""
    
    def __init__(self, lookback_window=100, prediction_horizon=4):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # Model ensemble
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        
        # LSTM model (if TensorFlow available)
        self.lstm_model = None
        self.lstm_trained = False
        
        # Training data storage
        self.feature_history = deque(maxlen=1000)
        self.target_history = deque(maxlen=1000)
        self.is_trained = False
        self.training_size = 200
        
        # Model weights for ensemble
        self.model_weights = {'rf': 0.3, 'xgb': 0.4, 'gb': 0.3, 'lstm': 0.0}
        self.performance_tracking = {'rf': [], 'xgb': [], 'gb': [], 'lstm': []}
        
        logger.info("🧠 Advanced ML Predictor initialized")
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive features for ML prediction"""
        try:
            if len(df) < 50:
                return np.array([])
            
            features = []
            
            # Price features
            returns_1 = df['close'].pct_change(1).fillna(0)
            returns_3 = df['close'].pct_change(3).fillna(0)
            returns_5 = df['close'].pct_change(5).fillna(0)
            returns_10 = df['close'].pct_change(10).fillna(0)
            
            # Technical indicators
            rsi = ta.rsi(df['close'], length=14).fillna(50)
            macd = ta.macd(df['close']).iloc[:, 0].fillna(0)
            bb = ta.bbands(df['close'], length=20)
            bb_percent = bb.iloc[:, 4].fillna(0.5) if bb is not None else pd.Series([0.5] * len(df))
            
            # Volume features
            volume_sma = ta.sma(df['volume'], length=20).fillna(df['volume'].mean())
            volume_ratio = (df['volume'] / volume_sma).fillna(1.0)
            
            # Volatility features
            atr = ta.atr(df['high'], df['low'], df['close'], length=14).fillna(df['close'].std())
            volatility = returns_1.rolling(20).std().fillna(returns_1.std())
            
            # EMA features
            ema_9 = ta.ema(df['close'], length=9)
            ema_21 = ta.ema(df['close'], length=21)
            ema_50 = ta.ema(df['close'], length=50)
            
            # Feature combinations (last 20 values)
            feature_set = np.column_stack([
                returns_1.tail(20).values,
                returns_3.tail(20).values,
                returns_5.tail(20).values,
                rsi.tail(20).values / 100.0,  # Normalize
                macd.tail(20).values,
                bb_percent.tail(20).values,
                volume_ratio.tail(20).values,
                volatility.tail(20).values,
                (df['close'] / ema_9).tail(20).values,
                (df['close'] / ema_21).tail(20).values,
                (df['close'] / ema_50).tail(20).values,
                atr.tail(20).values / df['close'].tail(20).values  # Normalized ATR
            ])
            
            # Flatten to 1D feature vector
            features = feature_set.flatten()
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return np.array([])
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model for time series prediction"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                BatchNormalization(),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            return model
            
        except Exception as e:
            logger.debug(f"LSTM model creation error: {e}")
            return None
    
    def update_training_data(self, df: pd.DataFrame):
        """Update training data with new market data"""
        try:
            if len(df) < self.lookback_window:
                return
            
            features = self.extract_features(df)
            if len(features) == 0:
                return
            
            # Target: future price change (next 4 periods)
            if len(df) >= self.prediction_horizon:
                current_price = df['close'].iloc[-self.prediction_horizon]
                future_price = df['close'].iloc[-1]
                target = (future_price - current_price) / current_price
                
                self.feature_history.append(features)
                self.target_history.append(target)
                
                # Retrain models periodically
                if len(self.feature_history) >= self.training_size and len(self.feature_history) % 50 == 0:
                    self._retrain_models()
                    
        except Exception as e:
            logger.debug(f"Training data update error: {e}")
    
    def _retrain_models(self):
        """Retrain all models with recent data"""
        try:
            if len(self.feature_history) < self.training_size:
                return
            
            # Prepare training data
            X = np.array(list(self.feature_history))
            y = np.array(list(self.target_history))
            
            # Handle NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                return
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Train traditional ML models
            models = {
                'rf': self.rf_model,
                'xgb': self.xgb_model,
                'gb': self.gb_model
            }
            
            for name, model in models.items():
                try:
                    # Cross-validation performance
                    scores = []
                    for train_idx, val_idx in tscv.split(X_scaled):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = -mean_squared_error(y_val, y_pred)  # Negative MSE for maximization
                        scores.append(score)
                    
                    avg_score = np.mean(scores)
                    self.performance_tracking[name].append(avg_score)
                    
                    # Final training on all data
                    model.fit(X_scaled, y_scaled)
                    
                except Exception as e:
                    logger.debug(f"Model {name} training error: {e}")
            
            # Train LSTM if available
            if TENSORFLOW_AVAILABLE and not self.lstm_trained:
                self._train_lstm_model(X_scaled, y_scaled)
            
            # Update model weights based on performance
            self._update_model_weights()
            
            self.is_trained = True
            logger.debug(f"🧠 Models retrained with {len(X)} samples")
            
        except Exception as e:
            logger.debug(f"Model retraining error: {e}")
    
    def _train_lstm_model(self, X, y):
        """Train LSTM model"""
        try:
            if len(X) < 100:  # Need more data for LSTM
                return
            
            # Reshape for LSTM (samples, timesteps, features)
            X_lstm = X.reshape(X.shape[0], 1, X.shape[1])
            
            if self.lstm_model is None:
                self.lstm_model = self.create_lstm_model((1, X.shape[1]))
            
            if self.lstm_model is not None:
                # Train with validation split
                history = self.lstm_model.fit(
                    X_lstm, y,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                self.lstm_trained = True
                self.model_weights['lstm'] = 0.2  # Enable LSTM in ensemble
                logger.debug("🧠 LSTM model trained successfully")
                
        except Exception as e:
            logger.debug(f"LSTM training error: {e}")
    
    def _update_model_weights(self):
        """Update ensemble weights based on recent performance"""
        try:
            if not all(self.performance_tracking[name] for name in ['rf', 'xgb', 'gb']):
                return
            
            # Get recent performance (last 5 scores)
            recent_performance = {}
            for name in ['rf', 'xgb', 'gb']:
                scores = self.performance_tracking[name][-5:]
                recent_performance[name] = np.mean(scores) if scores else -1
            
            # Softmax for weight calculation
            scores = np.array(list(recent_performance.values()))
            scores = scores - np.max(scores)  # Numerical stability
            weights = np.exp(scores) / np.sum(np.exp(scores))
            
            # Update weights (excluding LSTM for now)
            total_traditional = sum(weights)
            self.model_weights['rf'] = weights[0] * 0.8  # 80% for traditional models
            self.model_weights['xgb'] = weights[1] * 0.8
            self.model_weights['gb'] = weights[2] * 0.8
            # LSTM keeps its weight or 0 if not trained
            
            logger.debug(f"🧠 Model weights updated: {self.model_weights}")
            
        except Exception as e:
            logger.debug(f"Weight update error: {e}")
    
    def predict_price_movement(self, df: pd.DataFrame) -> Dict[str, float]:
        """Predict future price movement using ensemble"""
        try:
            if not self.is_trained or len(df) < self.lookback_window:
                return {"prediction": 0.0, "confidence": 0.0, "direction": "NEUTRAL"}
            
            features = self.extract_features(df)
            if len(features) == 0:
                return {"prediction": 0.0, "confidence": 0.0, "direction": "NEUTRAL"}
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Get predictions from all models
            predictions = {}
            
            try:
                predictions['rf'] = self.rf_model.predict(features_scaled)[0]
            except:
                predictions['rf'] = 0.0
            
            try:
                predictions['xgb'] = self.xgb_model.predict(features_scaled)[0]
            except:
                predictions['xgb'] = 0.0
            
            try:
                predictions['gb'] = self.gb_model.predict(features_scaled)[0]
            except:
                predictions['gb'] = 0.0
            
            # LSTM prediction
            if self.lstm_model is not None and self.lstm_trained:
                try:
                    features_lstm = features_scaled.reshape(1, 1, -1)
                    predictions['lstm'] = self.lstm_model.predict(features_lstm, verbose=0)[0][0]
                except:
                    predictions['lstm'] = 0.0
            else:
                predictions['lstm'] = 0.0
            
            # Ensemble prediction
            ensemble_prediction = sum(
                predictions[model] * weight 
                for model, weight in self.model_weights.items()
            )
            
            # Scale back to original space
            try:
                ensemble_prediction = self.target_scaler.inverse_transform([[ensemble_prediction]])[0][0]
            except:
                pass  # Use unscaled prediction if scaler fails
            
            # Calculate confidence based on model agreement
            pred_values = list(predictions.values())
            confidence = 1.0 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8))
            confidence = max(0.0, min(1.0, confidence))
            
            # Determine direction
            if ensemble_prediction > 0.005:  # > 0.5% positive
                direction = "UP"
            elif ensemble_prediction < -0.005:  # < -0.5% negative
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
            
            return {
                "prediction": ensemble_prediction,
                "confidence": confidence,
                "direction": direction,
                "individual_predictions": predictions,
                "model_weights": self.model_weights.copy()
            }
            
        except Exception as e:
            logger.debug(f"ML prediction error: {e}")
            return {"prediction": 0.0, "confidence": 0.0, "direction": "NEUTRAL"}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        try:
            if not self.is_trained:
                return {}
            
            importance_dict = {}
            
            # Random Forest feature importance
            if hasattr(self.rf_model, 'feature_importances_'):
                rf_importance = self.rf_model.feature_importances_
                importance_dict['rf'] = rf_importance.tolist()
            
            # XGBoost feature importance
            if hasattr(self.xgb_model, 'feature_importances_'):
                xgb_importance = self.xgb_model.feature_importances_
                importance_dict['xgb'] = xgb_importance.tolist()
            
            return importance_dict
            
        except Exception as e:
            logger.debug(f"Feature importance error: {e}")
            return {}
        

class EnhancedMomentumStrategy:
    """🚀 PROFIT ENHANCED Momentum Strategy with Advanced Features"""
    
    def __init__(
        self, 
        portfolio: Portfolio, 
        symbol: str = "BTC/USDT",
        # Technical Indicators (optimized parameters from config)
        ema_short: Optional[int] = None,
        ema_medium: Optional[int] = None,
        ema_long: Optional[int] = None,
        rsi_period: Optional[int] = None,
        adx_period: Optional[int] = None,
        atr_period: Optional[int] = None,
        volume_sma_period: Optional[int] = None,
        
        # Position Management (enhanced)
        max_positions: Optional[int] = None,
        base_position_size_pct: Optional[float] = None,
        min_position_usdt: Optional[float] = None,
        max_position_usdt: Optional[float] = None,
        
        # Performance Based Sizing (enhanced)
        size_high_profit_pct: Optional[float] = None,
        size_good_profit_pct: Optional[float] = None,
        size_normal_profit_pct: Optional[float] = None,
        size_breakeven_pct: Optional[float] = None,
        size_loss_pct: Optional[float] = None,
        size_max_balance_pct: Optional[float] = None,
        
        # Performance Thresholds
        perf_high_profit_threshold: Optional[float] = None,
        perf_good_profit_threshold: Optional[float] = None,
        perf_normal_profit_threshold: Optional[float] = None,
        perf_breakeven_threshold: Optional[float] = None,
        
        # Risk Management (enhanced)
        max_loss_pct: Optional[float] = None,
        min_profit_target_usdt: Optional[float] = None,
        quick_profit_threshold_usdt: Optional[float] = None,
        max_hold_minutes: Optional[int] = None,
        breakeven_minutes: Optional[int] = None,
        
        # Buy Conditions (enhanced quality)
        buy_min_quality_score: Optional[int] = None,
        buy_min_ema_spread_1: Optional[float] = None,
        buy_min_ema_spread_2: Optional[float] = None,
        
        # All other parameters as before...
        **kwargs
    ):
        self.strategy_name = "EnhancedMomentum"
        self.portfolio = portfolio
        self.symbol = symbol if symbol else settings.SYMBOL
        
        # Load config parameters with enhanced defaults
        self.ema_short = ema_short if ema_short is not None else settings.MOMENTUM_EMA_SHORT
        self.ema_medium = ema_medium if ema_medium is not None else settings.MOMENTUM_EMA_MEDIUM
        self.ema_long = ema_long if ema_long is not None else settings.MOMENTUM_EMA_LONG
        self.rsi_period = rsi_period if rsi_period is not None else settings.MOMENTUM_RSI_PERIOD
        self.adx_period = adx_period if adx_period is not None else settings.MOMENTUM_ADX_PERIOD
        self.atr_period = atr_period if atr_period is not None else settings.MOMENTUM_ATR_PERIOD
        self.volume_sma_period = volume_sma_period if volume_sma_period is not None else settings.MOMENTUM_VOLUME_SMA_PERIOD
        
        # Enhanced Position Management
        self.max_positions = max_positions if max_positions is not None else settings.MOMENTUM_MAX_POSITIONS
        self.base_position_pct = base_position_size_pct if base_position_size_pct is not None else settings.MOMENTUM_BASE_POSITION_SIZE_PCT
        self.min_position_usdt = min_position_usdt if min_position_usdt is not None else settings.MOMENTUM_MIN_POSITION_USDT
        self.max_position_usdt = max_position_usdt if max_position_usdt is not None else settings.MOMENTUM_MAX_POSITION_USDT
        
        # Load all other parameters from config (keeping the same pattern as original)
        self.size_high_profit_pct = size_high_profit_pct if size_high_profit_pct is not None else settings.MOMENTUM_SIZE_HIGH_PROFIT_PCT
        self.size_good_profit_pct = size_good_profit_pct if size_good_profit_pct is not None else settings.MOMENTUM_SIZE_GOOD_PROFIT_PCT
        self.size_normal_profit_pct = size_normal_profit_pct if size_normal_profit_pct is not None else settings.MOMENTUM_SIZE_NORMAL_PROFIT_PCT
        self.size_breakeven_pct = size_breakeven_pct if size_breakeven_pct is not None else settings.MOMENTUM_SIZE_BREAKEVEN_PCT
        self.size_loss_pct = size_loss_pct if size_loss_pct is not None else settings.MOMENTUM_SIZE_LOSS_PCT
        self.size_max_balance_pct = size_max_balance_pct if size_max_balance_pct is not None else settings.MOMENTUM_SIZE_MAX_BALANCE_PCT
        
        self.perf_high_profit_threshold = perf_high_profit_threshold if perf_high_profit_threshold is not None else settings.MOMENTUM_PERF_HIGH_PROFIT_THRESHOLD
        self.perf_good_profit_threshold = perf_good_profit_threshold if perf_good_profit_threshold is not None else settings.MOMENTUM_PERF_GOOD_PROFIT_THRESHOLD
        self.perf_normal_profit_threshold = perf_normal_profit_threshold if perf_normal_profit_threshold is not None else settings.MOMENTUM_PERF_NORMAL_PROFIT_THRESHOLD
        self.perf_breakeven_threshold = perf_breakeven_threshold if perf_breakeven_threshold is not None else settings.MOMENTUM_PERF_BREAKEVEN_THRESHOLD
        
        self.max_loss_pct = max_loss_pct if max_loss_pct is not None else settings.MOMENTUM_MAX_LOSS_PCT
        self.min_profit_target_usdt = min_profit_target_usdt if min_profit_target_usdt is not None else settings.MOMENTUM_MIN_PROFIT_TARGET_USDT
        self.quick_profit_threshold_usdt = quick_profit_threshold_usdt if quick_profit_threshold_usdt is not None else settings.MOMENTUM_QUICK_PROFIT_THRESHOLD_USDT
        self.max_hold_minutes = max_hold_minutes if max_hold_minutes is not None else settings.MOMENTUM_MAX_HOLD_MINUTES
        self.breakeven_minutes = breakeven_minutes if breakeven_minutes is not None else settings.MOMENTUM_BREAKEVEN_MINUTES
        
        self.buy_min_quality_score = buy_min_quality_score if buy_min_quality_score is not None else settings.MOMENTUM_BUY_MIN_QUALITY_SCORE
        self.buy_min_ema_spread_1 = buy_min_ema_spread_1 if buy_min_ema_spread_1 is not None else settings.MOMENTUM_BUY_MIN_EMA_SPREAD_1
        self.buy_min_ema_spread_2 = buy_min_ema_spread_2 if buy_min_ema_spread_2 is not None else settings.MOMENTUM_BUY_MIN_EMA_SPREAD_2
        
        # Load remaining parameters from kwargs or config (keeping original structure)
        self._load_remaining_config_parameters(kwargs)
        
        # Enhanced features
        self.last_trade_time = None
        self.position_entry_reasons = {}
        self.market_regime_cache = {"regime": "UNKNOWN", "timestamp": None, "confidence": 0.0}
        self.quality_score_history = []

        # 🧠 ML PREDICTOR INTEGRATION
        self.ml_predictor = AdvancedMLPredictor(
            lookback_window=100,
            prediction_horizon=4
        )
        self.ml_predictions_history = deque(maxlen=500)
        self.ml_enabled = getattr(settings, 'MOMENTUM_ML_ENABLED', True)
        
        logger.info(f"🧠 ML Integration: {'ENABLED' if self.ml_enabled else 'DISABLED'}")
        
        # AI Provider with optimized parameters
        ai_param_overrides = self._create_ai_overrides()
        self.ai_provider = AiSignalProvider(overrides=ai_param_overrides) if settings.AI_ASSISTANCE_ENABLED else None
        
        logger.info(f"🚀 {self.strategy_name} Strategy initialized with PROFIT ENHANCEMENTS")
        logger.info(f"   📊 Technical: EMA({self.ema_short},{self.ema_medium},{self.ema_long}), RSI({self.rsi_period}), ADX({self.adx_period})")
        logger.info(f"   💰 Position: {self.base_position_pct}% base, ${self.min_position_usdt}-${self.max_position_usdt}, Max: {self.max_positions}")
        logger.info(f"   🎯 Quality Min: {self.buy_min_quality_score}, AI: {'ENHANCED' if self.ai_provider and self.ai_provider.is_enabled else 'OFF'}")
        logger.info(f"   🚀 Profit Targets: Quick=${self.quick_profit_threshold_usdt}, Min=${self.min_profit_target_usdt}")
    
    def _load_remaining_config_parameters(self, kwargs):
        """Load remaining config parameters (keeping original structure)"""
        # Buy momentum parameters
        self.buy_ema_mom_excellent = kwargs.get("buy_ema_mom_excellent", settings.MOMENTUM_BUY_EMA_MOM_EXCELLENT)
        self.buy_ema_mom_good = kwargs.get("buy_ema_mom_good", settings.MOMENTUM_BUY_EMA_MOM_GOOD)
        self.buy_ema_mom_decent = kwargs.get("buy_ema_mom_decent", settings.MOMENTUM_BUY_EMA_MOM_DECENT)
        self.buy_ema_mom_min = kwargs.get("buy_ema_mom_min", settings.MOMENTUM_BUY_EMA_MOM_MIN)
        
        # RSI parameters
        self.buy_rsi_excellent_min = kwargs.get("buy_rsi_excellent_min", settings.MOMENTUM_BUY_RSI_EXCELLENT_MIN)
        self.buy_rsi_excellent_max = kwargs.get("buy_rsi_excellent_max", settings.MOMENTUM_BUY_RSI_EXCELLENT_MAX)
        self.buy_rsi_good_min = kwargs.get("buy_rsi_good_min", settings.MOMENTUM_BUY_RSI_GOOD_MIN)
        self.buy_rsi_good_max = kwargs.get("buy_rsi_good_max", settings.MOMENTUM_BUY_RSI_GOOD_MAX)
        self.buy_rsi_extreme_min = kwargs.get("buy_rsi_extreme_min", settings.MOMENTUM_BUY_RSI_EXTREME_MIN)
        self.buy_rsi_extreme_max = kwargs.get("buy_rsi_extreme_max", settings.MOMENTUM_BUY_RSI_EXTREME_MAX)
        
        # ADX parameters
        self.buy_adx_excellent = kwargs.get("buy_adx_excellent", settings.MOMENTUM_BUY_ADX_EXCELLENT)
        self.buy_adx_good = kwargs.get("buy_adx_good", settings.MOMENTUM_BUY_ADX_GOOD)
        self.buy_adx_decent = kwargs.get("buy_adx_decent", settings.MOMENTUM_BUY_ADX_DECENT)
        
        # Volume parameters
        self.buy_volume_excellent = kwargs.get("buy_volume_excellent", settings.MOMENTUM_BUY_VOLUME_EXCELLENT)
        self.buy_volume_good = kwargs.get("buy_volume_good", settings.MOMENTUM_BUY_VOLUME_GOOD)
        self.buy_volume_decent = kwargs.get("buy_volume_decent", settings.MOMENTUM_BUY_VOLUME_DECENT)
        
        # Price momentum
        self.buy_price_mom_excellent = kwargs.get("buy_price_mom_excellent", settings.MOMENTUM_BUY_PRICE_MOM_EXCELLENT)
        self.buy_price_mom_good = kwargs.get("buy_price_mom_good", settings.MOMENTUM_BUY_PRICE_MOM_GOOD)
        self.buy_price_mom_decent = kwargs.get("buy_price_mom_decent", settings.MOMENTUM_BUY_PRICE_MOM_DECENT)
        
        # Sell parameters (load from config, keeping original structure)
        self.sell_min_hold_minutes = kwargs.get("sell_min_hold_minutes", settings.MOMENTUM_SELL_MIN_HOLD_MINUTES)
        self.sell_catastrophic_loss_pct = kwargs.get("sell_catastrophic_loss_pct", settings.MOMENTUM_SELL_CATASTROPHIC_LOSS_PCT)
        
        # Premium levels
        self.sell_premium_excellent = kwargs.get("sell_premium_excellent", settings.MOMENTUM_SELL_PREMIUM_EXCELLENT)
        self.sell_premium_great = kwargs.get("sell_premium_great", settings.MOMENTUM_SELL_PREMIUM_GREAT)
        self.sell_premium_good = kwargs.get("sell_premium_good", settings.MOMENTUM_SELL_PREMIUM_GOOD)
        
        # Phase parameters (all phases)
        self.sell_phase1_excellent = kwargs.get("sell_phase1_excellent", settings.MOMENTUM_SELL_PHASE1_EXCELLENT)
        self.sell_phase1_good = kwargs.get("sell_phase1_good", settings.MOMENTUM_SELL_PHASE1_GOOD)
        self.sell_phase1_loss_protection = kwargs.get("sell_phase1_loss_protection", settings.MOMENTUM_SELL_PHASE1_LOSS_PROTECTION)
        
        self.sell_phase2_excellent = kwargs.get("sell_phase2_excellent", settings.MOMENTUM_SELL_PHASE2_EXCELLENT)
        self.sell_phase2_good = kwargs.get("sell_phase2_good", settings.MOMENTUM_SELL_PHASE2_GOOD)
        self.sell_phase2_decent = kwargs.get("sell_phase2_decent", settings.MOMENTUM_SELL_PHASE2_DECENT)
        self.sell_phase2_loss_protection = kwargs.get("sell_phase2_loss_protection", settings.MOMENTUM_SELL_PHASE2_LOSS_PROTECTION)
        
        self.sell_phase3_excellent = kwargs.get("sell_phase3_excellent", settings.MOMENTUM_SELL_PHASE3_EXCELLENT)
        self.sell_phase3_good = kwargs.get("sell_phase3_good", settings.MOMENTUM_SELL_PHASE3_GOOD)
        self.sell_phase3_decent = kwargs.get("sell_phase3_decent", settings.MOMENTUM_SELL_PHASE3_DECENT)
        self.sell_phase3_breakeven_min = kwargs.get("sell_phase3_breakeven_min", settings.MOMENTUM_SELL_PHASE3_BREAKEVEN_MIN)
        self.sell_phase3_breakeven_max = kwargs.get("sell_phase3_breakeven_max", settings.MOMENTUM_SELL_PHASE3_BREAKEVEN_MAX)
        self.sell_phase3_loss_protection = kwargs.get("sell_phase3_loss_protection", settings.MOMENTUM_SELL_PHASE3_LOSS_PROTECTION)
        
        self.sell_phase4_excellent = kwargs.get("sell_phase4_excellent", settings.MOMENTUM_SELL_PHASE4_EXCELLENT)
        self.sell_phase4_good = kwargs.get("sell_phase4_good", settings.MOMENTUM_SELL_PHASE4_GOOD)
        self.sell_phase4_minimal = kwargs.get("sell_phase4_minimal", settings.MOMENTUM_SELL_PHASE4_MINIMAL)
        self.sell_phase4_breakeven_min = kwargs.get("sell_phase4_breakeven_min", settings.MOMENTUM_SELL_PHASE4_BREAKEVEN_MIN)
        self.sell_phase4_breakeven_max = kwargs.get("sell_phase4_breakeven_max", settings.MOMENTUM_SELL_PHASE4_BREAKEVEN_MAX)
        self.sell_phase4_force_exit_minutes = kwargs.get("sell_phase4_force_exit_minutes", settings.MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES)
        
        # Risk and technical sell
        self.sell_loss_multiplier = kwargs.get("sell_loss_multiplier", settings.MOMENTUM_SELL_LOSS_MULTIPLIER)
        self.sell_tech_min_minutes = kwargs.get("sell_tech_min_minutes", settings.MOMENTUM_SELL_TECH_MIN_MINUTES)
        self.sell_tech_min_loss = kwargs.get("sell_tech_min_loss", settings.MOMENTUM_SELL_TECH_MIN_LOSS)
        self.sell_tech_rsi_extreme = kwargs.get("sell_tech_rsi_extreme", settings.MOMENTUM_SELL_TECH_RSI_EXTREME)
        
        # Wait times
        self.wait_profit_5pct = kwargs.get("wait_profit_5pct", settings.MOMENTUM_WAIT_PROFIT_5PCT)
        self.wait_profit_2pct = kwargs.get("wait_profit_2pct", settings.MOMENTUM_WAIT_PROFIT_2PCT)
        self.wait_breakeven = kwargs.get("wait_breakeven", settings.MOMENTUM_WAIT_BREAKEVEN)
        self.wait_loss = kwargs.get("wait_loss", settings.MOMENTUM_WAIT_LOSS)
        
        # AI parameters
        self.ai_confidence_threshold = kwargs.get("ai_confidence_threshold", settings.AI_CONFIDENCE_THRESHOLD)
        self.ai_momentum_confidence_override = kwargs.get("ai_momentum_confidence_override", settings.AI_MOMENTUM_CONFIDENCE_OVERRIDE)
    
    def _create_ai_overrides(self) -> Dict[str, Any]:
        """Create AI parameter overrides for enhanced performance"""
        return {
            "ai_assistance_enabled": settings.AI_ASSISTANCE_ENABLED,
            "ai_confidence_threshold": self.ai_confidence_threshold,
            "ai_momentum_confidence_override": self.ai_momentum_confidence_override,
            "ai_weight_trend_main": settings.AI_TA_WEIGHT_TREND_MAIN,
            "ai_weight_trend_long": settings.AI_TA_WEIGHT_TREND_LONG,
            "ai_weight_volume": settings.AI_TA_WEIGHT_VOLUME,
            "ai_weight_divergence": settings.AI_TA_WEIGHT_DIVERGENCE,
            "ai_standalone_thresh_strong_buy": settings.AI_TA_STANDALONE_THRESH_STRONG_BUY,
            "ai_standalone_thresh_buy": settings.AI_TA_STANDALONE_THRESH_BUY,
            "ai_standalone_thresh_sell": settings.AI_TA_STANDALONE_THRESH_SELL,
            "ai_standalone_thresh_strong_sell": settings.AI_TA_STANDALONE_THRESH_STRONG_SELL,
        }

    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """🚀 Enhanced indicator calculation with multi-timeframe analysis"""
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
                logger.error(f"[{self.strategy_name}] Invalid parameter: '{name}' = '{value}'")
                return None
        
        min_data_length = max(self.ema_long, self.rsi_period, self.adx_period, self.volume_sma_period, 50)
        if len(df) < min_data_length:
            logger.debug(f"[{self.strategy_name}] Insufficient data: {len(df)} bars, need: {min_data_length}")
            return None
            
        try:
            df_copy = df.copy()
            indicators = pd.DataFrame(index=df_copy.index)
            
            # Basic OHLCV
            indicators['close'] = df_copy['close']
            indicators['volume'] = df_copy['volume']
            indicators['high'] = df_copy['high'] 
            indicators['low'] = df_copy['low']
            
            # 🚀 ENHANCED EMA SYSTEM
            indicators['ema_short'] = ta.ema(df_copy['close'], length=self.ema_short)
            indicators['ema_medium'] = ta.ema(df_copy['close'], length=self.ema_medium)
            indicators['ema_long'] = ta.ema(df_copy['close'], length=self.ema_long)
            
            # EMA distances and momentum
            indicators['ema_distance_ratio'] = (indicators['ema_short'] - indicators['ema_long']) / indicators['ema_long']
            indicators['ema_momentum'] = indicators['ema_short'].pct_change(1)
            indicators['ema_acceleration'] = indicators['ema_momentum'].diff()
            
            # RSI with enhanced analysis
            indicators['rsi'] = ta.rsi(df_copy['close'], length=self.rsi_period)
            indicators['rsi_momentum'] = indicators['rsi'].diff()
            indicators['rsi_smoothed'] = indicators['rsi'].rolling(window=3).mean()
            
            # ADX with directional indicators
            adx_result = ta.adx(df_copy['high'], df_copy['low'], df_copy['close'], length=self.adx_period)
            if adx_result is not None and not adx_result.empty:
                indicators['adx'] = adx_result.iloc[:, 0]
                if adx_result.shape[1] >= 3:
                    indicators['di_plus'] = adx_result.iloc[:, 1]
                    indicators['di_minus'] = adx_result.iloc[:, 2]
            else:
                indicators['adx'] = 20.0
                indicators['di_plus'] = 25.0
                indicators['di_minus'] = 25.0
            
            # MACD with signal
            macd_result = ta.macd(df_copy['close'])
            if macd_result is not None and not macd_result.empty:
                indicators['macd'] = macd_result.iloc[:, 0]
                indicators['macd_signal'] = macd_result.iloc[:, 1]
                indicators['macd_hist'] = macd_result.iloc[:, 2]
                indicators['macd_momentum'] = indicators['macd_hist'].diff()
            else:
                indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = 0.0, 0.0, 0.0
                indicators['macd_momentum'] = 0.0
            
            # 🚀 ENHANCED VOLUME ANALYSIS
            indicators['volume_sma'] = ta.sma(df_copy['volume'], length=self.volume_sma_period)
            indicators['volume_ratio'] = (indicators['volume'] / indicators['volume_sma'].replace(0, 1e-9)).fillna(1.0)
            indicators['volume_momentum'] = indicators['volume'].pct_change(1)
            indicators['volume_trend'] = indicators['volume_sma'].pct_change(3)
            
            # Price action analysis
            indicators['price_momentum_1'] = df_copy['close'].pct_change(1)
            indicators['price_momentum_3'] = df_copy['close'].pct_change(3)
            indicators['price_momentum_5'] = df_copy['close'].pct_change(5)
            
            # Volatility measures
            indicators['atr'] = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], length=self.atr_period)
            indicators['volatility'] = df_copy['close'].rolling(window=20).std()
            indicators['volatility_ratio'] = indicators['atr'] / df_copy['close']
            
            # Support/Resistance levels
            indicators['resistance'] = df_copy['high'].rolling(window=20).max()
            indicators['support'] = df_copy['low'].rolling(window=20).min()
            indicators['support_distance'] = (df_copy['close'] - indicators['support']) / indicators['support']
            indicators['resistance_distance'] = (indicators['resistance'] - df_copy['close']) / df_copy['close']
            
            # 🚀 MARKET STRUCTURE ANALYSIS
            indicators['higher_highs'] = (df_copy['high'] > df_copy['high'].shift(1)).rolling(window=5).sum()
            indicators['higher_lows'] = (df_copy['low'] > df_copy['low'].shift(1)).rolling(window=5).sum()
            indicators['trend_strength'] = (indicators['higher_highs'] + indicators['higher_lows']) / 10
            
            return indicators.tail(3)  # Return last 3 bars for analysis
            
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"🛑 [{self.strategy_name}] Indicator calculation interrupted")
            raise 
        except Exception as e:
            logger.error(f"[{self.strategy_name}] Indicator calculation error: {e}", exc_info=True)
            return None

    def detect_market_regime(self, indicators: pd.DataFrame) -> Dict[str, Any]:
        """🧠 Advanced market regime detection"""
        if indicators is None or indicators.empty:
            return {"regime": "UNKNOWN", "confidence": 0.0, "characteristics": []}
        
        try:
            current = indicators.iloc[-1]
            
            # Trend strength analysis
            ema_alignment = (current['ema_short'] > current['ema_medium'] > current['ema_long'])
            adx_strength = current.get('adx', 20)
            trend_strength = current.get('trend_strength', 0.5)
            
            # Volatility analysis
            volatility_ratio = current.get('volatility_ratio', 0.02)
            volume_momentum = current.get('volume_momentum', 0)
            
            # Momentum convergence
            price_momentum = current.get('price_momentum_3', 0)
            ema_momentum = current.get('ema_momentum', 0)
            macd_momentum = current.get('macd_momentum', 0)
            
            characteristics = []
            confidence = 0.0
            
            # STRONG TRENDING regime
            if (ema_alignment and adx_strength > 25 and trend_strength > 0.6 and 
                abs(price_momentum) > 0.01):
                regime = "STRONG_TRENDING"
                confidence = min(0.95, 0.6 + (adx_strength - 25) * 0.01 + trend_strength * 0.3)
                characteristics = ["Strong EMA alignment", "High ADX", "Consistent momentum"]
                
            # WEAK TRENDING regime  
            elif (ema_alignment and adx_strength > 18 and trend_strength > 0.4):
                regime = "WEAK_TRENDING"
                confidence = 0.4 + (adx_strength - 18) * 0.02 + trend_strength * 0.2
                characteristics = ["EMA alignment", "Moderate ADX", "Developing momentum"]
                
            # VOLATILE regime
            elif (volatility_ratio > 0.03 and abs(volume_momentum) > 0.5):
                regime = "VOLATILE"
                confidence = min(0.8, 0.3 + volatility_ratio * 10 + abs(volume_momentum) * 0.3)
                characteristics = ["High volatility", "Volume spikes", "Erratic price action"]
                
            # SIDEWAYS regime
            elif (abs(price_momentum) < 0.005 and adx_strength < 20):
                regime = "SIDEWAYS"
                confidence = 0.7 - abs(price_momentum) * 50 - max(0, adx_strength - 15) * 0.02
                characteristics = ["Low momentum", "Weak trend", "Range-bound"]
                
            else:
                regime = "TRANSITIONAL"
                confidence = 0.3
                characteristics = ["Mixed signals", "Uncertain direction"]
            
            # Cache the result
            regime_data = {
                "regime": regime,
                "confidence": round(confidence, 3),
                "characteristics": characteristics,
                "adx": adx_strength,
                "trend_strength": trend_strength,
                "volatility_ratio": volatility_ratio,
                "ema_alignment": ema_alignment
            }
            
            self.market_regime_cache = {
                **regime_data,
                "timestamp": datetime.now(timezone.utc)
            }
            
            return regime_data
            
        except Exception as e:
            logger.error(f"Market regime detection error: {e}")
            return {"regime": "UNKNOWN", "confidence": 0.0, "characteristics": ["Error in detection"]}

    def calculate_enhanced_quality_score(self, indicators: pd.DataFrame, market_regime: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """🎯 Enhanced quality scoring system with regime awareness"""
        if indicators is None or len(indicators) < 2:
            return 0, {"error": "Insufficient data"}
        
        current = indicators.iloc[-1]
        previous = indicators.iloc[-2]
        
        quality_score = 0
        score_breakdown = {}
        
        try:
            # 1. EMA SYSTEM ANALYSIS (Max: 5 points)
            ema_short, ema_medium, ema_long = current.get('ema_short'), current.get('ema_medium'), current.get('ema_long')
            
            if pd.notna(ema_short) and pd.notna(ema_medium) and pd.notna(ema_long):
                if ema_short > ema_medium > ema_long:
                    ema_distance_ratio = current.get('ema_distance_ratio', 0)
                    if ema_distance_ratio > 0.02:  # Strong separation
                        quality_score += 3
                        score_breakdown['ema_strong_alignment'] = 3
                    elif ema_distance_ratio > 0.01:  # Good separation
                        quality_score += 2
                        score_breakdown['ema_good_alignment'] = 2
                    else:  # Basic alignment
                        quality_score += 1
                        score_breakdown['ema_basic_alignment'] = 1
                    
                    # EMA momentum bonus
                    ema_momentum = current.get('ema_momentum', 0)
                    if ema_momentum > 0.001:
                        quality_score += 2
                        score_breakdown['ema_momentum_strong'] = 2
                    elif ema_momentum > 0.0005:
                        quality_score += 1
                        score_breakdown['ema_momentum_good'] = 1
            
            # 2. RSI ANALYSIS (Max: 3 points)
            rsi = current.get('rsi')
            if pd.notna(rsi):
                if self.buy_rsi_excellent_min <= rsi <= self.buy_rsi_excellent_max:
                    quality_score += 3
                    score_breakdown['rsi_excellent'] = 3
                elif self.buy_rsi_good_min <= rsi <= self.buy_rsi_good_max:
                    quality_score += 2
                    score_breakdown['rsi_good'] = 2
                elif self.buy_rsi_extreme_min < rsi < self.buy_rsi_extreme_max:
                    quality_score += 1
                    score_breakdown['rsi_acceptable'] = 1
            
            # 3. ADX TREND STRENGTH (Max: 3 points)
            adx = current.get('adx')
            if pd.notna(adx):
                if adx > self.buy_adx_excellent:
                    quality_score += 3
                    score_breakdown['adx_excellent'] = 3
                elif adx > self.buy_adx_good:
                    quality_score += 2
                    score_breakdown['adx_good'] = 2
                elif adx > self.buy_adx_decent:
                    quality_score += 1
                    score_breakdown['adx_decent'] = 1
            
            # 4. VOLUME CONFIRMATION (Max: 3 points)
            volume_ratio = current.get('volume_ratio', 1.0)
            if volume_ratio > self.buy_volume_excellent:
                quality_score += 3
                score_breakdown['volume_excellent'] = 3
            elif volume_ratio > self.buy_volume_good:
                quality_score += 2
                score_breakdown['volume_good'] = 2
            elif volume_ratio > self.buy_volume_decent:
                quality_score += 1
                score_breakdown['volume_decent'] = 1
            
            # 5. PRICE MOMENTUM (Max: 3 points)
            price_momentum = current.get('price_momentum_3', 0)
            if price_momentum > self.buy_price_mom_excellent:
                quality_score += 3
                score_breakdown['price_momentum_excellent'] = 3
            elif price_momentum > self.buy_price_mom_good:
                quality_score += 2
                score_breakdown['price_momentum_good'] = 2
            elif price_momentum > self.buy_price_mom_decent:
                quality_score += 1
                score_breakdown['price_momentum_decent'] = 1
            
            # 6. MACD CONFIRMATION (Max: 2 points)
            macd_hist = current.get('macd_hist', 0)
            macd_momentum = current.get('macd_momentum', 0)
            if macd_hist > 0 and macd_momentum > 0:
                quality_score += 2
                score_breakdown['macd_strong'] = 2
            elif macd_hist > 0:
                quality_score += 1
                score_breakdown['macd_positive'] = 1
            
            # 7. MARKET REGIME BONUS (Max: 3 points)
            regime = market_regime.get('regime', 'UNKNOWN')
            regime_confidence = market_regime.get('confidence', 0)
            
            if regime == "STRONG_TRENDING" and regime_confidence > 0.7:
                quality_score += 3
                score_breakdown['regime_strong_trend'] = 3
            elif regime == "WEAK_TRENDING" and regime_confidence > 0.5:
                quality_score += 2
                score_breakdown['regime_weak_trend'] = 2
            elif regime == "TRANSITIONAL" and regime_confidence > 0.4:
                quality_score += 1
                score_breakdown['regime_transitional'] = 1
            
            # 8. SUPPORT/RESISTANCE BONUS (Max: 2 points)
            support_distance = current.get('support_distance', 0)
            resistance_distance = current.get('resistance_distance', 0)
            
            if support_distance > 0.02 and resistance_distance > 0.02:  # Away from both
                quality_score += 2
                score_breakdown['clear_levels'] = 2
            elif support_distance > 0.01:  # Away from support
                quality_score += 1
                score_breakdown['above_support'] = 1
            
            # Maximum quality score cap
            quality_score = min(quality_score, 25)  # Increased max for enhanced scoring
            
            score_breakdown['total_score'] = quality_score
            score_breakdown['market_regime'] = regime
            score_breakdown['regime_confidence'] = regime_confidence
            
            return quality_score, score_breakdown
            
        except Exception as e:
            logger.error(f"Quality score calculation error: {e}")
            return 0, {"error": str(e)}

    def calculate_dynamic_position_size(self, current_price: float, quality_score: int, market_regime: Dict[str, Any]) -> float:
        """💰 Dynamic position sizing with Kelly Criterion and regime awareness"""
        try:
            available_usdt = self.portfolio.get_available_usdt()
            initial_capital = self.portfolio.initial_capital_usdt
            current_portfolio_value = self.portfolio.get_total_portfolio_value_usdt(current_price)
            
            # Performance-based base sizing (from config)
            profit_pct = 0.0
            if initial_capital > 0:
                profit_pct = (current_portfolio_value - initial_capital) / initial_capital
            
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
            
            # 🚀 QUALITY SCORE MULTIPLIER
            quality_multiplier = 1.0
            if quality_score >= 20:
                quality_multiplier = 1.6  # Premium signals get much bigger positions
            elif quality_score >= 16:
                quality_multiplier = 1.4
            elif quality_score >= 12:
                quality_multiplier = 1.2
            elif quality_score >= 8:
                quality_multiplier = 1.0
            else:
                quality_multiplier = 0.7  # Reduce size for low quality
            
            # 🧠 MARKET REGIME MULTIPLIER  
            regime_multiplier = 1.0
            regime = market_regime.get('regime', 'UNKNOWN')
            regime_confidence = market_regime.get('confidence', 0)
            
            if regime == "STRONG_TRENDING" and regime_confidence > 0.7:
                regime_multiplier = 1.5  # Aggressive in strong trends
            elif regime == "WEAK_TRENDING":
                regime_multiplier = 1.2
            elif regime == "VOLATILE":
                regime_multiplier = 0.8  # Conservative in volatile markets
            elif regime == "SIDEWAYS":
                regime_multiplier = 0.6  # Very conservative in sideways
            
            # 📊 KELLY CRITERION APPROXIMATION
            # Simple Kelly based on recent trade history
            kelly_multiplier = 1.0
            if len(self.portfolio.closed_trades) >= 10:
                recent_trades = self.portfolio.closed_trades[-20:]  # Last 20 trades
                wins = [t for t in recent_trades if t.get("pnl_usdt", 0) > 0]
                losses = [t for t in recent_trades if t.get("pnl_usdt", 0) <= 0]
                
                if wins and losses:
                    win_rate = len(wins) / len(recent_trades)
                    avg_win = sum(t["pnl_usdt"] for t in wins) / len(wins)
                    avg_loss = abs(sum(t["pnl_usdt"] for t in losses) / len(losses))
                    
                    if avg_loss > 0:
                        # Simplified Kelly: (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_loss
                        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
                        kelly_fraction = max(0.1, min(2.0, kelly_fraction))  # Cap between 10% and 200%
                        kelly_multiplier = kelly_fraction
            
            # 🎯 COMBINED SIZING CALCULATION
            enhanced_size_pct = base_size_pct * quality_multiplier * regime_multiplier * kelly_multiplier
            
            # Apply limits
            enhanced_size_pct = min(enhanced_size_pct, self.size_max_balance_pct)
            position_amount = available_usdt * (enhanced_size_pct / 100.0)
            position_amount = max(self.min_position_usdt, min(position_amount, self.max_position_usdt))
            
            # Final safety check
            if position_amount > available_usdt * 0.95:
                position_amount = available_usdt * 0.95
            
            logger.debug(f"💰 Dynamic Sizing: Base={base_size_pct:.1f}%, Quality={quality_multiplier:.2f}x, "
                        f"Regime={regime_multiplier:.2f}x, Kelly={kelly_multiplier:.2f}x, Final=${position_amount:.2f}")
            
            return position_amount
            
        except Exception as e:
            logger.error(f"Dynamic position sizing error: {e}")
            # Fallback to basic sizing
            available_usdt = self.portfolio.get_available_usdt()
            fallback_amount = available_usdt * (self.base_position_pct / 100.0)
            return max(self.min_position_usdt, min(fallback_amount, self.max_position_usdt))

    def calculate_adaptive_stop_loss(self, entry_price: float, indicators: Optional[pd.DataFrame], market_regime: Dict[str, Any]) -> float:
        """🛡️ Adaptive stop-loss based on market conditions"""
        base_sl_pct = self.max_loss_pct
        
        try:
            # Market regime adjustment
            regime = market_regime.get('regime', 'UNKNOWN')
            if regime == "VOLATILE":
                base_sl_pct *= 1.5  # Wider stops in volatile markets
            elif regime == "STRONG_TRENDING":
                base_sl_pct *= 0.8  # Tighter stops in strong trends
            elif regime == "SIDEWAYS":
                base_sl_pct *= 1.2  # Slightly wider in sideways
            
            # ATR-based adjustment
            if indicators is not None and 'atr' in indicators.columns:
                current_atr = indicators.iloc[-1].get('atr')
                if pd.notna(current_atr) and entry_price > 0:
                    atr_sl_pct = (current_atr * 2) / entry_price  # 2x ATR stop
                    # Use the wider of the two stops for safety
                    base_sl_pct = max(base_sl_pct, atr_sl_pct)
            
            sl_price = entry_price * (1 - base_sl_pct)
            
            logger.debug(f"🛡️ Adaptive Stop-Loss: Entry=${entry_price:.2f}, SL=${sl_price:.2f} "
                        f"({base_sl_pct*100:.2f}%), Regime={regime}")
            
            return sl_price
            
        except Exception as e:
            logger.error(f"Stop-loss calculation error: {e}")
            return entry_price * (1 - self.max_loss_pct)

    # Keep the existing should_sell method structure but enhance it
    async def should_sell(self, position: Position, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """🚀 Enhanced sell logic with trailing stops and dynamic exits"""
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        
        sell_context: Dict[str, Any] = {
            "current_price": current_price, 
            "position_id": position.position_id,
            "entry_price": position.entry_price, 
            "indicators": {}
        }
        
        current_time = getattr(self, '_current_backtest_time', datetime.now(timezone.utc))
        
        # Calculate position metrics
        position_age_minutes = 0.0
        if position.timestamp:
            try:
                entry_time = datetime.fromisoformat(position.timestamp.replace('Z', '+00:00'))
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)
                position_age_minutes = (current_time - entry_time).total_seconds() / 60.0
            except Exception as e:
                logger.warning(f"Error calculating position age: {e}")

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

        # Get indicators and market regime
        indicators = await self.calculate_indicators(df)
        market_regime = {"regime": "UNKNOWN", "confidence": 0.0}
        if indicators is not None and not indicators.empty:
            market_regime = self.detect_market_regime(indicators)
            ml_prediction = self.get_ml_prediction(df)
            sell_context.update({
                "ml_prediction": ml_prediction,
                "market_regime": market_regime
        })
        
        # 🧠 CHECK ML-BASED EXIT SIGNALS (HIGH PRIORITY)
        ml_exit_signal, ml_exit_reason = self.get_ml_exit_signal(position, df, ml_prediction)
        if ml_exit_signal:
            return True, ml_exit_reason, sell_context
            current_indicators = indicators.iloc[-1]
            sell_context["indicators"].update({
                "rsi": current_indicators.get('rsi', 50),
                "adx": current_indicators.get('adx', 20),
                "volume_ratio": current_indicators.get('volume_ratio', 1.0),
                "market_regime": market_regime['regime']
            })

        # 🚀 ENHANCED PREMIUM PROFIT LEVELS
        if profit_usd >= self.sell_premium_excellent:
            return True, f"PREMIUM_EXCELLENT_${profit_usd:.2f}", sell_context
        elif profit_usd >= self.sell_premium_great:
            return True, f"PREMIUM_GREAT_${profit_usd:.2f}", sell_context
        elif profit_usd >= self.sell_premium_good:
            return True, f"PREMIUM_GOOD_${profit_usd:.2f}", sell_context

        # 🛡️ ENHANCED RISK MANAGEMENT
        # Minimum hold time with catastrophic protection
        if position_age_minutes < self.sell_min_hold_minutes:
            if profit_pct < self.sell_catastrophic_loss_pct * 100:
                return True, f"EMERGENCY_EXIT_CATASTROPHIC_{profit_pct:.1f}%", sell_context
            else:
                return False, f"MIN_HOLD_{position_age_minutes:.0f}m_of_{self.sell_min_hold_minutes}m", sell_context

        # 🎯 REGIME-AWARE SELLING
        regime = market_regime.get('regime', 'UNKNOWN')
        regime_confidence = market_regime.get('confidence', 0)
        
        # In volatile markets, take profits earlier
        if regime == "VOLATILE" and regime_confidence > 0.6:
            if profit_usd >= self.sell_premium_good * 0.7:  # 30% earlier profit taking
                return True, f"VOLATILE_REGIME_PROFIT_${profit_usd:.2f}", sell_context
        
        # In strong trends, be more patient
        if regime == "STRONG_TRENDING" and regime_confidence > 0.7:
            # Extend normal profit targets by 50%
            adjusted_phase1_excellent = self.sell_phase1_excellent * 1.5
            adjusted_phase2_excellent = self.sell_phase2_excellent * 1.5
        else:
            adjusted_phase1_excellent = self.sell_phase1_excellent
            adjusted_phase2_excellent = self.sell_phase2_excellent

        # 📊 ENHANCED PHASE-BASED SELLING
        if self.sell_min_hold_minutes <= position_age_minutes <= 60:
            if profit_usd >= adjusted_phase1_excellent: 
                return True, f"P1_EXC_${profit_usd:.2f}", sell_context
            if profit_usd >= self.sell_phase1_good: 
                return True, f"P1_GOOD_${profit_usd:.2f}", sell_context
            if profit_usd <= self.sell_phase1_loss_protection: 
                return True, f"P1_LOSS_PROT_${profit_usd:.2f}", sell_context
                
        elif 60 < position_age_minutes <= 120:
            if profit_usd >= adjusted_phase2_excellent: 
                return True, f"P2_EXC_${profit_usd:.2f}", sell_context
            if profit_usd >= self.sell_phase2_good: 
                return True, f"P2_GOOD_${profit_usd:.2f}", sell_context
            if profit_usd <= self.sell_phase2_loss_protection: 
                return True, f"P2_LOSS_PROT_${profit_usd:.2f}", sell_context
                
        elif 120 < position_age_minutes <= 180:
            if profit_usd >= self.sell_phase3_excellent: 
                return True, f"P3_EXC_${profit_usd:.2f}", sell_context
            if profit_usd >= self.sell_phase3_good: 
                return True, f"P3_GOOD_${profit_usd:.2f}", sell_context
            if self.sell_phase3_breakeven_min <= profit_usd <= self.sell_phase3_breakeven_max: 
                return True, f"P3_BREAKEVEN_${profit_usd:.2f}", sell_context
            if profit_usd <= self.sell_phase3_loss_protection: 
                return True, f"P3_LOSS_PROT_${profit_usd:.2f}", sell_context
                
        elif position_age_minutes >= self.sell_phase4_force_exit_minutes:
            return True, f"P4_FORCE_EXIT_{position_age_minutes:.0f}m_${profit_usd:.2f}", sell_context

        # 🚨 ENHANCED TECHNICAL EXIT CONDITIONS
        if indicators is not None and not indicators.empty:
            current_indicators = indicators.iloc[-1]
            
            # RSI extreme conditions
            rsi = current_indicators.get('rsi')
            if (pd.notna(rsi) and rsi < self.sell_tech_rsi_extreme and 
                position_age_minutes >= self.sell_tech_min_minutes and
                profit_usd <= self.sell_tech_min_loss):
                return True, f"TECH_RSI_EXTREME_{rsi:.1f}_${profit_usd:.2f}", sell_context
            
            # ADX divergence (trend weakening)
            adx = current_indicators.get('adx', 20)
            if adx < 15 and profit_usd < 0 and position_age_minutes >= 45:
                return True, f"TECH_TREND_WEAK_ADX_{adx:.1f}_${profit_usd:.2f}", sell_context

        # 💥 ABSOLUTE LOSS LIMIT
        position_entry_cost = position.entry_cost_usdt_total
        total_trading_cost_estimate = (position_entry_cost * settings.FEE_BUY) + (potential_gross * settings.FEE_SELL)
        max_acceptable_loss_abs = total_trading_cost_estimate * self.sell_loss_multiplier
        
        if profit_usd <= -max_acceptable_loss_abs:
            return True, f"MAX_LOSS_LIMIT_${profit_usd:.2f}_vs_${-max_acceptable_loss_abs:.2f}", sell_context
        
        return False, f"HOLD_AGE_{position_age_minutes:.0f}m_PNL_${profit_usd:.2f}_REGIME_{regime}", sell_context

    # Keep the existing should_buy method but enhance it with the new features
    async def should_buy(self, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """🎯 Enhanced buy logic with multi-timeframe analysis and regime awareness"""
        indicators = await self.calculate_indicators(df)
        if indicators is None or len(indicators) < 2:
            return False, "No or insufficient indicators", {}
            
        current = indicators.iloc[-1]
        current_price = current['close']
        
        buy_context: Dict[str, Any] = {
            "current_price": current_price, 
            "quality_score": 0, 
            "indicators": {}, 
            "reason": "", 
            "strategy_name": self.strategy_name
        }
        
        # Position and balance checks
        open_positions = self.portfolio.get_open_positions(self.symbol, strategy_name=self.strategy_name)
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions ({len(open_positions)}/{self.max_positions})", buy_context
        
        # Market regime detection
        market_regime = self.detect_market_regime(indicators)
        ml_prediction = self.get_ml_prediction(df)
        buy_context["ml_prediction"] = ml_prediction
        buy_context["market_regime"] = market_regime
        
        quality_score, score_breakdown = self.calculate_ml_enhanced_quality_score(indicators, market_regime, ml_prediction)
        # Enhanced quality scoring
        
        buy_context.update({
            "quality_score": quality_score,
            "score_breakdown": score_breakdown,
            "min_quality_required": self.buy_min_quality_score
        })
        
        # Quality threshold check (enhanced)
        if quality_score < self.buy_min_quality_score:
            return False, f"QUALITY_LOW_Q{quality_score}_REQ{self.buy_min_quality_score}", buy_context
        
        # Dynamic position sizing
        required_amount = self.calculate_dynamic_position_size(current_price, quality_score, market_regime)
        available_usdt = self.portfolio.get_available_usdt()
        buy_context["required_amount"] = required_amount
        buy_context["available_usdt"] = available_usdt
        
        if available_usdt < required_amount:
            return False, f"Insufficient balance (need ${required_amount:.2f}, have ${available_usdt:.2f})", buy_context
        
        # Time-based restrictions
        if self.last_trade_time:
            current_time = getattr(self, '_current_backtest_time', datetime.now(timezone.utc))
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            seconds_since_last = (current_time - self.last_trade_time).total_seconds()
            if seconds_since_last < 25:  # Enhanced from config
                return False, f"Wait time {seconds_since_last:.0f}s of 25s", buy_context
        
        # 🚀 CORE TECHNICAL ANALYSIS (Enhanced)
        ema_short, ema_medium, ema_long = current.get('ema_short'), current.get('ema_medium'), current.get('ema_long')
        buy_context["indicators"].update({"ema_short": ema_short, "ema_medium": ema_medium, "ema_long": ema_long})

        # EMA trend validation (stricter)
        if not (pd.notna(ema_short) and pd.notna(ema_medium) and pd.notna(ema_long) and ema_short > ema_medium > ema_long):
            return False, "EMA_TREND_FAIL", buy_context
        
        # Enhanced EMA spread validation
        ema_spread_1 = (ema_short - ema_medium) / ema_medium if ema_medium else 0
        ema_spread_2 = (ema_medium - ema_long) / ema_long if ema_long else 0
        buy_context.update({"ema_spread_1": ema_spread_1, "ema_spread_2": ema_spread_2})
        
        if ema_spread_1 < self.buy_min_ema_spread_1 or ema_spread_2 < self.buy_min_ema_spread_2:
            return False, f"EMA_SPREAD_NARROW_{ema_spread_1*100:.3f}%_{ema_spread_2*100:.3f}%", buy_context

        # 🧠 AI CONFIRMATION (Enhanced)
        buy_context["ai_approved"] = "N/A"
        if self.ai_provider:
            # Add regime information to AI context
            ai_context = {**buy_context, **market_regime}
            ai_confirmation = await self.ai_provider.get_ai_confirmation(
                current_signal_type="BUY", 
                ohlcv_df=df, 
                context=ai_context
            )
            buy_context["ai_approved"] = str(ai_confirmation)
            if not ai_confirmation:
                return False, f"AI_REJECT_Q{quality_score}_REGIME_{market_regime['regime']}", buy_context
        
        # 🎯 FINAL BUY DECISION
        buy_reason_final = f"ENHANCED_MOM_Q{quality_score}_REGIME_{market_regime['regime'][:4]}"
        if self.ai_provider and buy_context["ai_approved"] == "True":
            buy_reason_final += "_AI_ENHANCED"
        
        buy_context["reason"] = buy_reason_final
        
        # Update quality score history
        self.quality_score_history.append({
            "timestamp": datetime.now(timezone.utc),
            "quality_score": quality_score,
            "market_regime": market_regime['regime'],
            "price": current_price
        })
        
        # Keep only last 100 scores
        if len(self.quality_score_history) > 100:
            self.quality_score_history = self.quality_score_history[-100:]
        
        logger.info(f"🎯 ENHANCED BUY SIGNAL: Q={quality_score}, Regime={market_regime['regime']}, "
                   f"Conf={market_regime['confidence']:.2f}, Size=${required_amount:.0f}")
        
        return True, buy_reason_final, buy_context

    # Keep the existing process_data method structure
    async def process_data(self, df: pd.DataFrame) -> None:
        """🚀 Enhanced main strategy execution with advanced features"""
        try:
            if df.empty:
                return
                
            current_bar = df.iloc[-1]
            current_price = current_bar['close']
            
            current_time_for_process = getattr(self, '_current_backtest_time', datetime.now(timezone.utc))
            current_time_iso = current_time_for_process.isoformat()
            
            # Get open positions for this strategy
            open_positions = self.portfolio.get_open_positions(self.symbol, strategy_name=self.strategy_name)
            
            # 🚀 Enhanced sell processing with trailing stops
            for position in list(open_positions):
                should_sell_flag, sell_reason, sell_context_dict = await self.should_sell(position, df)
                if should_sell_flag:
                    await self.portfolio.execute_sell(
                        position_to_close=position, 
                        current_price=current_price,
                        timestamp=current_time_iso, 
                        reason=sell_reason, 
                        sell_context=sell_context_dict
                    )
                    logger.info(f"📤 ENHANCED SELL: {position.position_id} at ${current_price:.2f} - {sell_reason}")

            # Refresh position list after sells
            open_positions_after_sell = self.portfolio.get_open_positions(self.symbol, strategy_name=self.strategy_name)

            # 🎯 Enhanced buy processing with dynamic sizing
            if len(open_positions_after_sell) < self.max_positions:
                should_buy_flag, buy_reason_str, buy_context_dict = await self.should_buy(df)
                if should_buy_flag:
                    # Calculate enhanced position details
                    position_amount = buy_context_dict.get("required_amount")
                    if not position_amount:
                        position_amount = self.calculate_dynamic_position_size(
                            current_price, 
                            buy_context_dict.get("quality_score", 10),
                            buy_context_dict.get("market_regime", {"regime": "UNKNOWN", "confidence": 0})
                        )
                    
                    # Calculate adaptive stop loss
                    indicators_for_sl = await self.calculate_indicators(df)
                    market_regime = buy_context_dict.get("market_regime", {"regime": "UNKNOWN", "confidence": 0})
                    stop_loss_price = self.calculate_adaptive_stop_loss(current_price, indicators_for_sl, market_regime)
                    
                    # Execute enhanced buy order
                    new_position = await self.portfolio.execute_buy(
                        strategy_name=self.strategy_name, 
                        symbol=self.symbol,
                        current_price=current_price, 
                        timestamp=current_time_iso,
                        reason=buy_reason_str, 
                        amount_usdt_override=position_amount,
                        stop_loss_price_from_strategy=stop_loss_price,
                        buy_context=buy_context_dict
                    )
                    
                    if new_position:
                        self.position_entry_reasons[new_position.position_id] = buy_reason_str
                        self.last_trade_time = current_time_for_process
                        
                        quality_score = buy_context_dict.get("quality_score", 0)
                        regime = market_regime.get("regime", "UNKNOWN")
                        
                        logger.info(f"📥 ENHANCED BUY: {new_position.position_id} ${position_amount:.0f} "
                                  f"at ${current_price:.2f} SL=${stop_loss_price:.2f} - Q{quality_score} {regime}")

            # 🧠 ML PERFORMANCE TRACKING
            if hasattr(self, 'ml_predictor') and self.ml_enabled:
                try:
                    # Track ML prediction accuracy
                    if len(self.ml_predictions_history) >= 5:
                        self._track_ml_performance(current_price)
                except Exception as e:
                    logger.debug(f"ML performance tracking error: {e}")
                
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"🛑 [{self.strategy_name}] Enhanced strategy processing interrupted")
            raise
        except Exception as e:
            logger.error(f"[{self.strategy_name}] Enhanced process data error: {e}", exc_info=True)

    def get_ml_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """🧠 Get ML-based price prediction"""
        try:
            if not self.ml_enabled:
                return {"prediction": 0.0, "confidence": 0.0, "direction": "NEUTRAL", "enabled": False}
            
            # Update ML training data
            self.ml_predictor.update_training_data(df)
            
            # Get prediction
            ml_result = self.ml_predictor.predict_price_movement(df)
            ml_result["enabled"] = True
            
            # Store prediction history
            prediction_record = {
                "timestamp": datetime.now(timezone.utc),
                "prediction": ml_result["prediction"],
                "confidence": ml_result["confidence"],
                "direction": ml_result["direction"],
                "current_price": df['close'].iloc[-1]
            }
            self.ml_predictions_history.append(prediction_record)
            
            return ml_result
            
        except Exception as e:
            logger.debug(f"ML prediction error: {e}")
            return {"prediction": 0.0, "confidence": 0.0, "direction": "NEUTRAL", "enabled": False}
    
    def calculate_ml_enhanced_quality_score(self, indicators: pd.DataFrame, 
                                          market_regime: Dict[str, Any], 
                                          ml_prediction: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """🎯 ML-Enhanced quality scoring system"""
        # Get base quality score
        base_score, score_breakdown = self.calculate_enhanced_quality_score(indicators, market_regime)
        
        if not ml_prediction.get("enabled", False):
            return base_score, score_breakdown
        
        try:
            ml_direction = ml_prediction.get("direction", "NEUTRAL")
            ml_confidence = ml_prediction.get("confidence", 0.0)
            ml_prediction_value = ml_prediction.get("prediction", 0.0)
            
            # ML Enhancement bonus (Max: 5 points)
            ml_bonus = 0
            
            # Strong ML bullish signal
            if ml_direction == "UP" and ml_confidence > 0.7 and ml_prediction_value > 0.01:
                ml_bonus = 5
                score_breakdown['ml_strong_bullish'] = 5
            elif ml_direction == "UP" and ml_confidence > 0.5 and ml_prediction_value > 0.005:
                ml_bonus = 3
                score_breakdown['ml_moderate_bullish'] = 3
            elif ml_direction == "UP" and ml_confidence > 0.3:
                ml_bonus = 2
                score_breakdown['ml_weak_bullish'] = 2
            
            # ML bearish penalty
            elif ml_direction == "DOWN" and ml_confidence > 0.6:
                ml_bonus = -3
                score_breakdown['ml_bearish_penalty'] = -3
            elif ml_direction == "DOWN" and ml_confidence > 0.4:
                ml_bonus = -1
                score_breakdown['ml_weak_bearish_penalty'] = -1
            
            # High confidence neutral (market uncertainty)
            elif ml_direction == "NEUTRAL" and ml_confidence > 0.7:
                ml_bonus = -2
                score_breakdown['ml_uncertainty_penalty'] = -2
            
            enhanced_score = base_score + ml_bonus
            enhanced_score = max(0, min(30, enhanced_score))  # Cap at 30 for ML-enhanced
            
            score_breakdown['ml_bonus'] = ml_bonus
            score_breakdown['ml_confidence'] = ml_confidence
            score_breakdown['ml_direction'] = ml_direction
            score_breakdown['ml_prediction_value'] = ml_prediction_value
            score_breakdown['enhanced_total'] = enhanced_score
            
            return enhanced_score, score_breakdown
            
        except Exception as e:
            logger.debug(f"ML quality score enhancement error: {e}")
            return base_score, score_breakdown

    def get_ml_exit_signal(self, position: Position, df: pd.DataFrame, 
                          ml_prediction: Dict[str, Any]) -> Tuple[bool, str]:
        """🧠 ML-based exit signal analysis"""
        try:
            if not ml_prediction.get("enabled", False):
                return False, "ML_DISABLED"
            
            ml_direction = ml_prediction.get("direction", "NEUTRAL")
            ml_confidence = ml_prediction.get("confidence", 0.0)
            ml_prediction_value = ml_prediction.get("prediction", 0.0)
            
            current_price = df['close'].iloc[-1]
            position_profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # ML Exit Conditions
            
            # 1. Strong bearish ML signal with profit protection
            if (ml_direction == "DOWN" and ml_confidence > 0.75 and 
                ml_prediction_value < -0.01 and position_profit_pct > 1.0):
                return True, f"ML_STRONG_BEARISH_EXIT_PROFIT_{position_profit_pct:.1f}%"
            
            # 2. Moderate bearish signal with substantial profit
            if (ml_direction == "DOWN" and ml_confidence > 0.6 and 
                position_profit_pct > 3.0):
                return True, f"ML_BEARISH_EXIT_BIG_PROFIT_{position_profit_pct:.1f}%"
            
            # 3. ML trend reversal detection
            if len(self.ml_predictions_history) >= 3:
                recent_directions = [p["direction"] for p in list(self.ml_predictions_history)[-3:]]
                if all(d == "DOWN" for d in recent_directions) and position_profit_pct > 0.5:
                    return True, f"ML_TREND_REVERSAL_EXIT_PROFIT_{position_profit_pct:.1f}%"
            
            # 4. High confidence neutral signal (uncertainty increase)
            if (ml_direction == "NEUTRAL" and ml_confidence > 0.8 and 
                position_profit_pct > 2.0):
                return True, f"ML_UNCERTAINTY_EXIT_PROFIT_{position_profit_pct:.1f}%"
            
            return False, "ML_HOLD"
            
        except Exception as e:
            logger.debug(f"ML exit signal error: {e}")
            return False, "ML_ERROR"

    def _track_ml_performance(self, current_price: float):
        """🧠 Track ML prediction accuracy"""
        try:
            if len(self.ml_predictions_history) < 5:
                return
            
            # Check predictions from 4 periods ago (prediction horizon)
            prediction_to_check = None
            current_time = datetime.now(timezone.utc)
            
            for pred in self.ml_predictions_history:
                pred_time = pred["timestamp"]
                time_diff = (current_time - pred_time).total_seconds() / 60
                
                # Check prediction from ~60 minutes ago (4 * 15min bars)
                if 55 <= time_diff <= 65:
                    prediction_to_check = pred
                    break
            
            if prediction_to_check is None:
                return
            
            # Calculate actual price change
            predicted_price = prediction_to_check["current_price"]
            actual_change = (current_price - predicted_price) / predicted_price
            predicted_change = prediction_to_check["prediction"]
            
            # Check prediction accuracy
            prediction_error = abs(actual_change - predicted_change)
            direction_correct = (
                (actual_change > 0.005 and prediction_to_check["direction"] == "UP") or
                (actual_change < -0.005 and prediction_to_check["direction"] == "DOWN") or
                (abs(actual_change) <= 0.005 and prediction_to_check["direction"] == "NEUTRAL")
            )
            
            # Log performance periodically
            if len(self.ml_predictions_history) % 20 == 0:
                logger.info(f"🧠 ML Performance Check: Error={prediction_error:.4f}, "
                           f"Direction={'✅' if direction_correct else '❌'}, "
                           f"Predicted={predicted_change:.4f}, Actual={actual_change:.4f}")
            
        except Exception as e:
            logger.debug(f"ML performance tracking detailed error: {e}")

    def get_ml_performance_summary(self) -> Dict[str, Any]:
        """🧠 Get ML model performance summary"""
        try:
            if not hasattr(self, 'ml_predictor') or not self.ml_enabled:
                return {"ml_enabled": False}
            
            feature_importance = self.ml_predictor.get_feature_importance()
            model_weights = self.ml_predictor.model_weights.copy()
            
            performance_summary = {
                "ml_enabled": True,
                "is_trained": self.ml_predictor.is_trained,
                "training_samples": len(self.ml_predictor.feature_history),
                "prediction_history_size": len(self.ml_predictions_history),
                "model_weights": model_weights,
                "feature_importance_available": len(feature_importance) > 0,
                "lstm_trained": getattr(self.ml_predictor, 'lstm_trained', False)
            }
            
            # Add recent prediction summary
            if len(self.ml_predictions_history) >= 10:
                recent_predictions = list(self.ml_predictions_history)[-10:]
                directions = [p["direction"] for p in recent_predictions]
                confidences = [p["confidence"] for p in recent_predictions]
                
                performance_summary.update({
                    "recent_direction_distribution": {
                        "UP": directions.count("UP"),
                        "DOWN": directions.count("DOWN"),
                        "NEUTRAL": directions.count("NEUTRAL")
                    },
                    "avg_confidence_recent": sum(confidences) / len(confidences),
                    "max_confidence_recent": max(confidences),
                    "min_confidence_recent": min(confidences)
                })
            
            return performance_summary
            
        except Exception as e:
            logger.debug(f"ML performance summary error: {e}")
            return {"ml_enabled": False, "error": str(e)}

# Create an alias for backward compatibility
MomentumStrategy = EnhancedMomentumStrategy