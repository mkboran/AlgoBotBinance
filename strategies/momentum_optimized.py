# strategies/momentum_optimized.py - Enhanced Standalone Momentum Strategy

import pandas as pd
import pandas_ta as ta
from typing import Optional, Dict, Tuple, Any  # Any eklendi
from datetime import datetime, timezone
import asyncio

from utils.portfolio import Portfolio, Position
from utils.config import settings
from utils.logger import logger
from utils.ai_signal_provider import AiSignalProvider

class MomentumStrategy:
    """🚀 Advanced Standalone Momentum Strategy with AI Enhancement"""
    
    def __init__(self, portfolio: Portfolio, symbol: str = "BTC/USDT"):
        self.strategy_name = "Momentum"
        self.portfolio = portfolio
        self.symbol = symbol
        
        # 🔧 ALL PARAMETERS FROM CONFIG NOW!
        # Core Technical Parameters
        self.ema_short = settings.MOMENTUM_EMA_SHORT
        self.ema_medium = settings.MOMENTUM_EMA_MEDIUM
        self.ema_long = settings.MOMENTUM_EMA_LONG
        self.rsi_period = settings.MOMENTUM_RSI_PERIOD
        self.adx_period = settings.MOMENTUM_ADX_PERIOD
        self.atr_period = settings.MOMENTUM_ATR_PERIOD
        self.volume_sma_period = settings.MOMENTUM_VOLUME_SMA_PERIOD
        
        # Position Management
        self.max_positions = settings.MOMENTUM_MAX_POSITIONS
        self.base_position_pct = settings.MOMENTUM_BASE_POSITION_SIZE_PCT
        self.max_position_usdt = settings.MOMENTUM_MAX_POSITION_USDT
        self.min_position_usdt = settings.MOMENTUM_MIN_POSITION_USDT
        self.max_total_exposure_pct = settings.MOMENTUM_MAX_TOTAL_EXPOSURE_PCT
        
        # Position tracking
        self.last_trade_time = None
        self.position_entry_reasons = {}
        
        # AI Integration - CONFIG'den
        self.ai_provider = AiSignalProvider() if settings.AI_ASSISTANCE_ENABLED else None
        
        logger.info(f"✅ {self.strategy_name} Strategy initialized from CONFIG")
        logger.info(f"   Technical: EMA({self.ema_short},{self.ema_medium},{self.ema_long}), RSI({self.rsi_period}), ADX({self.adx_period})")
        logger.info(f"   Position: {self.base_position_pct}% base, ${self.min_position_usdt}-${self.max_position_usdt}, Max: {self.max_positions}")
        logger.info(f"   Buy Quality Min: {settings.MOMENTUM_BUY_MIN_QUALITY_SCORE}, AI: {'ON' if self.ai_provider else 'OFF'}")

    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate all technical indicators efficiently with error handling"""
        if len(df) < 50:
            return None
            
        try:
            # Pandas TA için veri kopyalama (KeyboardInterrupt sorununu önler)
            df_copy = df.copy()
            indicators = pd.DataFrame(index=df_copy.index)
            
            # Price data
            indicators['close'] = df_copy['close']
            indicators['volume'] = df_copy['volume']
            indicators['high'] = df_copy['high'] 
            indicators['low'] = df_copy['low']
            
            # EMAs - Error handling eklendi
            try:
                indicators['ema_short'] = ta.ema(df_copy['close'], length=self.ema_short)
                indicators['ema_medium'] = ta.ema(df_copy['close'], length=self.ema_medium)
                indicators['ema_long'] = ta.ema(df_copy['close'], length=self.ema_long)
            except (KeyboardInterrupt, SystemExit):
                raise  # KeyboardInterrupt'ı yeniden fırlat
            except Exception as e:
                logger.error(f"EMA calculation error: {e}")
                return None
            
            # RSI
            try:
                indicators['rsi'] = ta.rsi(df_copy['close'], length=self.rsi_period)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.debug(f"RSI calculation warning: {e}")
                indicators['rsi'] = 50.0  # Fallback value
            
            # ADX for trend strength
            try:
                adx_result = ta.adx(df_copy['high'], df_copy['low'], df_copy['close'], length=self.adx_period)
                if adx_result is not None and not adx_result.empty:
                    indicators['adx'] = adx_result.iloc[:, 0]
                else:
                    indicators['adx'] = 20.0  # Fallback
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.debug(f"ADX calculation warning: {e}")
                indicators['adx'] = 20.0  # Fallback
            
            # MACD
            try:
                macd_result = ta.macd(df_copy['close'])
                if macd_result is not None and not macd_result.empty:
                    indicators['macd'] = macd_result.iloc[:, 0]
                    indicators['macd_signal'] = macd_result.iloc[:, 1]
                    indicators['macd_hist'] = macd_result.iloc[:, 2]
                else:
                    indicators['macd'] = 0.0
                    indicators['macd_signal'] = 0.0
                    indicators['macd_hist'] = 0.0
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.debug(f"MACD calculation warning: {e}")
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_hist'] = 0.0
            
            # Volume analysis
            try:
                indicators['volume_sma'] = ta.sma(df_copy['volume'], length=self.volume_sma_period)
                indicators['volume_ratio'] = indicators['volume'] / indicators['volume_sma']
                indicators['volume_ratio'] = indicators['volume_ratio'].fillna(1.0)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                indicators['volume_ratio'] = 1.0
            
            # ATR for volatility
            try:
                indicators['atr'] = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], length=self.atr_period)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                indicators['atr'] = df_copy['close'] * 0.02  # 2% fallback
            
            # Support/Resistance levels
            try:
                indicators['resistance'] = df_copy['high'].rolling(window=20).max()
                indicators['support'] = df_copy['low'].rolling(window=20).min()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                indicators['resistance'] = df_copy['close']
                indicators['support'] = df_copy['close']
            
            return indicators.tail(2)  # Return last 2 bars
            
        except (KeyboardInterrupt, SystemExit):
            logger.info("🛑 Indicator calculation interrupted by user")
            raise  # KeyboardInterrupt'ı tekrar fırlat
        except Exception as e:
            logger.error(f"[{self.strategy_name}] Indicator calculation error: {e}")
            return None

    def calculate_position_size(self, current_price: float) -> float:
        """🚀 CONFIG BAZLI DİNAMİK POSITION SIZING"""
        available_usdt = self.portfolio.get_available_usdt()
        
        # CONFIG'DEN performance thresholds
        initial_capital = self.portfolio.initial_capital_usdt
        current_portfolio_value = self.portfolio.get_total_portfolio_value_usdt(current_price)
        profit_pct = (current_portfolio_value - initial_capital) / initial_capital
        
        # CONFIG'DEN position sizing based on performance
        if profit_pct > settings.MOMENTUM_PERF_HIGH_PROFIT_THRESHOLD:
            base_size_pct = settings.MOMENTUM_SIZE_HIGH_PROFIT_PCT
            logger.info(f"🔥 HIGH PROFIT MODE: {profit_pct*100:.1f}% - Using {base_size_pct}% position size")
        elif profit_pct > settings.MOMENTUM_PERF_GOOD_PROFIT_THRESHOLD:
            base_size_pct = settings.MOMENTUM_SIZE_GOOD_PROFIT_PCT
            logger.info(f"💰 GOOD PROFIT MODE: {profit_pct*100:.1f}% - Using {base_size_pct}% position size")
        elif profit_pct > settings.MOMENTUM_PERF_NORMAL_PROFIT_THRESHOLD:
            base_size_pct = settings.MOMENTUM_SIZE_NORMAL_PROFIT_PCT
        elif profit_pct > settings.MOMENTUM_PERF_BREAKEVEN_THRESHOLD:
            base_size_pct = settings.MOMENTUM_SIZE_BREAKEVEN_PCT
        else:
            base_size_pct = settings.MOMENTUM_SIZE_LOSS_PCT
            logger.info(f"⚠️ LOSS MODE: {profit_pct*100:.1f}% - Using smaller {base_size_pct}% position size")
        
        # Calculate position amount
        base_amount = available_usdt * (base_size_pct / 100.0)
        
        # CONFIG'den limits
        min_position = settings.MOMENTUM_SIZE_MIN_USD
        max_position = settings.MOMENTUM_SIZE_MAX_USD
        position_amount = max(min_position, min(base_amount, max_position))
        
        # CONFIG'den max balance percentage
        max_safe_amount = available_usdt * (settings.MOMENTUM_SIZE_MAX_BALANCE_PCT / 100.0)
        position_amount = min(position_amount, max_safe_amount)
        
        # Final backup check
        if position_amount > available_usdt * 0.9:
            position_amount = available_usdt * 0.9
            
        logger.debug(f"💵 Position Calc: ${position_amount:.0f} ({base_size_pct:.0f}% of ${available_usdt:.0f}) | P&L: {profit_pct*100:+.1f}%")
        return position_amount

    def calculate_stop_loss(self, entry_price: float, indicators: pd.DataFrame) -> float:
        """Calculate adaptive stop-loss for larger positions"""
        # Base stop-loss for larger positions (tighter)
        base_sl_pct = 0.006  # 0.6%
        
        # Adjust for volatility
        if indicators is not None and 'atr' in indicators.columns:
            current_atr = indicators.iloc[-1]['atr']
            if pd.notna(current_atr):
                volatility_ratio = current_atr / entry_price
                if volatility_ratio > 0.02:  # High volatility
                    base_sl_pct = 0.008  # Wider SL
                elif volatility_ratio < 0.01:  # Low volatility  
                    base_sl_pct = 0.005  # Tighter SL
        
        # Calculate final stop-loss
        sl_price = entry_price * (1 - base_sl_pct)
        
        logger.debug(f"[{self.strategy_name}] Stop-loss: ${sl_price:.2f} ({base_sl_pct*100:.2f}% from ${entry_price:.2f})")
        return sl_price

    async def should_sell(self, position: Position, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """💎 CONFIG BAZLI PHASE SİSTEMİ - Context bilgileriyle birlikte!"""
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        
        # Context dictionary - satım bilgilerini topla
        sell_context = {
            "current_price": current_price,
            "position_id": position.position_id,
            "entry_price": position.entry_price,
            "indicators": {}
        }
        
        # Backtest için doğru zaman hesabı
        if hasattr(self, '_current_backtest_time'):
            current_time = self._current_backtest_time
        else:
            current_time = datetime.now(timezone.utc)
        
        # Position metrics
        entry_time = datetime.fromisoformat(position.timestamp.replace('Z', '+00:00'))
        position_age_minutes = (current_time.timestamp() - entry_time.timestamp()) / 60 if hasattr(current_time, 'timestamp') else 0
        
        # Profit calculations
        potential_gross = abs(position.quantity_btc) * current_price
        potential_fee = potential_gross * settings.FEE_SELL
        potential_net = potential_gross - potential_fee
        profit_usd = potential_net - position.entry_cost_usdt_with_fee
        profit_pct = profit_usd / position.entry_cost_usdt_with_fee
        
        # Context'e bilgileri ekle
        sell_context["position_age_minutes"] = position_age_minutes
        sell_context["profit_usd"] = profit_usd
        sell_context["profit_pct"] = profit_pct
        sell_context["price_change"] = (current_price - position.entry_price) / position.entry_price
        
        position_size = position.entry_cost_usdt_with_fee
        
        # İndikatörleri context'e ekle
        indicators = await self.calculate_indicators(df)
        if indicators is not None and not indicators.empty:
            current_indicators = indicators.iloc[-1]
            sell_context["indicators"]["rsi"] = current_indicators.get('rsi', 50)
            sell_context["indicators"]["adx"] = current_indicators.get('adx', 20)
            sell_context["indicators"]["volume_ratio"] = current_indicators.get('volume_ratio', 1.0)

        # ==============================================
        # 🛡️ CONFIG'DEN MİNİMUM TUTMA SÜRESİ!
        # ==============================================
        MIN_HOLD_MINUTES = settings.MOMENTUM_SELL_MIN_HOLD_MINUTES  # Config'den 25 dakika
        
        if position_age_minutes < MIN_HOLD_MINUTES:
            # CONFIG'DEN felaket zarar eşiği
            if profit_pct < settings.MOMENTUM_SELL_CATASTROPHIC_LOSS_PCT:  # Config'den -0.03
                return True, f"EMERGENCY_EXIT_CATASTROPHIC_LOSS_{profit_pct*100:.1f}%", sell_context
            else:
                # Yoksa SABRET!
                return False, f"FORCED_HOLD_{position_age_minutes:.0f}min_of_{MIN_HOLD_MINUTES}min", sell_context
        
        # ==============================================
        # 🎯 CONFIG'DEN PREMIUM PROFIT LEVELS!
        # ==============================================
        
        if profit_usd >= settings.MOMENTUM_SELL_PREMIUM_EXCELLENT:  # Config'den $5+
            return True, f"PREMIUM_EXCELLENT_PROFIT_{settings.MOMENTUM_SELL_PREMIUM_EXCELLENT:.0f}_{profit_usd:.2f}", sell_context
        elif profit_usd >= settings.MOMENTUM_SELL_PREMIUM_GREAT:  # Config'den $3+
            return True, f"PREMIUM_GREAT_PROFIT_{settings.MOMENTUM_SELL_PREMIUM_GREAT:.0f}_{profit_usd:.2f}", sell_context
        elif profit_usd >= settings.MOMENTUM_SELL_PREMIUM_GOOD:  # Config'den $2+
            return True, f"PREMIUM_GOOD_PROFIT_{settings.MOMENTUM_SELL_PREMIUM_GOOD:.0f}_{profit_usd:.2f}", sell_context

        # ==============================================
        # 📊 CONFIG BAZLI PHASE SYSTEM!
        # ==============================================
        if 25 <= position_age_minutes <= 60:
            # PHASE 1: CONFIG'DEN HEDEFLER
            if profit_usd >= settings.MOMENTUM_SELL_PHASE1_EXCELLENT:  # Config'den $1.5
                return True, f"PHASE1_EXCELLENT_{settings.MOMENTUM_SELL_PHASE1_EXCELLENT:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            elif profit_usd >= settings.MOMENTUM_SELL_PHASE1_GOOD:  # Config'den $1.0
                return True, f"PHASE1_GOOD_{settings.MOMENTUM_SELL_PHASE1_GOOD:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            
            # PHASE 1 zarar koruması - CONFIG'DEN
            if profit_usd <= settings.MOMENTUM_SELL_PHASE1_LOSS_PROTECTION:  # Config'den -$2
                return True, f"PHASE1_LOSS_PROTECTION_{settings.MOMENTUM_SELL_PHASE1_LOSS_PROTECTION:.2f}_{profit_usd:.2f}", sell_context
        
        elif 60 < position_age_minutes <= 120:
            # PHASE 2: CONFIG'DEN HEDEFLER
            if profit_usd >= settings.MOMENTUM_SELL_PHASE2_EXCELLENT:  # Config'den $1.0
                return True, f"PHASE2_EXCELLENT_{settings.MOMENTUM_SELL_PHASE2_EXCELLENT:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            elif profit_usd >= settings.MOMENTUM_SELL_PHASE2_GOOD:  # Config'den $0.75
                return True, f"PHASE2_GOOD_{settings.MOMENTUM_SELL_PHASE2_GOOD:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            elif profit_usd >= settings.MOMENTUM_SELL_PHASE2_DECENT and position_age_minutes >= 90:  # Config'den $0.50 (90dk+)
                return True, f"PHASE2_DECENT_{settings.MOMENTUM_SELL_PHASE2_DECENT:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            
            # PHASE 2 zarar koruması - CONFIG'DEN
            if profit_usd <= settings.MOMENTUM_SELL_PHASE2_LOSS_PROTECTION:  # Config'den -$1.5
                return True, f"PHASE2_LOSS_PROTECTION_{settings.MOMENTUM_SELL_PHASE2_LOSS_PROTECTION:.2f}_{profit_usd:.2f}", sell_context
        
        elif 120 < position_age_minutes <= 180:
            # PHASE 3: CONFIG'DEN HEDEFLER
            if profit_usd >= settings.MOMENTUM_SELL_PHASE3_EXCELLENT:  # Config'den $0.75
                return True, f"PHASE3_EXCELLENT_{settings.MOMENTUM_SELL_PHASE3_EXCELLENT:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            elif profit_usd >= settings.MOMENTUM_SELL_PHASE3_GOOD:  # Config'den $0.50
                return True, f"PHASE3_GOOD_{settings.MOMENTUM_SELL_PHASE3_GOOD:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            elif profit_usd >= settings.MOMENTUM_SELL_PHASE3_DECENT:  # Config'den $0.25
                return True, f"PHASE3_DECENT_{settings.MOMENTUM_SELL_PHASE3_DECENT:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            
            # PHASE 3 kırılma noktası koruması - CONFIG'DEN
            elif settings.MOMENTUM_SELL_PHASE3_BREAKEVEN_MIN <= profit_usd <= settings.MOMENTUM_SELL_PHASE3_BREAKEVEN_MAX:
                return True, f"PHASE3_BREAKEVEN_PROTECTION_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            
            # PHASE 3 zarar koruması - CONFIG'DEN
            if profit_usd <= settings.MOMENTUM_SELL_PHASE3_LOSS_PROTECTION:  # Config'den -$1
                return True, f"PHASE3_LOSS_PROTECTION_{settings.MOMENTUM_SELL_PHASE3_LOSS_PROTECTION:.2f}_{profit_usd:.2f}", sell_context
        
        else:  # 180+ dakika - PHASE 4
            # PHASE 4: CONFIG'DEN HEDEFLER
            if profit_usd >= settings.MOMENTUM_SELL_PHASE4_EXCELLENT:  # Config'den $0.50
                return True, f"PHASE4_EXCELLENT_{settings.MOMENTUM_SELL_PHASE4_EXCELLENT:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            elif profit_usd >= settings.MOMENTUM_SELL_PHASE4_GOOD:  # Config'den $0.25
                return True, f"PHASE4_GOOD_{settings.MOMENTUM_SELL_PHASE4_GOOD:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            elif profit_usd >= settings.MOMENTUM_SELL_PHASE4_MINIMAL:  # Config'den $0.10
                return True, f"PHASE4_MINIMAL_{settings.MOMENTUM_SELL_PHASE4_MINIMAL:.2f}_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            
            # PHASE 4 kırılma noktası koruması - CONFIG'DEN (geniş)
            elif settings.MOMENTUM_SELL_PHASE4_BREAKEVEN_MIN <= profit_usd <= settings.MOMENTUM_SELL_PHASE4_BREAKEVEN_MAX:
                return True, f"PHASE4_BREAKEVEN_WIDE_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
            
            # ZORLA ÇIKIŞ - CONFIG'DEN
            if position_age_minutes >= settings.MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES:  # Config'den 240 dakika
                return True, f"PHASE4_FORCE_EXIT_{settings.MOMENTUM_SELL_PHASE4_FORCE_EXIT_MINUTES}min_{profit_usd:.2f}_{position_age_minutes:.0f}min", sell_context
        
        # ==============================================
        # 🛡️ CONFIG'DEN MUTLAK ZARAR LİMİTİ
        # ==============================================
        
        # İşlem maliyeti hesabı
        total_trading_cost = (position_size * settings.FEE_BUY) + (potential_gross * settings.FEE_SELL) + (position_size * 0.003)
        max_acceptable_loss = total_trading_cost * settings.MOMENTUM_SELL_LOSS_MULTIPLIER  # Config'den 4.0 çarpan
        
        # MUTLAK ZARAR LİMİTİ
        if profit_usd <= -max_acceptable_loss:
            return True, f"ABSOLUTE_LOSS_LIMIT_{profit_usd:.2f}_COST_{max_acceptable_loss:.2f}", sell_context
        
        # ==============================================
        # 📊 CONFIG BAZLI TEKNİK ÇIKIŞLAR
        # ==============================================
        
        if (position_age_minutes >= settings.MOMENTUM_SELL_TECH_MIN_MINUTES and 
            profit_usd < settings.MOMENTUM_SELL_TECH_MIN_LOSS):  # Config'den 90dk ve -$1.5
            
            indicators = await self.calculate_indicators(df)
            if indicators is not None and not indicators.empty:
                current_indicators = indicators.iloc[-1]
                
                # CONFIG'DEN RSI aşırı satım eşiği
                rsi = current_indicators.get('rsi')
                if pd.notna(rsi) and rsi < settings.MOMENTUM_SELL_TECH_RSI_EXTREME:  # Config'den 12
                    return True, f"TECHNICAL_RSI_EXTREME_OVERSOLD_{settings.MOMENTUM_SELL_TECH_RSI_EXTREME}_{rsi:.1f}_{position_age_minutes:.0f}min", sell_context
                
                # EMA trend tamamen kırılması
                ema_short = current_indicators.get('ema_short')
                ema_medium = current_indicators.get('ema_medium')
                ema_long = current_indicators.get('ema_long')
                if all(pd.notna(x) for x in [ema_short, ema_medium, ema_long]):
                    if ema_short < ema_medium < ema_long:  # Tam ters trend
                        return True, f"TECHNICAL_COMPLETE_TREND_REVERSAL_{position_age_minutes:.0f}min", sell_context
        
        return False, f"HOLD_WITH_PATIENCE_PHASE{min(4, max(1, int((position_age_minutes-25)/60)+1))}_{position_age_minutes:.0f}min", sell_context

    async def should_buy(self, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """🎯 CONFIG BAZLI BUY LOGIC - All values from config!"""
        indicators = await self.calculate_indicators(df)
        if indicators is None or indicators.empty:
            return False, "No indicators", {}
            
        current = indicators.iloc[-1]
        previous = indicators.iloc[-2] if len(indicators) > 1 else current
        current_price = current['close']
        
        # Context dictionary
        buy_context = {
            "current_price": current_price,
            "quality_score": 0,
            "indicators": {},
            "reason": ""  # Bu satırı ekle
        }
        
        # Check position limits
        open_positions = self.portfolio.get_open_positions(self.symbol, strategy_name=self.strategy_name)
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions ({len(open_positions)}/{self.max_positions})", buy_context
        
        # Check available balance
        required_amount = self.calculate_position_size(current_price)
        available_usdt = self.portfolio.get_available_usdt()
        if available_usdt < required_amount:
            return False, "Insufficient balance", buy_context
        
        # Portfolio performance-based wait times - CONFIG'DEN!
        portfolio_value = self.portfolio.get_total_portfolio_value_usdt(current_price)
        profit_pct = (portfolio_value - self.portfolio.initial_capital_usdt) / self.portfolio.initial_capital_usdt
        buy_context["portfolio_profit_pct"] = profit_pct
        
        # CONFIG'DEN wait times
        if profit_pct > settings.MOMENTUM_PERF_GOOD_PROFIT_THRESHOLD:
            min_wait_seconds = settings.MOMENTUM_WAIT_PROFIT_5PCT
        elif profit_pct > settings.MOMENTUM_PERF_NORMAL_PROFIT_THRESHOLD:
            min_wait_seconds = settings.MOMENTUM_WAIT_PROFIT_2PCT
        elif profit_pct > settings.MOMENTUM_PERF_BREAKEVEN_THRESHOLD:
            min_wait_seconds = settings.MOMENTUM_WAIT_BREAKEVEN
        else:
            min_wait_seconds = settings.MOMENTUM_WAIT_LOSS
        
        buy_context["wait_time_seconds"] = min_wait_seconds
        
        # Wait time check
        if hasattr(self, 'last_trade_time') and self.last_trade_time:
            if hasattr(self, '_current_backtest_time'):
                current_time = self._current_backtest_time
            else:
                current_time = datetime.now(timezone.utc)
                
            if hasattr(current_time, 'timestamp'):
                time_diff_seconds = (current_time.timestamp() - self.last_trade_time.timestamp()) if hasattr(self.last_trade_time, 'timestamp') else 0
            else:
                time_diff_seconds = 0
            
            if time_diff_seconds < min_wait_seconds:
                wait_remaining = min_wait_seconds - time_diff_seconds
                return False, f"PATIENCE_WAIT_{wait_remaining/60:.0f}min_more", buy_context
        
        # ==============================================
        # 🎯 CONFIG BASED QUALITY SCORING!
        # ==============================================
        quality_score = 0
        
        # 1. EMA Trend - CONFIG values
        ema_short = current['ema_short']
        ema_medium = current['ema_medium'] 
        ema_long = current['ema_long']
        
        buy_context["indicators"]["ema_short"] = ema_short
        buy_context["indicators"]["ema_medium"] = ema_medium
        buy_context["indicators"]["ema_long"] = ema_long
        buy_context["ema_trend"] = f"{ema_short:.1f}>{ema_medium:.1f}>{ema_long:.1f}"
        
        if pd.isna(ema_short) or pd.isna(ema_medium) or pd.isna(ema_long):
            return False, "Missing EMA data", buy_context
            
        if not (ema_short > ema_medium > ema_long):
            return False, "EMA trend not aligned", buy_context
        
        # EMA spreads - CONFIG values
        ema_spread_1 = (ema_short - ema_medium) / ema_medium
        ema_spread_2 = (ema_medium - ema_long) / ema_long
        
        buy_context["ema_spread_1"] = ema_spread_1
        buy_context["ema_spread_2"] = ema_spread_2
        
        if ema_spread_1 < settings.MOMENTUM_BUY_MIN_EMA_SPREAD_1 or ema_spread_2 < settings.MOMENTUM_BUY_MIN_EMA_SPREAD_2:
            return False, f"EMA spread too narrow ({ema_spread_1*100:.4f}%, {ema_spread_2*100:.4f}%)", buy_context
        
        # EMA momentum - CONFIG scoring
        prev_ema_short = previous['ema_short']
        if pd.notna(prev_ema_short):
            ema_momentum = (ema_short - prev_ema_short) / prev_ema_short
            buy_context["ema_momentum"] = ema_momentum
            
            if ema_momentum > settings.MOMENTUM_BUY_EMA_MOM_EXCELLENT:
                quality_score += 4
            elif ema_momentum > settings.MOMENTUM_BUY_EMA_MOM_GOOD:
                quality_score += 3
            elif ema_momentum > settings.MOMENTUM_BUY_EMA_MOM_DECENT:
                quality_score += 2
            elif ema_momentum > settings.MOMENTUM_BUY_EMA_MOM_MIN:
                quality_score += 1
            else:
                return False, f"EMA momentum insufficient ({ema_momentum*100:.4f}%)", buy_context
        
        # 2. RSI - CONFIG ranges
        rsi = current['rsi']
        buy_context["rsi"] = rsi
        buy_context["indicators"]["rsi"] = rsi
        
        if pd.notna(rsi):
            if settings.MOMENTUM_BUY_RSI_EXCELLENT_MIN <= rsi <= settings.MOMENTUM_BUY_RSI_EXCELLENT_MAX:
                quality_score += 3
            elif settings.MOMENTUM_BUY_RSI_GOOD_MIN <= rsi <= settings.MOMENTUM_BUY_RSI_GOOD_MAX:
                quality_score += 2
            elif rsi > settings.MOMENTUM_BUY_RSI_EXTREME_MAX or rsi < settings.MOMENTUM_BUY_RSI_EXTREME_MIN:
                return False, f"RSI extreme ({rsi:.1f})", buy_context
            else:
                quality_score += 1
        
        # 3. ADX - CONFIG levels
        adx = current['adx']
        buy_context["adx"] = adx
        buy_context["indicators"]["adx"] = adx
        
        if pd.notna(adx):
            if adx > settings.MOMENTUM_BUY_ADX_EXCELLENT:
                quality_score += 3
            elif adx > settings.MOMENTUM_BUY_ADX_GOOD:
                quality_score += 2
            elif adx > settings.MOMENTUM_BUY_ADX_DECENT:
                quality_score += 1
        
        # 4. Volume - CONFIG levels
        volume_ratio = current.get('volume_ratio', 1.0)
        buy_context["volume_ratio"] = volume_ratio
        buy_context["indicators"]["volume_ratio"] = volume_ratio
        
        if pd.notna(volume_ratio):
            if volume_ratio > settings.MOMENTUM_BUY_VOLUME_EXCELLENT:
                quality_score += 3
            elif volume_ratio > settings.MOMENTUM_BUY_VOLUME_GOOD:
                quality_score += 2
            elif volume_ratio > settings.MOMENTUM_BUY_VOLUME_DECENT:
                quality_score += 1
        
        # 5. MACD
        macd = current.get('macd')
        macd_signal = current.get('macd_signal')
        if pd.notna(macd) and pd.notna(macd_signal):
            macd_diff = macd - macd_signal
            buy_context["macd_strength"] = macd_diff
            buy_context["indicators"]["macd"] = macd
            buy_context["indicators"]["macd_signal"] = macd_signal
            
            if macd > macd_signal:
                quality_score += 3
            elif macd > 0:
                quality_score += 2
            else:
                quality_score += 1
        
        # 6. Price momentum - CONFIG levels
        price_change_pct = (current_price - previous['close']) / previous['close']
        buy_context["price_momentum"] = price_change_pct
        
        if price_change_pct > settings.MOMENTUM_BUY_PRICE_MOM_EXCELLENT:
            quality_score += 3
        elif price_change_pct > settings.MOMENTUM_BUY_PRICE_MOM_GOOD:
            quality_score += 2
        elif price_change_pct >= settings.MOMENTUM_BUY_PRICE_MOM_DECENT:
            quality_score += 1
        
        # ==============================================
        # 🎯 CONFIG MINIMUM SCORE CHECK!
        # ==============================================
        min_quality_score = settings.MOMENTUM_BUY_MIN_QUALITY_SCORE
        buy_context["quality_score"] = quality_score
        buy_context["min_quality_required"] = min_quality_score
        
        if quality_score < min_quality_score:
            return False, f"Quality insufficient ({quality_score}/{min_quality_score})", buy_context
        
        # ==============================================
        # 🤖 AI CONFIRMATION
        # ==============================================
        if self.ai_provider:
            ai_confirmation = await self.ai_provider.get_ai_confirmation(
                current_signal_type="BUY",
                ohlcv_df=df,
                context=buy_context
            )
            
            if not ai_confirmation:
                return False, f"AI_REJECTED_Q{quality_score}_CAUTIOUS", buy_context
        
        # Return statement'ını güncelle
        buy_context["reason"] = f"CONFIG_BASED_BUY_Q{quality_score}"
        return True, f"CONFIG_BASED_BUY_Q{quality_score}", buy_context

    async def process_data(self, df: pd.DataFrame) -> None:
        """Main strategy execution with enhanced position management"""
        try:
            if df.empty:
                return
                
            current_bar = df.iloc[-1]
            current_price = current_bar['close']
            
            # BACKTEST TARİH DÜZELTMESİ
            if hasattr(self, '_current_backtest_time'):
                current_time = self._current_backtest_time
            else:
                current_time = datetime.now(timezone.utc)
            
            # Get all open positions for this strategy
            open_positions = self.portfolio.get_open_positions(
                self.symbol, 
                strategy_name=self.strategy_name
            )
            
            # Portfolio durumu
            portfolio_value = self.portfolio.get_total_portfolio_value_usdt(current_price)
            initial_capital = self.portfolio.initial_capital_usdt
            profit_pct = (portfolio_value - initial_capital) / initial_capital
            
            # ==============================================
            # 1. SELL SIGNALS (Priority: Protect Profits)
            # ==============================================
            for position in open_positions:
                should_sell_flag, sell_reason, sell_context = await self.should_sell(position, df)
                if should_sell_flag:
                    await self.portfolio.execute_sell(
                        position_to_close=position,
                        current_price=current_price,
                        timestamp=current_time.isoformat() if hasattr(current_time, 'isoformat') else str(current_time),
                        reason=sell_reason,
                        sell_context=sell_context
                    )
            
            # ==============================================
            # 2. BUY SIGNAL (Enhanced with dynamic sizing)
            # ==============================================
            should_buy_flag, buy_reason, buy_context = await self.should_buy(df)
            if should_buy_flag:
                # Calculate dynamic position size
                position_amount = self.calculate_position_size(current_price)
                indicators = await self.calculate_indicators(df)
                stop_loss_price = self.calculate_stop_loss(current_price, indicators)
                
                # Execute buy order - FIXED: reason parameter added
                new_position = await self.portfolio.execute_buy(
                    strategy_name=self.strategy_name,
                    symbol=self.symbol,
                    current_price=current_price,
                    timestamp=current_time.isoformat() if hasattr(current_time, 'isoformat') else str(current_time),
                    reason=buy_reason,  # ✅ FIXED: Missing reason parameter
                    amount_usdt_override=position_amount,
                    stop_loss_price_from_strategy=stop_loss_price,
                    buy_context=buy_context
                )
                
                if new_position:
                    self.position_entry_reasons[new_position.position_id] = buy_reason
                    self.last_trade_time = current_time
            
            # ==============================================
            # 3. MINIMAL PORTFOLIO STATUS - Backtest için sadece GEREKLI!
            # ==============================================
            if hasattr(self, '_log_counter'):
                self._log_counter += 1
            else:
                self._log_counter = 1
                
            # ⚡ BACKTEST için log'ları çok daha az yap!
            if self._log_counter % 500 == 0:  # 50 → 500 (10x daha az)
                total_exposure = sum(abs(pos.quantity_btc) * current_price for pos in open_positions)
                # SADECE trade count log!
                if len(self.portfolio.closed_trades) > 0:
                    logger.info(f"💵 Position Update | Portfolio: ${portfolio_value:.0f} ({profit_pct*100:+.1f}%) | "
                               f"Positions: {len(open_positions)} | Exposure: ${total_exposure:.0f}")
                
        except (KeyboardInterrupt, SystemExit):
            logger.info("🛑 Strategy processing interrupted by user")
            raise  # KeyboardInterrupt'ı tekrar fırlat
        except Exception as e:
            logger.error(f"[{self.strategy_name}] Process data error: {e}", exc_info=True)
