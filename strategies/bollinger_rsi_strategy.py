"""
Bollinger Bands + RSI Mean Reversion Strategy with AI Enhancement
"""

import pandas as pd
import pandas_ta as ta
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone
import asyncio

from utils.portfolio import Portfolio, Position
from utils.config import settings
from utils.logger import logger
from utils.ai_signal_provider import AiSignalProvider

class BollingerRsiStrategy:
    """🎯 Advanced Bollinger Bands + RSI Mean Reversion Strategy with AI Enhancement"""
    
    def __init__(self, portfolio: Portfolio, symbol: str = "BTC/USDT"):
        self.strategy_name = "BollingerRSI"
        self.portfolio = portfolio
        self.symbol = symbol
        
        # Core Technical Parameters (Config'den)
        self.bb_period = settings.BOLLINGER_RSI_BB_PERIOD
        self.bb_std_dev = settings.BOLLINGER_RSI_BB_STD_DEV
        self.rsi_period = settings.BOLLINGER_RSI_RSI_PERIOD
        self.volume_sma_period = settings.BOLLINGER_RSI_VOLUME_SMA_PERIOD
        
        # Position Management (Config'den)
        self.max_positions = settings.BOLLINGER_RSI_MAX_POSITIONS
        self.base_position_pct = settings.BOLLINGER_RSI_BASE_POSITION_SIZE_PCT
        self.max_position_usdt = settings.BOLLINGER_RSI_MAX_POSITION_USDT
        self.min_position_usdt = settings.BOLLINGER_RSI_MIN_POSITION_USDT
        self.max_total_exposure_pct = settings.BOLLINGER_RSI_MAX_TOTAL_EXPOSURE_PCT
        
        # Risk Management (Config'den)
        self.max_loss_pct = settings.BOLLINGER_RSI_MAX_LOSS_PCT
        self.min_profit_target = settings.BOLLINGER_RSI_MIN_PROFIT_TARGET_USDT
        self.quick_profit_threshold = settings.BOLLINGER_RSI_QUICK_PROFIT_THRESHOLD_USDT
        
        # Timing (Config'den)
        self.max_hold_minutes = settings.BOLLINGER_RSI_MAX_HOLD_MINUTES
        self.breakeven_minutes = settings.BOLLINGER_RSI_BREAKEVEN_MINUTES
        
        # Position tracking
        self.last_trade_time = None
        self.position_entry_reasons = {}
        
        # AI Integration
        self.ai_provider = AiSignalProvider() if settings.AI_ASSISTANCE_ENABLED else None
        
        logger.info(f"✅ {self.strategy_name} Strategy initialized for {symbol}")
        logger.info(f"   Config: BB({self.bb_period},{self.bb_std_dev}), RSI({self.rsi_period}), "
                   f"Pos({self.base_position_pct}%, ${self.min_position_usdt}-${self.max_position_usdt})")
        if self.ai_provider:
            logger.info(f"   🤖 AI Enhancement: ENABLED")

    async def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate all technical indicators for mean reversion"""
        if len(df) < 50:
            return None
            
        try:
            indicators = pd.DataFrame(index=df.index)
            
            # Price data
            indicators['close'] = df['close']
            indicators['volume'] = df['volume']
            indicators['high'] = df['high'] 
            indicators['low'] = df['low']
            
            # Bollinger Bands
            bb_result = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std_dev)
            if bb_result is not None and not bb_result.empty:
                indicators['bb_upper'] = bb_result.iloc[:, 0]
                indicators['bb_middle'] = bb_result.iloc[:, 1]
                indicators['bb_lower'] = bb_result.iloc[:, 2]
                indicators['bb_bandwidth'] = bb_result.iloc[:, 3]
                indicators['bb_percent'] = bb_result.iloc[:, 4]
            
            # RSI
            indicators['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
            
            # Volume analysis
            indicators['volume_sma'] = ta.sma(df['volume'], length=self.volume_sma_period)
            indicators['volume_ratio'] = indicators['volume'] / indicators['volume_sma']
            
            # Support/Resistance levels
            indicators['resistance'] = df['high'].rolling(window=20).max()
            indicators['support'] = df['low'].rolling(window=20).min()
            
            return indicators.tail(2)  # Return last 2 bars
            
        except Exception as e:
            logger.error(f"[{self.strategy_name}] Indicator calculation error: {e}")
            return None

    def calculate_position_size(self, current_price: float) -> float:
        """Calculate optimal position size for mean reversion"""
        available_usdt = self.portfolio.get_available_usdt()
        
        # Calculate base position from percentage
        base_amount = available_usdt * (self.base_position_pct / 100.0)
        
        # Apply limits
        position_amount = max(self.min_position_usdt, min(base_amount, self.max_position_usdt))
        
        # Ensure we don't exceed available balance
        if position_amount > available_usdt * 0.9:
            position_amount = available_usdt * 0.9
            
        logger.debug(f"[{self.strategy_name}] Position size: ${position_amount:.2f} (Available: ${available_usdt:.2f})")
        return position_amount

    def calculate_stop_loss(self, entry_price: float, indicators: pd.DataFrame) -> float:
        """Calculate adaptive stop-loss for mean reversion"""
        # Tighter stop-loss for mean reversion
        base_sl_pct = self.max_loss_pct
        
        # Calculate final stop-loss
        sl_price = entry_price * (1 - base_sl_pct)
        
        logger.debug(f"[{self.strategy_name}] Stop-loss: ${sl_price:.2f} ({base_sl_pct*100:.2f}% from ${entry_price:.2f})")
        return sl_price

    async def should_buy(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """🎯 Mean reversion buy signals with AI confirmation"""
        indicators = await self.calculate_indicators(df)
        if indicators is None or indicators.empty:
            return False, "No indicators"
            
        current = indicators.iloc[-1]
        current_price = current['close']
        
        # Check position limits
        open_positions = self.portfolio.get_open_positions(self.symbol, strategy_name=self.strategy_name)
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions ({len(open_positions)}/{self.max_positions})"
        
        # Check available balance
        required_amount = self.calculate_position_size(current_price)
        if self.portfolio.get_available_usdt() < required_amount:
            return False, "Insufficient balance"
        
        # Technical Analysis - Mean Reversion Quality Scoring
        quality_score = 0
        
        # 1. Bollinger Bands Position (MANDATORY)
        bb_lower = current['bb_lower']
        bb_upper = current['bb_upper'] 
        bb_percent = current.get('bb_percent', 0)
        
        if pd.isna(bb_lower) or pd.isna(bb_upper):
            return False, "Missing BB data"
            
        # Must be near lower band for mean reversion
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        if bb_position > 0.2:  # Not close enough to lower band
            return False, f"Not oversold (BB pos: {bb_position:.2f})"
        
        if bb_position <= 0.1:  # Very close to lower band
            quality_score += 2
        elif bb_position <= 0.2:  # Moderately close
            quality_score += 1
        
        # 2. RSI Oversold Conditions
        rsi = current['rsi']
        if pd.notna(rsi):
            if rsi <= 25:  # Very oversold
                quality_score += 2
            elif rsi <= 35:  # Oversold
                quality_score += 1
            elif rsi > 50:  # Not oversold enough
                return False, f"RSI not oversold ({rsi:.1f})"
        
        # 3. Volume Confirmation
        volume_ratio = current.get('volume_ratio', 1.0)
        if pd.notna(volume_ratio):
            if volume_ratio > 1.5:  # High volume confirmation
                quality_score += 2
            elif volume_ratio > 1.1:  # Above average volume
                quality_score += 1
            elif volume_ratio < 0.7:  # Too low volume
                return False, f"Low volume ({volume_ratio:.2f}x)"
        
        # 4. Support Level Proximity
        support_level = current.get('support')
        if pd.notna(support_level):
            support_distance = (current_price - support_level) / support_level
            if support_distance <= 0.01:  # Very close to support
                quality_score += 2
            elif support_distance <= 0.02:  # Close to support
                quality_score += 1
        
        # 5. BB Bandwidth (volatility check)
        bb_bandwidth = current.get('bb_bandwidth')
        if pd.notna(bb_bandwidth):
            if bb_bandwidth > 0.02:  # Good volatility for mean reversion
                quality_score += 1
        
        # Minimum quality score for mean reversion
        min_quality_score = 6
        if quality_score < min_quality_score:
            return False, f"Quality score too low ({quality_score}/{min_quality_score})"
        
        # 🤖 AI HOOK: Mean reversion AI confirmation
        if self.ai_provider:
            ai_context = {
                "strategy": self.strategy_name,
                "signal_type": "MEAN_REVERSION",
                "quality_score": quality_score,
                "bb_position": bb_position,
                "rsi": rsi,
                "volume_ratio": volume_ratio,
                "support_proximity": current_price <= support_level * 1.01 if pd.notna(support_level) else False
            }
            
            ai_confirmation = await self.ai_provider.get_ai_confirmation(
                current_signal_type="BUY",
                ohlcv_df=df,
                context=ai_context
            )
            
            if not ai_confirmation:
                return False, f"AI_REJECTED_MEAN_REV_{quality_score}"
        
        logger.info(f"✅ MEAN REVERSION BUY: Price=${current_price:.2f}, Quality={quality_score}, "
                   f"BB_pos={bb_position:.2f}, RSI={rsi:.1f}, Volume={volume_ratio:.1f}x")
        
        return True, f"AI_ENHANCED_MEAN_REV_SCORE_{quality_score}"

    async def should_sell(self, position: Position, df: pd.DataFrame) -> Tuple[bool, str]:
        """🛡️ Mean reversion profit taking and protection"""
        current_bar = df.iloc[-1]
        current_price = current_bar['close']
        current_time = datetime.now(timezone.utc)
        
        # Position metrics
        entry_time = datetime.fromisoformat(position.timestamp.replace('Z', '+00:00'))
        position_age_minutes = (current_time - entry_time).total_seconds() / 60
        
        # Profit calculations
        potential_gross = abs(position.quantity_btc) * current_price
        potential_fee = potential_gross * settings.FEE_SELL
        potential_net = potential_gross - potential_fee
        profit_usd = potential_net - position.entry_cost_usdt_with_fee
        
        # ==============================================
        # 🎯 MEAN REVERSION PROFIT TAKING
        # ==============================================
        
        # Quick profit taking (mean reversion is faster)
        if position_age_minutes >= 1 and profit_usd >= self.quick_profit_threshold:
            return True, "QUICK_MEAN_REV_PROFIT"
        
        if position_age_minutes >= 3 and profit_usd >= self.min_profit_target:
            return True, "TARGET_MEAN_REV_PROFIT"
        
        # Any reasonable profit
        if position_age_minutes >= 10 and profit_usd >= 0.3:
            return True, "ANY_PROFIT_MEAN_REV"
        
        # ==============================================
        # 🛡️ LOSS PROTECTION
        # ==============================================
        
        max_loss_usd = position.entry_cost_usdt_with_fee * self.max_loss_pct
        
        if profit_usd <= -max_loss_usd:
            return True, "MAX_LOSS_PROTECTION"
        
        # ==============================================
        # ⏰ TIME-BASED EXITS
        # ==============================================
        
        if position_age_minutes >= self.breakeven_minutes and -0.1 <= profit_usd <= 0.1:
            return True, "BREAKEVEN_PROTECTION"
        
        if position_age_minutes >= self.max_hold_minutes:
            return True, "MAX_HOLD_TIME_EXIT"
        
        # ==============================================
        # 📊 TECHNICAL EXITS (Mean Reversion Specific)
        # ==============================================
        
        indicators = await self.calculate_indicators(df)
        if indicators is not None and not indicators.empty:
            current_indicators = indicators.iloc[-1]
            
            # RSI overbought (mean reversion target reached)
            rsi = current_indicators.get('rsi')
            if pd.notna(rsi) and rsi > 65 and profit_usd >= 0.2:
                return True, "RSI_OVERBOUGHT_TARGET"
            
            # Bollinger upper band reached
            bb_upper = current_indicators.get('bb_upper')
            if pd.notna(bb_upper) and current_price >= bb_upper * 0.995 and profit_usd >= 0.2:
                return True, "BB_UPPER_TARGET"
        
        return False, "HOLD"

    def update_stop_loss(self, position: Position, current_price: float) -> None:
        """Update stop-loss for mean reversion positions"""
        position_entry_time = datetime.fromisoformat(position.timestamp.replace('Z', '+00:00'))
        position_age_minutes = (datetime.now(timezone.utc) - position_entry_time).total_seconds() / 60
        
        # Calculate current profit
        potential_gross = abs(position.quantity_btc) * current_price
        potential_fee = potential_gross * settings.FEE_SELL
        potential_net = potential_gross - potential_fee
        profit_usd = potential_net - position.entry_cost_usdt_with_fee
        
        # Move to breakeven after some profit
        if (position_age_minutes >= self.breakeven_minutes and 
            profit_usd >= 0.15 and 
            position.stop_loss_price < position.entry_price * 0.999):
            
            new_sl = position.entry_price * 0.9995  # Tiny buffer below breakeven
            position.stop_loss_price = new_sl
            logger.info(f"🛡️ BREAKEVEN: {position.position_id} SL moved to ${new_sl:.2f}")

    async def process_data(self, df: pd.DataFrame) -> None:
        """Main strategy execution loop for mean reversion with AI intelligence"""
        try:
            if df.empty:
                return
                
            current_bar = df.iloc[-1]
            current_price = current_bar['close']
            current_time = datetime.now(timezone.utc)
            
            # Get all open positions for this strategy
            open_positions = self.portfolio.get_open_positions(
                self.symbol, 
                strategy_name=self.strategy_name
            )
            
            # ==============================================
            # 1. UPDATE STOP-LOSSES FOR ALL POSITIONS
            # ==============================================
            for position in open_positions:
                self.update_stop_loss(position, current_price)
            
            # ==============================================
            # 2. CHECK SELL SIGNALS (Priority: Lock Profits)
            # ==============================================
            for position in open_positions:
                should_sell_flag, sell_reason = await self.should_sell(position, df)
                if should_sell_flag:
                    await self.portfolio.execute_sell(
                        position_to_close=position,
                        current_price=current_price,
                        timestamp=current_time.isoformat(),
                        reason=sell_reason
                    )
                    logger.info(f"📤 SELL: {position.position_id} at ${current_price:.2f} - {sell_reason}")
            
            # ==============================================
            # 3. CHECK BUY SIGNALS (Mean Reversion)
            # ==============================================
            should_buy_flag, buy_reason = await self.should_buy(df)
            if should_buy_flag:
                # Calculate position details
                position_amount = self.calculate_position_size(current_price)
                indicators = await self.calculate_indicators(df)
                stop_loss_price = self.calculate_stop_loss(current_price, indicators)
                
                # Execute buy order
                new_position = await self.portfolio.execute_buy(
                    strategy_name=self.strategy_name,
                    symbol=self.symbol,
                    current_price=current_price,
                    timestamp=current_time.isoformat(),
                    amount_usdt_override=position_amount,
                    stop_loss_price_from_strategy=stop_loss_price
                )
                
                if new_position:
                    # Store entry reason for analysis
                    self.position_entry_reasons[new_position.position_id] = buy_reason
                    self.last_trade_time = current_time
                    logger.info(f"📥 BUY: {new_position.position_id} ${position_amount:.2f} "
                              f"at ${current_price:.2f} SL=${stop_loss_price:.2f} - {buy_reason}")
            
            # ==============================================
            # 4. PORTFOLIO STATUS LOG
            # ==============================================
            if len(open_positions) > 0:
                total_value = sum(abs(pos.quantity_btc) * current_price for pos in open_positions)
                total_profit = sum((abs(pos.quantity_btc) * current_price - 
                                  pos.entry_cost_usdt_with_fee * (1 + settings.FEE_SELL)) 
                                 for pos in open_positions)
                
                logger.debug(f"📊 {self.strategy_name}: {len(open_positions)} positions, "
                           f"${total_value:.2f} exposure, ${total_profit:.2f} unrealized P&L")
            
            # 🤖 AI HOOK: Market intelligence (every 15 iterations for mean reversion)
            if self.ai_provider and hasattr(self, '_iteration_count'):
                self._iteration_count = getattr(self, '_iteration_count', 0) + 1
                if self._iteration_count % 15 == 0:
                    market_intel = await self.ai_provider.get_market_intelligence(df)
                    if market_intel.get("risk_level") in ["HIGH", "EXTREME"]:
                        logger.warning(f"🤖 {market_intel.get('risk_level')} MARKET RISK: {market_intel.get('recommendation')}")
                
        except Exception as e:
            logger.error(f"[{self.strategy_name}] Process data error: {e}", exc_info=True)
