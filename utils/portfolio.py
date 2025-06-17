# utils/portfolio.py - MOMENTUM ULTRA OPTIMIZED VERSION
import uuid
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from utils.logger import logger 
from utils.config import settings

class Position:
    """🚀 MOMENTUM ULTRA OPTIMIZED Position Class"""
    
    def __init__(
        self,
        position_id: str,
        strategy_name: str,
        symbol: str,
        quantity_btc: float,
        entry_price: float,
        entry_cost_usdt_total: float,
        timestamp: str,
        stop_loss_price: Optional[float] = None,
        entry_context: Optional[Dict[str, Any]] = None
    ):
        self.position_id = position_id
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.quantity_btc = quantity_btc 
        self.entry_price = entry_price
        self.entry_cost_usdt_total = entry_cost_usdt_total 
        self.timestamp = timestamp 
        self.stop_loss_price = stop_loss_price
        self.entry_context = entry_context or {}
        self.exit_time_iso: Optional[str] = None
        
        # 🚀 MOMENTUM ULTRA ENHANCEMENTS
        self.trailing_stop_price: Optional[float] = None
        self.highest_price_seen: float = entry_price
        self.lowest_price_seen: float = entry_price
        self.partial_sells: List[Dict] = []  # Kademeli satış tracking
        self.quality_score: int = entry_context.get('quality_score', 0) if entry_context else 0
        self.ai_approved: bool = entry_context.get('ai_approved', False) if entry_context else False
        
        # 💎 ADVANCED PERFORMANCE TRACKING
        self.max_profit_usd: float = 0.0
        self.max_profit_pct: float = 0.0
        self.max_drawdown_from_peak: float = 0.0
        self.profit_peaks: List[Tuple[datetime, float]] = []  # (time, profit) peaks
        self.risk_metrics: Dict[str, float] = {}
        self.hold_time_minutes: float = 0.0
        
        # 🎯 MOMENTUM SPECIFIC METRICS
        self.momentum_entry_strength: float = entry_context.get('momentum_score', 0.0) if entry_context else 0.0
        self.regime_at_entry: str = entry_context.get('market_regime', 'UNKNOWN') if entry_context else 'UNKNOWN'
        self.volatility_at_entry: float = entry_context.get('volatility', 0.02) if entry_context else 0.02
        
        # 📊 REAL-TIME ANALYTICS
        self.price_updates: List[Tuple[datetime, float]] = [(datetime.now(timezone.utc), entry_price)]
        self.last_update_time: datetime = datetime.now(timezone.utc)
        
    def update_performance_metrics(self, current_price: float) -> Dict[str, Any]:
        """🎯 Update comprehensive performance metrics"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Update price history
            self.price_updates.append((current_time, current_price))
            if len(self.price_updates) > 1000:  # Keep last 1000 updates
                self.price_updates = self.price_updates[-1000:]
            
            # Update extremes
            if current_price > self.highest_price_seen:
                self.highest_price_seen = current_price
            if current_price < self.lowest_price_seen:
                self.lowest_price_seen = current_price
            
            # Calculate current metrics
            current_value = abs(self.quantity_btc) * current_price
            current_profit = current_value - self.entry_cost_usdt_total
            current_profit_pct = (current_profit / self.entry_cost_usdt_total) * 100 if self.entry_cost_usdt_total > 0 else 0
            
            # Update max profit
            if current_profit > self.max_profit_usd:
                self.max_profit_usd = current_profit
                self.max_profit_pct = current_profit_pct
                self.profit_peaks.append((current_time, current_profit))
            
            # Calculate drawdown from peak
            peak_value = abs(self.quantity_btc) * self.highest_price_seen
            current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
            if current_drawdown > self.max_drawdown_from_peak:
                self.max_drawdown_from_peak = current_drawdown
            
            # Update hold time
            entry_time = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            self.hold_time_minutes = (current_time - entry_time).total_seconds() / 60
            
            # Calculate advanced risk metrics
            self._calculate_risk_metrics(current_price)
            
            self.last_update_time = current_time
            
            return {
                "current_profit": current_profit,
                "current_profit_pct": current_profit_pct,
                "max_profit": self.max_profit_usd,
                "max_drawdown": self.max_drawdown_from_peak,
                "hold_time": self.hold_time_minutes
            }
            
        except Exception as e:
            logger.debug(f"Performance metrics update error: {e}")
            return {}
    
    def _calculate_risk_metrics(self, current_price: float):
        """Calculate advanced risk metrics"""
        try:
            if len(self.price_updates) < 10:
                return
            
            # Get recent price changes
            recent_prices = [p[1] for p in self.price_updates[-20:]]
            returns = [((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]) 
                      for i in range(1, len(recent_prices))]
            
            if returns:
                volatility = np.std(returns) * np.sqrt(96)  # Annualized (15min bars)
                downside_returns = [r for r in returns if r < 0]
                downside_volatility = np.std(downside_returns) if downside_returns else 0
                
                self.risk_metrics = {
                    "volatility": volatility,
                    "downside_volatility": downside_volatility,
                    "value_at_risk_5": np.percentile(returns, 5) if len(returns) > 5 else 0,
                    "sharpe_estimate": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                }
        except Exception as e:
            logger.debug(f"Risk metrics calculation error: {e}")
    
    def should_trail_stop(self, current_price: float, trail_activation_pct: float = 3.0, 
                         trail_distance_pct: float = 1.5) -> Tuple[bool, Optional[float]]:
        """🎯 Advanced trailing stop logic"""
        try:
            current_profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            
            # Activate trailing stop if profit threshold reached
            if current_profit_pct >= trail_activation_pct:
                trail_distance = trail_distance_pct / 100.0
                new_trailing_stop = current_price * (1 - trail_distance)
                
                # Only move trailing stop up
                if self.trailing_stop_price is None or new_trailing_stop > self.trailing_stop_price:
                    old_stop = self.trailing_stop_price
                    self.trailing_stop_price = new_trailing_stop
                    return True, old_stop
            
            return False, None
            
        except Exception as e:
            logger.debug(f"Trailing stop calculation error: {e}")
            return False, None
    
    def get_performance_summary(self, current_price: float) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        self.update_performance_metrics(current_price)
        
        return {
            "position_id": self.position_id,
            "strategy": self.strategy_name,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "quantity": self.quantity_btc,
            "entry_cost": self.entry_cost_usdt_total,
            "current_value": abs(self.quantity_btc) * current_price,
            "unrealized_pnl": (abs(self.quantity_btc) * current_price) - self.entry_cost_usdt_total,
            "unrealized_pnl_pct": ((current_price - self.entry_price) / self.entry_price) * 100,
            "max_profit": self.max_profit_usd,
            "max_drawdown": self.max_drawdown_from_peak,
            "hold_time_minutes": self.hold_time_minutes,
            "quality_score": self.quality_score,
            "ai_approved": self.ai_approved,
            "trailing_stop": self.trailing_stop_price,
            "risk_metrics": self.risk_metrics,
            "momentum_strength": self.momentum_entry_strength,
            "entry_regime": self.regime_at_entry
        }

    def __repr__(self):
        base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol
        return (f"Position(id={self.position_id[:8]}, {self.strategy_name}, "
                f"qty={self.quantity_btc:.8f} {base_currency}, entry=${self.entry_price:.2f}, "
                f"Q={self.quality_score}, AI={self.ai_approved})")


class Portfolio:
    """💰 MOMENTUM ULTRA OPTIMIZED Portfolio Management System"""
    
    def __init__(self, initial_capital_usdt: float):
        self.initial_capital_usdt = initial_capital_usdt
        self.available_usdt = initial_capital_usdt
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = [] 
        self.cumulative_pnl = 0.0
        
        # 🚀 MOMENTUM ULTRA ENHANCEMENTS
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        self.trade_performance_stats: Dict[str, Any] = {}
        self.risk_analytics: Dict[str, Any] = {}
        
        # 🎯 MOMENTUM SPECIFIC METRICS
        self.momentum_trade_stats: Dict[str, Any] = {
            "total_momentum_trades": 0,
            "momentum_win_rate": 0.0,
            "momentum_avg_profit": 0.0,
            "momentum_profit_factor": 0.0,
            "best_momentum_trade": 0.0,
            "worst_momentum_trade": 0.0
        }
        
        # 📊 ADVANCED ANALYTICS
        self.kelly_fraction_history: List[float] = []
        self.sharpe_ratio_rolling: List[float] = []
        self.max_drawdown_periods: List[Dict] = []
        
        # Performance tracking
        self.last_portfolio_update: datetime = datetime.now(timezone.utc)
        self.performance_snapshots: List[Dict] = []
        
        logger.info(f"💰 MOMENTUM OPTIMIZED Portfolio initialized with ${initial_capital_usdt:,.2f} USDT")

    def track_portfolio_value(self, current_price: float) -> None:
        """📊 Enhanced portfolio value tracking - FIXED DATA FORMAT"""
        try:
            current_time = datetime.now(timezone.utc)
            current_value = self.get_total_portfolio_value_usdt(current_price)
            
            # FIXED: Store only numeric values in portfolio_value_history
            self.portfolio_value_history.append(current_value)  # Only float, no tuple!
            
            # Store timestamps separately if needed
            if not hasattr(self, 'portfolio_timestamps'):
                self.portfolio_timestamps = []
            self.portfolio_timestamps.append(current_time)
            
            # Calculate daily P&L (using only values)
            if len(self.portfolio_value_history) > 96:  # 24 hours of 15min bars
                value_24h_ago = self.portfolio_value_history[-96]
                daily_pnl = current_value - value_24h_ago
                if not hasattr(self, 'daily_pnl_values'):
                    self.daily_pnl_values = []
                self.daily_pnl_values.append(daily_pnl)
            
            # Keep last 30 days of data
            if len(self.portfolio_value_history) > 96 * 30:
                self.portfolio_value_history = self.portfolio_value_history[-96 * 30:]
                if hasattr(self, 'portfolio_timestamps'):
                    self.portfolio_timestamps = self.portfolio_timestamps[-96 * 30:]
            
            # Update advanced metrics
            self._update_advanced_metrics(current_value)
            
            self.last_portfolio_update = current_time
            
        except Exception as e:
            logger.error(f"Portfolio value tracking error: {e}")

    def _update_advanced_metrics(self, current_value: float):
        """Calculate advanced portfolio metrics - FIXED"""
        try:
            if len(self.portfolio_value_history) < 10:
                return
            
            # FIXED: Direct use of numeric values
            values = self.portfolio_value_history[-30:]  # Already numeric!
            
            if len(values) >= 2:
                returns = [
                    (values[i] - values[i-1]) / values[i-1] 
                    for i in range(1, len(values)) 
                    if values[i-1] != 0
                ]
                
                if returns:
                    # Calculate Sharpe ratio (simplified)
                    mean_return = sum(returns) / len(returns)
                    if len(returns) > 1:
                        std_return = (sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
                        if std_return > 0:
                            sharpe = (mean_return / std_return) * (96 ** 0.5)  # Annualized
                            self.sharpe_ratio_rolling.append(sharpe)
                            
                            # Keep last 100 Sharpe calculations
                            if len(self.sharpe_ratio_rolling) > 100:
                                self.sharpe_ratio_rolling = self.sharpe_ratio_rolling[-100:]
            
            # Analyze drawdowns
            self._analyze_drawdowns(values)
            
        except Exception as e:
            logger.debug(f"Advanced metrics error: {e}")

    def _calculate_kelly_fraction(self) -> float:
        """🎯 Calculate optimal Kelly Fraction"""
        try:
            if len(self.closed_trades) < 10:
                return 0.1
            
            recent_trades = self.closed_trades[-30:]  # Last 30 trades
            wins = [t for t in recent_trades if t.get('pnl_usdt', 0) > 0]
            losses = [t for t in recent_trades if t.get('pnl_usdt', 0) <= 0]
            
            if not wins or not losses:
                return 0.05  # Conservative default
            
            win_rate = len(wins) / len(recent_trades)
            avg_win = np.mean([t['pnl_usdt'] for t in wins])
            avg_loss = abs(np.mean([t['pnl_usdt'] for t in losses]))
            
            if avg_loss == 0:
                return 0.05
            
            # Kelly formula: f = (bp - q) / b
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            return max(0.01, min(0.25, kelly_fraction))  # Cap between 1% and 25%
            
        except Exception as e:
            logger.debug(f"Kelly fraction calculation error: {e}")
            return 0.1

    def _analyze_drawdowns(self, values: List[float]):
        """Analyze drawdown periods"""
        try:
            if len(values) < 10:
                return
            
            peak = values[0]
            current_dd_start = 0
            
            for i, value in enumerate(values):
                if value > peak:
                    # New peak, end any current drawdown
                    if i > current_dd_start:
                        dd_depth = (peak - min(values[current_dd_start:i])) / peak
                        if dd_depth > 0.02:  # Only track significant drawdowns (>2%)
                            self.max_drawdown_periods.append({
                                "start_idx": current_dd_start,
                                "end_idx": i,
                                "depth": dd_depth,
                                "duration": i - current_dd_start
                            })
                    peak = value
                    current_dd_start = i
            
            # Keep only last 10 drawdown periods
            if len(self.max_drawdown_periods) > 10:
                self.max_drawdown_periods = self.max_drawdown_periods[-10:]
                
        except Exception as e:
            logger.debug(f"Drawdown analysis error: {e}")

    def get_available_usdt(self) -> float:
        return self.available_usdt

    def get_total_portfolio_value_usdt(self, current_btc_price: float) -> float:
        """💎 Enhanced portfolio value calculation"""
        try:
            asset_value = 0
            main_base_currency = settings.SYMBOL.split('/')[0]

            for pos in self.positions:
                position_base_currency = pos.symbol.split('/')[0]
                if position_base_currency == main_base_currency:
                    # Update position metrics while calculating
                    pos.update_performance_metrics(current_btc_price)
                    asset_value += abs(pos.quantity_btc) * current_btc_price
            
            total_value = self.available_usdt + asset_value
            return total_value
            
        except Exception as e:
            logger.error(f"Portfolio value calculation error: {e}")
            return self.available_usdt

    def get_open_positions(self, symbol: Optional[str] = None, strategy_name: Optional[str] = None) -> List[Position]:
        filtered_positions = self.positions
        if symbol:
            filtered_positions = [pos for pos in filtered_positions if pos.symbol == symbol]
        if strategy_name:
            filtered_positions = [pos for pos in filtered_positions if pos.strategy_name == strategy_name]
        return filtered_positions

    def calculate_optimal_position_size(self, current_price: float, quality_score: int, 
                                      market_regime: str = "UNKNOWN") -> float:
        """🎯 Calculate optimal position size using Kelly + Quality scoring"""
        try:
            available_usdt = self.get_available_usdt()
            
            # Base size from Kelly Fraction
            kelly_fraction = self._calculate_kelly_fraction()
            base_size = available_usdt * kelly_fraction
            
            # Quality score multiplier (higher quality = larger position)
            quality_multiplier = 1.0
            if quality_score >= 20:
                quality_multiplier = 1.8
            elif quality_score >= 16:
                quality_multiplier = 1.5
            elif quality_score >= 12:
                quality_multiplier = 1.2
            elif quality_score >= 8:
                quality_multiplier = 1.0
            else:
                quality_multiplier = 0.7
            
            # Market regime multiplier
            regime_multiplier = 1.0
            if market_regime == "BULL_TRENDING":
                regime_multiplier = 1.3
            elif market_regime == "BEAR_TRENDING":
                regime_multiplier = 0.6
            elif market_regime == "VOLATILE_EXPANSION":
                regime_multiplier = 0.8
            elif market_regime == "SIDEWAYS_CONSOLIDATION":
                regime_multiplier = 0.9
            
            # Portfolio performance adjustment
            portfolio_performance = self.get_portfolio_performance_multiplier()
            
            # Final position size
            optimal_size = base_size * quality_multiplier * regime_multiplier * portfolio_performance
            
            # Apply limits
            optimal_size = max(settings.MOMENTUM_MIN_POSITION_USDT, 
                             min(optimal_size, settings.MOMENTUM_MAX_POSITION_USDT))
            
            # Safety check
            if optimal_size > available_usdt * 0.9:
                optimal_size = available_usdt * 0.9
            
            logger.debug(f"💰 Optimal Position: Kelly={kelly_fraction:.3f}, Quality={quality_multiplier:.2f}x, "
                        f"Regime={regime_multiplier:.2f}x, Performance={portfolio_performance:.2f}x, "
                        f"Final=${optimal_size:.2f}")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return settings.MOMENTUM_MIN_POSITION_USDT

    def get_portfolio_performance_multiplier(self) -> float:
        """Calculate position size multiplier based on recent performance"""
        try:
            if len(self.closed_trades) < 5:
                return 1.0
            
            # Look at last 10 trades
            recent_trades = self.closed_trades[-10:]
            recent_pnl = sum(t.get('pnl_usdt', 0) for t in recent_trades)
            
            # Calculate multiplier based on recent performance
            if recent_pnl > 50:  # Very good recent performance
                return 1.3
            elif recent_pnl > 20:  # Good performance
                return 1.1
            elif recent_pnl > -10:  # Neutral
                return 1.0
            elif recent_pnl > -30:  # Poor performance
                return 0.8
            else:  # Very poor performance
                return 0.6
                
        except Exception as e:
            return 1.0

    async def execute_buy(
        self,
        strategy_name: str,
        symbol: str,
        current_price: float,
        timestamp: str, 
        reason: str,    
        amount_usdt_override: Optional[float] = None,
        stop_loss_price_from_strategy: Optional[float] = None,
        buy_context: Optional[Dict[str, Any]] = None 
    ) -> Optional[Position]:
        """🚀 Enhanced buy execution with advanced analytics"""
        try:
            # Use optimal position sizing if not overridden
            if amount_usdt_override is None:
                quality_score = buy_context.get('quality_score', 10) if buy_context else 10
                market_regime = buy_context.get('market_regime', {}).get('regime', 'UNKNOWN') if buy_context else 'UNKNOWN'
                gross_spend_usdt = self.calculate_optimal_position_size(current_price, quality_score, market_regime)
            else:
                gross_spend_usdt = amount_usdt_override
            
            if gross_spend_usdt <= 0:
                logger.warning(f"Buy amount must be positive. Received: ${gross_spend_usdt:.2f}")
                return None

            fee_usdt = gross_spend_usdt * settings.FEE_BUY
            total_cost_usdt = gross_spend_usdt + fee_usdt

            if self.available_usdt < total_cost_usdt:
                logger.warning(f"Insufficient balance: Have ${self.available_usdt:.2f}, Need ${total_cost_usdt:.2f}")
                return None
            
            if current_price <= 0:
                logger.error(f"Invalid current price for BUY: ${current_price:.2f}")
                return None
                
            quantity_asset_bought = gross_spend_usdt / current_price
            
            # Enhanced position creation
            position = Position(
                position_id=str(uuid.uuid4()),
                strategy_name=strategy_name,
                symbol=symbol,
                quantity_btc=quantity_asset_bought,
                entry_price=current_price,
                entry_cost_usdt_total=total_cost_usdt,
                timestamp=timestamp, 
                stop_loss_price=stop_loss_price_from_strategy,
                entry_context=buy_context or {}
            )
            
            self.available_usdt -= total_cost_usdt
            self.positions.append(position)
            
            # Enhanced logging
            reason_detailed = f"BUY ({reason})"
            if buy_context:
                quality_score = buy_context.get('quality_score', 0)
                ai_approved = buy_context.get('ai_approved', 'N/A')
                reason_detailed = f"BUY_Q{quality_score}_AI:{ai_approved} ({reason})"
            
            await self._log_trade_to_file(
                action="BUY", position=position, price=current_price,
                gross_value_usdt=gross_spend_usdt, net_value_usdt=total_cost_usdt,      
                fee_usdt=fee_usdt, pnl_usdt_trade=0.0, hold_duration_min=0.0,     
                reason_detailed=reason_detailed
            )
            
            base_currency = symbol.split('/')[0]
            logger.info(f"✅ {reason_detailed}: ${total_cost_usdt:.2f} for {quantity_asset_bought:.8f} {base_currency} "
                       f"@ ${current_price:.2f} | Fee: ${fee_usdt:.4f} | Available: ${self.available_usdt:.2f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Buy execution error: {e}", exc_info=True)
            return None

    async def execute_sell(
        self,
        position_to_close: Position,
        current_price: float,
        timestamp: str, 
        reason: str,    
        sell_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """💎 Enhanced sell execution with advanced analytics"""
        try:
            if current_price <= 0:
                logger.error(f"Invalid current price for SELL: ${current_price:.2f}")
                return False

            quantity_asset_sold = abs(position_to_close.quantity_btc)
            gross_proceeds_usdt = quantity_asset_sold * current_price
            fee_usdt = gross_proceeds_usdt * settings.FEE_SELL
            net_proceeds_usdt = gross_proceeds_usdt - fee_usdt
            
            profit_usdt = net_proceeds_usdt - position_to_close.entry_cost_usdt_total 
            
            # Calculate hold time
            hold_duration_min = 0.0
            try:
                entry_time_dt = datetime.fromisoformat(position_to_close.timestamp.replace('Z', '+00:00'))
                exit_time_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hold_duration_min = (exit_time_dt - entry_time_dt).total_seconds() / 60
                position_to_close.exit_time_iso = timestamp 
                position_to_close.hold_time_minutes = hold_duration_min
            except Exception as e_time:
                logger.warning(f"Time calculation error for sell log: {e_time}")
            
            # Update portfolio
            self.cumulative_pnl += profit_usdt
            self.available_usdt += net_proceeds_usdt
            
            if position_to_close in self.positions:
                 self.positions.remove(position_to_close)

            profit_pct_calc = (profit_usdt / position_to_close.entry_cost_usdt_total) * 100 if position_to_close.entry_cost_usdt_total > 0 else 0
            
            # Enhanced closed trade record
            closed_trade_info = {
                "position_id": position_to_close.position_id,
                "strategy_name": position_to_close.strategy_name,
                "symbol": position_to_close.symbol,
                "entry_timestamp": position_to_close.timestamp,
                "exit_timestamp": timestamp,
                "entry_price": position_to_close.entry_price,
                "exit_price": current_price,
                "quantity_asset": quantity_asset_sold,
                "entry_cost_total_usdt": position_to_close.entry_cost_usdt_total,
                "gross_proceeds_usdt": gross_proceeds_usdt,
                "fee_usdt": fee_usdt,
                "net_proceeds_usdt": net_proceeds_usdt,
                "pnl_usdt": profit_usdt,
                "pnl_pct": profit_pct_calc,
                "hold_duration_min": hold_duration_min,
                "reason_detailed": f"SELL ({reason})",
                "entry_context": position_to_close.entry_context,
                "sell_context": sell_context or {},
                
                # 🚀 MOMENTUM ENHANCED METRICS
                "quality_score": position_to_close.quality_score,
                "ai_approved": position_to_close.ai_approved,
                "max_profit_seen": position_to_close.max_profit_usd,
                "max_drawdown": position_to_close.max_drawdown_from_peak,
                "highest_price": position_to_close.highest_price_seen,
                "momentum_strength": position_to_close.momentum_entry_strength,
                "entry_regime": position_to_close.regime_at_entry,
                "risk_metrics": position_to_close.risk_metrics
            }
            self.closed_trades.append(closed_trade_info)
            
            # Update momentum-specific stats
            self._update_momentum_stats(closed_trade_info)

            await self._log_trade_to_file(
                action="SELL", position=position_to_close, price=current_price,
                gross_value_usdt=gross_proceeds_usdt, net_value_usdt=net_proceeds_usdt,   
                fee_usdt=fee_usdt, pnl_usdt_trade=profit_usdt, hold_duration_min=hold_duration_min,
                reason_detailed=f"SELL ({reason})"
            )
            
            # Enhanced logging
            profit_emoji = "💎" if profit_usdt > 5 else "💰" if profit_usdt > 0 else "📉" if profit_usdt < 0 else "⚖️"
            quality_info = f"Q{position_to_close.quality_score}" if hasattr(position_to_close, 'quality_score') else ""
            
            logger.info(f"✅ SELL ({reason}): {quality_info} | "
                       f"P&L: {profit_usdt:+.2f} USDT ({profit_pct_calc:+.2f}%) | "
                       f"Hold: {hold_duration_min:.1f}min | Available: ${self.available_usdt:.2f} {profit_emoji}")
            
            return True
            
        except Exception as e:
            logger.error(f"Sell execution error: {e}", exc_info=True)
            return False

    def _update_momentum_stats(self, trade_info: Dict[str, Any]):
        """Update momentum-specific statistics"""
        try:
            if trade_info.get('strategy_name') != 'EnhancedMomentum':
                return
            
            self.momentum_trade_stats["total_momentum_trades"] += 1
            
            pnl = trade_info.get('pnl_usdt', 0)
            
            # Update best/worst
            if pnl > self.momentum_trade_stats["best_momentum_trade"]:
                self.momentum_trade_stats["best_momentum_trade"] = pnl
            if pnl < self.momentum_trade_stats["worst_momentum_trade"]:
                self.momentum_trade_stats["worst_momentum_trade"] = pnl
            
            # Calculate win rate and avg profit
            momentum_trades = [t for t in self.closed_trades if t.get('strategy_name') == 'EnhancedMomentum']
            if momentum_trades:
                wins = [t for t in momentum_trades if t.get('pnl_usdt', 0) > 0]
                losses = [t for t in momentum_trades if t.get('pnl_usdt', 0) <= 0]
                
                self.momentum_trade_stats["momentum_win_rate"] = len(wins) / len(momentum_trades) * 100
                self.momentum_trade_stats["momentum_avg_profit"] = np.mean([t['pnl_usdt'] for t in momentum_trades])
                
                # Profit factor
                total_wins = sum(t['pnl_usdt'] for t in wins)
                total_losses = abs(sum(t['pnl_usdt'] for t in losses))
                self.momentum_trade_stats["momentum_profit_factor"] = total_wins / total_losses if total_losses > 0 else float('inf')
            
        except Exception as e:
            logger.debug(f"Momentum stats update error: {e}")
    
    async def _log_trade_to_file(
        self,
        action: str,
        position: Position,
        price: float, 
        gross_value_usdt: float, 
        net_value_usdt: float,   
        fee_usdt: float,
        pnl_usdt_trade: float,   
        hold_duration_min: float,
        reason_detailed: str = "" 
    ) -> None:
        """Enhanced trade logging with momentum analytics"""
        try:
            if not hasattr(settings, 'TRADES_CSV_LOG_PATH') or not settings.TRADES_CSV_LOG_PATH:
                return
            if not settings.ENABLE_CSV_LOGGING:
                return
            
            from utils.logger import ensure_csv_header 
            csv_path_str = settings.TRADES_CSV_LOG_PATH
            ensure_csv_header(csv_path_str) 
                
            def safe_float(value, default=0.0, precision=8): 
                try: return round(float(value), precision) if value is not None else default
                except (ValueError, TypeError): return default
            
            def safe_str(value, default="", max_length=150):
                try:
                    clean_str = str(value).replace(',', ';').replace('\n', ' ').strip()
                    return clean_str[:max_length]
                except: return default

            # Enhanced reason with momentum data
            enhanced_reason = reason_detailed
            if hasattr(position, 'quality_score') and hasattr(position, 'ai_approved'):
                enhanced_reason += f"_Q{position.quality_score}_AI{position.ai_approved}"
            if hasattr(position, 'momentum_entry_strength'):
                enhanced_reason += f"_M{position.momentum_entry_strength:.2f}"

            action_timestamp_str = ""
            if action == "BUY":
                action_timestamp_str = position.timestamp 
            elif action == "SELL" and position.exit_time_iso: 
                 action_timestamp_str = position.exit_time_iso
            else: 
                action_timestamp_str = datetime.now(timezone.utc).isoformat()

            try:
                dt_obj = datetime.fromisoformat(action_timestamp_str.replace('Z', '+00:00'))
                formatted_timestamp = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError: 
                formatted_timestamp = action_timestamp_str[:19]

            price_fmt = f"{safe_float(price, precision=settings.PRICE_PRECISION):.{settings.PRICE_PRECISION}f}"
            quantity_asset_fmt = f"{safe_float(abs(position.quantity_btc), precision=settings.ASSET_PRECISION):.{settings.ASSET_PRECISION}f}"
            gross_value_usdt_fmt = f"{safe_float(gross_value_usdt, precision=2):.2f}"
            fee_usdt_fmt = f"{safe_float(fee_usdt, precision=4):.4f}"
            net_value_usdt_fmt = f"{safe_float(net_value_usdt, precision=2):.2f}"
            
            pnl_usdt_trade_str = f"{safe_float(pnl_usdt_trade, precision=2):.2f}" if action == 'SELL' else "0.00"
            hold_duration_min_str = f"{safe_float(hold_duration_min, precision=1):.1f}" if action == 'SELL' else "0.0"
            
            cumulative_pnl_usdt_fmt = f"{safe_float(self.cumulative_pnl, precision=2):.2f}"

            log_values = [
                safe_str(formatted_timestamp, max_length=19),      
                safe_str(position.position_id, max_length=36),      
                safe_str(position.strategy_name, max_length=25),   
                safe_str(position.symbol, max_length=15),           
                safe_str(action, max_length=4),                     
                price_fmt,                                           
                quantity_asset_fmt,                                  
                gross_value_usdt_fmt,                                 
                fee_usdt_fmt,                                        
                net_value_usdt_fmt,                                   
                safe_str(enhanced_reason),                                         
                hold_duration_min_str,       
                pnl_usdt_trade_str,     
                cumulative_pnl_usdt_fmt                
            ]
            
            log_entry = ",".join(log_values) + "\n"
            
            csv_path_obj = Path(csv_path_str) 
            with open(csv_path_obj, 'a', encoding='utf-8', newline='') as f: 
                f.write(log_entry)
                f.flush() 
                
        except Exception as e:
            logger.error(f"Trade CSV logging error: {e}", exc_info=True)

    def get_closed_trades_for_summary(self) -> List[Dict]:
        """Get closed trades for performance analysis"""
        return self.closed_trades.copy()

    def get_performance_summary(self, current_price: float) -> Dict[str, Any]: 
        """💎 Enhanced performance summary with momentum analytics"""
        try:
            initial_capital = self.initial_capital_usdt
            current_value = self.get_total_portfolio_value_usdt(current_price)
            total_profit = current_value - initial_capital
            total_profit_pct = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
            
            total_trades = len(self.closed_trades)
            
            if total_trades == 0:
                return {
                    "initial_capital": initial_capital,
                    "current_value": current_value,
                    "total_profit": total_profit,
                    "total_profit_pct": total_profit_pct,
                    "total_trades": 0,
                    "win_count": 0,
                    "loss_count": 0,
                    "win_rate": 0,
                    "total_wins": 0,
                    "total_losses": 0,
                    "profit_factor": 0,
                    "avg_trade": 0,
                    "max_win": 0,
                    "max_loss": 0,
                    "current_exposure": 0,
                    "exposure_pct": 0,
                    "open_positions": len(self.positions),
                    "available_usdt": self.available_usdt,
                    "sharpe_ratio": 0,
                    "max_drawdown_pct": 0,
                    "kelly_fraction": 0.1,
                    "momentum_stats": self.momentum_trade_stats
                }
            
            # Basic trade stats
            wins = [t for t in self.closed_trades if t.get("pnl_usdt", 0) > 0] 
            losses = [t for t in self.closed_trades if t.get("pnl_usdt", 0) <= 0] 
            
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
            
            total_wins = sum(t.get("pnl_usdt", 0) for t in wins) 
            total_losses = abs(sum(t.get("pnl_usdt", 0) for t in losses)) 
            
            profit_factor = total_wins / total_losses if total_losses > 0 else (float('inf') if total_wins > 0 else 0)
            avg_trade = total_profit / total_trades if total_trades > 0 else 0
            max_win = max((t.get("pnl_usdt", 0) for t in wins), default=0) 
            max_loss = min((t.get("pnl_usdt", 0) for t in losses), default=0) 
            
            # Portfolio metrics
            current_exposure = sum(abs(pos.quantity_btc) * current_price for pos in self.positions)
            exposure_pct = (current_exposure / current_value) * 100 if current_value > 0 else 0
            
            # 🚀 ENHANCED METRICS
            # Sharpe ratio
            sharpe_ratio = np.mean(self.sharpe_ratio_rolling) if self.sharpe_ratio_rolling else 0
            
            # Max drawdown
            max_drawdown_pct = 0
            if len(self.portfolio_value_history) > 10:
                values = [v[1] for v in self.portfolio_value_history]
                peak = values[0]
                max_dd = 0
                for value in values:
                    if value > peak:
                        peak = value
                    dd = (peak - value) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                max_drawdown_pct = max_dd * 100
            
            # Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction()
            
            # Average hold time
            avg_hold_time = 0.0
            if total_trades > 0:
                hold_times = [t.get("hold_duration_min", 0.0) for t in self.closed_trades if t.get("hold_duration_min") is not None]
                if hold_times:
                    avg_hold_time = sum(hold_times) / len(hold_times)
            
            return {
                # Basic metrics
                "initial_capital": initial_capital,
                "current_value": current_value,
                "total_profit": total_profit,
                "total_profit_pct": total_profit_pct,
                "total_trades": total_trades,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": win_rate,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "profit_factor": profit_factor,
                "avg_trade": avg_trade,
                "max_win": max_win,
                "max_loss": max_loss,
                "current_exposure": current_exposure,
                "exposure_pct": exposure_pct,
                "open_positions": len(self.positions),
                "available_usdt": self.available_usdt,
                "avg_hold_time": avg_hold_time,
                
                # 🚀 ENHANCED METRICS
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown_pct,
                "kelly_fraction": kelly_fraction,
                "momentum_stats": self.momentum_trade_stats,
                
                # Risk metrics
                "sortino_ratio": self._calculate_sortino_ratio(),
                "calmar_ratio": total_profit_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0,
                "recovery_factor": total_profit / max_drawdown_pct if max_drawdown_pct > 0 else 0,
                
                # Additional analytics
                "daily_pnl_avg": np.mean([d[1] for d in self.daily_pnl_history]) if self.daily_pnl_history else 0,
                "daily_pnl_std": np.std([d[1] for d in self.daily_pnl_history]) if self.daily_pnl_history else 0,
                "consecutive_wins": self._calculate_consecutive_wins(),
                "consecutive_losses": self._calculate_consecutive_losses()
            }
            
        except Exception as e:
            logger.error(f"Performance summary error: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if len(self.closed_trades) < 10:
                return 0.0
            
            returns = [t.get('pnl_pct', 0) / 100 for t in self.closed_trades]
            negative_returns = [r for r in returns if r < 0]
            
            if not negative_returns:
                return 10.0  # No negative returns
            
            mean_return = np.mean(returns)
            downside_std = np.std(negative_returns)
            
            if downside_std == 0:
                return 10.0
            
            # Annualize (assuming average trade every few hours)
            annualization_factor = np.sqrt(365 * 4)  # ~4 trades per day
            return (mean_return / downside_std) * annualization_factor
            
        except Exception as e:
            return 0.0

    def _calculate_consecutive_wins(self) -> int:
        """Calculate maximum consecutive wins"""
        try:
            if not self.closed_trades:
                return 0
            
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in self.closed_trades:
                if trade.get('pnl_usdt', 0) > 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            return 0

    def _calculate_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        try:
            if not self.closed_trades:
                return 0
            
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in self.closed_trades:
                if trade.get('pnl_usdt', 0) <= 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            return 0

    def __repr__(self):
        main_symbol_base = settings.SYMBOL.split('/')[0]
        asset_total = sum(abs(pos.quantity_btc) for pos in self.positions if pos.symbol.startswith(main_symbol_base))
        
        return (f"Portfolio(USDT: ${self.available_usdt:.2f}, "
                f"{main_symbol_base}: {asset_total:.{settings.ASSET_PRECISION}f}, " 
                f"OpenPos: {len(self.positions)}, "
                f"ClosedTrades: {len(self.closed_trades)}, "
                f"CumP&L: ${self.cumulative_pnl:.2f}, "
                f"Kelly: {self._calculate_kelly_fraction():.3f})")