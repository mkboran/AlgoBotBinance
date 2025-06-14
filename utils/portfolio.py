# utils/portfolio.py
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import logger 
from utils.config import settings
# import logging # logger.py'den gelen logger kullanıldığı için gereksiz

class Position:
    """Bir trading pozisyonunu temsil eden sınıf"""
    
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
        self.exit_time_iso: Optional[str] = None # Satış zamanını saklamak için
        self.portfolio_value_history: List[float] = []
    def __repr__(self):
        base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol
        return (f"Position(id={self.position_id[:8]}, strategy={self.strategy_name}, "
                f"qty={self.quantity_btc:.8f} {base_currency}, entry=${self.entry_price:.{getattr(settings, 'PRICE_PRECISION', 2)}f})")


        

        
class Portfolio:
    """📊 Portfolio Management System"""
    
    def __init__(self, initial_capital_usdt: float):
        self.initial_capital_usdt = initial_capital_usdt
        self.available_usdt = initial_capital_usdt
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = [] 
        self.cumulative_pnl = 0.0
        
        # EKSİK OLAN PORTFOLIO VALUE HISTORY EKLENDİ
        self.portfolio_value_history: List[float] = []
        
        self.open_positions = self.positions
        
        logger.info(f"✅ Portfolio initialized with ${initial_capital_usdt:,.2f} USDT")

    def track_portfolio_value(self, current_price: float) -> None:
        """Portfolio değerini history'ye kaydet"""
        current_value = self.get_total_portfolio_value_usdt(current_price)
        self.portfolio_value_history.append(current_value)

    def get_closed_trades_for_summary(self) -> List[Dict]:
        """Kapalı işlemleri özet için döndür"""
        return self.closed_trades.copy()

    def get_available_usdt(self) -> float:
        return self.available_usdt

    def get_total_portfolio_value_usdt(self, current_btc_price: float) -> float:
        try:
            asset_value = 0
            main_base_currency = settings.SYMBOL.split('/')[0]

            for pos in self.positions:
                position_base_currency = pos.symbol.split('/')[0]
                if position_base_currency == main_base_currency:
                    asset_value += abs(pos.quantity_btc) * current_btc_price
            
            total_value = self.available_usdt + asset_value
            return total_value
        except Exception as e:
            logger.error(f"Portfolio value calculation error: {e}", exc_info=True)
            return self.available_usdt

    def get_open_positions(self, symbol: Optional[str] = None, strategy_name: Optional[str] = None) -> List[Position]:
        filtered_positions = self.positions
        if symbol:
            filtered_positions = [pos for pos in filtered_positions if pos.symbol == symbol]
        if strategy_name:
            filtered_positions = [pos for pos in filtered_positions if pos.strategy_name == strategy_name]
        return filtered_positions

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
        try:
            gross_spend_usdt = amount_usdt_override if amount_usdt_override is not None \
                               else settings.MOMENTUM_MAX_POSITION_USDT
            
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
            
            reason_detailed = f"BUY ({reason})"
            if buy_context:
                reason_detailed = f"BUY ({buy_context.get('reason', reason)})"
                if "quality_score" in buy_context and "ai_approved" in buy_context:
                     reason_detailed = f"BUY ({buy_context.get('reason', reason)})_Q{buy_context['quality_score']}_AI:{buy_context['ai_approved']}"
            await self._log_trade_to_file(
                action="BUY", position=position, price=current_price,
                gross_value_usdt=gross_spend_usdt, net_value_usdt=total_cost_usdt,      
                fee_usdt=fee_usdt, pnl_usdt_trade=0.0, hold_duration_min=0.0,     
                reason_detailed=reason_detailed
            )
            
            base_currency = symbol.split('/')[0]
            price_precision = settings.PRICE_PRECISION
            asset_precision = settings.ASSET_PRECISION
            logger.info(f"✅ {reason_detailed}: {strategy_name} | "
                       f"Cost: ${total_cost_usdt:.{price_precision}f} for {quantity_asset_bought:.{asset_precision}f} {base_currency} @ ${current_price:.{price_precision}f} | "
                       f"Fee: ${fee_usdt:.4f} | Avail USDT: ${self.available_usdt:.2f}")
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
        try:
            if current_price <= 0:
                logger.error(f"Invalid current price for SELL: ${current_price:.2f}")
                return False

            quantity_asset_sold = abs(position_to_close.quantity_btc)
            gross_proceeds_usdt = quantity_asset_sold * current_price
            fee_usdt = gross_proceeds_usdt * settings.FEE_SELL
            net_proceeds_usdt = gross_proceeds_usdt - fee_usdt
            
            profit_usdt = net_proceeds_usdt - position_to_close.entry_cost_usdt_total 
            
            hold_duration_min = 0.0
            try:
                entry_time_dt = datetime.fromisoformat(position_to_close.timestamp.replace('Z', '+00:00'))
                exit_time_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hold_duration_min = (exit_time_dt - entry_time_dt).total_seconds() / 60
                position_to_close.exit_time_iso = timestamp 
            except Exception as e_time:
                logger.warning(f"Time calculation error for sell log: {e_time}")
            
            self.cumulative_pnl += profit_usdt
            self.available_usdt += net_proceeds_usdt
            
            if position_to_close in self.positions:
                 self.positions.remove(position_to_close)

            profit_pct_calc = (profit_usdt / position_to_close.entry_cost_usdt_total) * 100 if position_to_close.entry_cost_usdt_total > 0 else 0
            
            reason_detailed = f"SELL ({reason})"

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
                "reason_detailed": reason_detailed,
                "entry_context": position_to_close.entry_context,
                "sell_context": sell_context or {}
            }
            self.closed_trades.append(closed_trade_info)

            await self._log_trade_to_file(
                action="SELL", position=position_to_close, price=current_price,
                gross_value_usdt=gross_proceeds_usdt, net_value_usdt=net_proceeds_usdt,   
                fee_usdt=fee_usdt, pnl_usdt_trade=profit_usdt, hold_duration_min=hold_duration_min,
                reason_detailed=reason_detailed
            )
            
            profit_emoji = "💎" if profit_usdt > 2 else "💰" if profit_usdt > 0 else "📉" if profit_usdt < 0 else "⚖️"
            price_precision = settings.PRICE_PRECISION
            logger.info(f"✅ {reason_detailed}: {position_to_close.strategy_name} | "
                       f"P&L: {profit_usdt:+.{price_precision}f} USDT ({profit_pct_calc:+.2f}%) | "
                       f"Hold: {hold_duration_min:.1f}min | Avail USDT: ${self.available_usdt:.2f} {profit_emoji}")
            return True
        except Exception as e:
            logger.error(f"Sell execution error (Position ID: {position_to_close.position_id if position_to_close else 'N/A'}): {e}", exc_info=True)
            return False
    
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
        try:
            if not hasattr(settings, 'TRADES_CSV_LOG_PATH') or not settings.TRADES_CSV_LOG_PATH:
                logger.debug("TRADES_CSV_LOG_PATH not configured. Skipping CSV log.")
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

            action_timestamp_str = ""
            if action == "BUY":
                action_timestamp_str = position.timestamp 
            elif action == "SELL" and position.exit_time_iso: 
                 action_timestamp_str = position.exit_time_iso
            else: 
                logger.warning(f"Could not determine action timestamp for {action} on position {position.position_id}, using current time.")
                action_timestamp_str = datetime.now(timezone.utc).isoformat()

            try:
                dt_obj = datetime.fromisoformat(action_timestamp_str.replace('Z', '+00:00'))
                formatted_timestamp = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError: 
                logger.warning(f"Could not parse timestamp '{action_timestamp_str}' to YYYY-MM-DD HH:MM:SS, using raw substring.")
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
                safe_str(reason_detailed),                                         
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

    def get_performance_summary(self, current_price: float) -> Dict[str, Any]: 
        try:
            initial_capital = self.initial_capital_usdt
            current_value = self.get_total_portfolio_value_usdt(current_price)
            total_profit = current_value - initial_capital
            total_profit_pct = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
            
            total_trades = len(self.closed_trades)
            avg_hold_time = 0.0
            if total_trades > 0:
                hold_times = [t.get("hold_duration_min", 0.0) for t in self.closed_trades if t.get("hold_duration_min") is not None]
                if hold_times:
                    avg_hold_time = sum(hold_times) / len(hold_times)
            
            if total_trades == 0:
                return {
                    "initial_capital": initial_capital, "current_value": current_value,
                    "total_profit": total_profit, "total_profit_pct": total_profit_pct,
                    "total_trades": 0, "win_count": 0, "loss_count": 0, "win_rate": 0,
                    "total_wins": 0, "total_losses": 0, "profit_factor": 0, "avg_trade": 0,
                    "max_win": 0, "max_loss": 0, "current_exposure": 0, "exposure_pct": 0,
                    "open_positions": len(self.positions), "available_usdt": self.available_usdt,
                    "avg_hold_time": avg_hold_time 
                }
            
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
            
            current_exposure = sum(abs(pos.quantity_btc) * current_price for pos in self.positions)
            exposure_pct = (current_exposure / current_value) * 100 if current_value > 0 else 0
            
            return {
                "initial_capital": initial_capital, "current_value": current_value,
                "total_profit": total_profit, "total_profit_pct": total_profit_pct,
                "total_trades": total_trades, "win_count": win_count, "loss_count": loss_count,
                "win_rate": win_rate, "total_wins": total_wins, "total_losses": total_losses,
                "profit_factor": profit_factor, "avg_trade": avg_trade, "max_win": max_win,
                "max_loss": max_loss, "current_exposure": current_exposure, "exposure_pct": exposure_pct,
                "open_positions": len(self.positions), "available_usdt": self.available_usdt,
                "avg_hold_time": avg_hold_time 
            }
        except Exception as e:
            logger.error(f"Performance summary error: {e}", exc_info=True)
            return {"error": str(e)}

    def __repr__(self):
        main_symbol_base = settings.SYMBOL.split('/')[0]
        base_currency_display = main_symbol_base

        if self.positions: 
            base_currency_display = self.positions[0].symbol.split('/')[0]
        elif self.closed_trades: 
             base_currency_display = self.closed_trades[-1]['symbol'].split('/')[0]

        asset_total = sum(abs(pos.quantity_btc) for pos in self.positions if pos.symbol.startswith(main_symbol_base))
        return (f"Portfolio(USDT: ${self.available_usdt:.2f}, "
                f"{base_currency_display}: {asset_total:.{settings.ASSET_PRECISION}f}, " 
                f"OpenPos: {len(self.positions)}, "
                f"ClosedTrades: {len(self.closed_trades)}, "
                f"CumP&L: ${self.cumulative_pnl:.2f})")