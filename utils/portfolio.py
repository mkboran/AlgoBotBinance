# utils/portfolio.py
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import logger
from utils.config import settings

class Position:
    """Bir trading pozisyonunu temsil eden sınıf"""
    
    def __init__(
        self,
        position_id: str,
        strategy_name: str,
        symbol: str,
        quantity_btc: float,
        entry_price: float,
        entry_cost_usdt_with_fee: float,
        timestamp: str,
        stop_loss_price: Optional[float] = None,
        entry_context: Optional[Dict[str, Any]] = None
    ):
        self.position_id = position_id
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.quantity_btc = quantity_btc
        self.entry_price = entry_price
        self.entry_cost_usdt_with_fee = entry_cost_usdt_with_fee
        self.timestamp = timestamp
        self.stop_loss_price = stop_loss_price
        self.entry_context = entry_context or {}

    def __repr__(self):
        return (f"Position(id={self.position_id[:8]}, strategy={self.strategy_name}, "
                f"qty={self.quantity_btc:.6f} BTC, entry=${self.entry_price:.2f})")

class Portfolio:
    """📊 Portfolio Management System"""
    
    def __init__(self, initial_capital_usdt: float):
        self.initial_capital_usdt = initial_capital_usdt
        self.available_usdt = initial_capital_usdt
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.cumulative_pnl = 0.0  # Kümülatif kar/zarar takibi
        
        # open_positions property for compatibility
        self.open_positions = self.positions
        
        logger.info(f"✅ Portfolio initialized with ${initial_capital_usdt:,.2f} USDT")

    def get_available_usdt(self) -> float:
        """Kullanılabilir USDT miktarını döndür"""
        return self.available_usdt

    def get_total_portfolio_value_usdt(self, current_btc_price: float) -> float:
        """Toplam portfolio değerini hesapla"""
        try:
            btc_value = sum(abs(pos.quantity_btc) * current_btc_price for pos in self.positions)
            total_value = self.available_usdt + btc_value
            return total_value
        except Exception as e:
            logger.error(f"Portfolio value calculation error: {e}")
            return self.available_usdt

    def get_open_positions(self, symbol: Optional[str] = None, strategy_name: Optional[str] = None) -> List[Position]:
        """Açık pozisyonları filtrele ve döndür"""
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
        """Execute buy order"""
        try:
            # Position amount calculation
            if amount_usdt_override is not None:
                position_amount = amount_usdt_override
            else:
                position_amount = getattr(settings, 'trade_amount_usdt', 190.0)
            
            # Balance check
            if self.available_usdt < position_amount:
                logger.warning(f"Insufficient balance: {self.available_usdt:.2f} < {position_amount:.2f}")
                return None
            
            # Calculate quantities and fees
            gross_btc_amount = position_amount / current_price
            fee_amount = position_amount * settings.FEE_BUY
            net_btc_amount = gross_btc_amount - (fee_amount / current_price)
            
            # Create position
            position = Position(
                position_id=str(uuid.uuid4()),
                strategy_name=strategy_name,
                symbol=symbol,
                quantity_btc=net_btc_amount,
                entry_price=current_price,
                entry_cost_usdt_with_fee=position_amount,
                timestamp=timestamp,
                stop_loss_price=stop_loss_price_from_strategy,
                entry_context=buy_context
            )
            
            # Update portfolio
            self.available_usdt -= position_amount
            self.positions.append(position)
            
            # Determine quality from context
            quality = "UNKNOWN"
            if buy_context and "quality_score" in buy_context:
                quality = f"Q{buy_context['quality_score']}"
            
            # ✅ FIXED: Parameter name corrected to match function signature
            await self._log_trade_to_file(
                action="BUY",
                position=position,
                price=current_price,
                amount_usdt=position_amount,
                fee_usd=fee_amount,
                pnl_usd=0.0,
                reason=reason,
                extra_info=quality,
                duration_min=0.0,
                profit_pct=0.0
            )
            
            # Terminal log
            logger.info(f"✅ BUY ({reason}): {strategy_name} | "
                       f"${position_amount:.2f} → {net_btc_amount:.6f} BTC @ ${current_price:.2f} | "
                       f"Balance: ${self.available_usdt:.2f}")

            return position
            
        except Exception as e:
            logger.error(f"Buy execution error: {e}")
            return None

    async def execute_sell(
        self,
        position_to_close: Position,
        current_price: float,
        timestamp: str,
        reason: str,
        sell_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Execute sell order"""
        try:
            gross_proceeds = abs(position_to_close.quantity_btc) * current_price
            fee_amount = gross_proceeds * settings.FEE_SELL
            net_proceeds = gross_proceeds - fee_amount
            
            profit_usd = net_proceeds - position_to_close.entry_cost_usdt_with_fee
            profit_pct = profit_usd / position_to_close.entry_cost_usdt_with_fee
            
            # Position age calculation
            try:
                entry_time = datetime.fromisoformat(position_to_close.timestamp.replace('Z', '+00:00'))
                current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hold_minutes = (current_time - entry_time).total_seconds() / 60
            except Exception as e:
                logger.warning(f"Time calculation error: {e}")
                hold_minutes = 0.0
            
            # Update cumulative P&L
            self.cumulative_pnl += profit_usd

            # Update portfolio
            self.available_usdt += net_proceeds
            self.positions.remove(position_to_close)
            
            # Record closed trade
            trade_record = {
                "position_id": position_to_close.position_id,
                "strategy": position_to_close.strategy_name,
                "symbol": position_to_close.symbol,
                "entry_price": position_to_close.entry_price,
                "exit_price": current_price,
                "quantity_btc": position_to_close.quantity_btc,
                "entry_cost": position_to_close.entry_cost_usdt_with_fee,
                "exit_proceeds": net_proceeds,
                "profit_usd": profit_usd,
                "profit_pct": profit_pct,
                "hold_minutes": hold_minutes,
                "entry_time": position_to_close.timestamp,
                "exit_time": timestamp,
                "exit_reason": reason
            }
            self.closed_trades.append(trade_record)
            
            # Log to file
            await self._log_trade_to_file(
                action="SELL",
                position=position_to_close,
                price=current_price,
                amount_usdt=net_proceeds,
                fee_usd=fee_amount,
                pnl_usd=profit_usd,
                reason=reason,
                extra_info=sell_context.get("quality", "") if sell_context else "",
                duration_min=hold_minutes,
                profit_pct=profit_pct
            )
            
            # Terminal log
            profit_emoji = "💎" if profit_usd > 2 else "💰" if profit_usd > 0 else "📉"
            logger.info(f"✅ SELL ({reason}): {position_to_close.strategy_name} | "
                       f"{profit_usd:+.2f} USD ({profit_pct*100:+.2f}%) | "
                       f"Hold: {hold_minutes:.0f}min | Balance: ${self.available_usdt:.2f} {profit_emoji}")
            
            return True
            
        except Exception as e:
            logger.error(f"Sell execution error: {e}")
            return False
    
    async def _log_trade_to_file(
        self,
        action: str,
        position: Position,
        price: float,
        amount_usdt: float,
        fee_usd: float,
        pnl_usd: float,
        reason: str = "",
        extra_info: str = "",
        duration_min: float = 0.0,
        profit_pct: float = 0.0
    ) -> None:
        """💎 SADELEŞTİRİLMİŞ CSV LOGGING - Temiz ve Anlaşılabilir Format"""
        try:
            if not hasattr(settings, 'TRADES_CSV_LOG_PATH') or not settings.TRADES_CSV_LOG_PATH:
                return
            
            from utils.logger import ensure_csv_header
            csv_path = settings.TRADES_CSV_LOG_PATH
            
            # CSV header kontrolü
            ensure_csv_header(csv_path)
                
            # Portfolio value after trade
            try:
                portfolio_value = self.available_usdt + sum(abs(pos.quantity_btc) * price for pos in self.positions)
            except Exception:
                portfolio_value = self.available_usdt
            
            # Güvenli değer dönüştürme
            def safe_float(value, default=0.0, precision=2):
                try:
                    return round(float(value) if value is not None else default, precision)
                except (ValueError, TypeError):
                    return default
            
            def safe_str(value, default="", max_length=15):
                try:
                    clean_str = str(value).replace(',', ';').replace('\n', ' ').strip()
                    return clean_str[:max_length] if len(clean_str) > max_length else clean_str
                except Exception:
                    return default
            
            # 📊 SADELEŞTİRİLMİŞ FORMAT - 12 sütun
            timestamp = safe_str(position.timestamp[:19])  # Sadece tarih-saat
            action_type = safe_str(action)
            trade_price = safe_float(price, precision=2)
            btc_amount = safe_float(abs(position.quantity_btc), precision=6)
            usdt_amount = safe_float(amount_usdt, precision=2)
            fee_amount = safe_float(fee_usd, precision=3)
            
            # P&L hesaplamaları
            pnl_amount = safe_float(pnl_usd, precision=2)
            pnl_percentage = safe_float(profit_pct * 100, precision=2)
            cumulative_pnl = safe_float(self.cumulative_pnl, precision=2)
            hold_time = safe_float(duration_min, precision=0)
            
            # Performans kategorisi
            if action == "SELL":
                if pnl_amount >= 2.0:
                    performance = "EXCELLENT"
                elif pnl_amount >= 1.0:
                    performance = "GREAT"
                elif pnl_amount >= 0.5:
                    performance = "GOOD"
                elif pnl_amount >= -0.5:
                    performance = "OK"
                elif pnl_amount >= -2.0:
                    performance = "POOR"
                else:
                    performance = "BAD"
            else:
                performance = "BUY"
            
            # Kısa reason
            short_reason = safe_str(reason.split('_')[0] if '_' in reason else reason, max_length=12)
            
            # 💎 SADECE 12 SÜTUN - Anlaşılabilir Format
            log_entry = (
                f"{timestamp},"           # 1. Tarih-Saat
                f"{action_type},"         # 2. İşlem (BUY/SELL)
                f"{trade_price},"         # 3. Fiyat
                f"{btc_amount},"          # 4. BTC Miktarı
                f"{usdt_amount},"         # 5. USDT Tutarı
                f"{fee_amount},"          # 6. Komisyon
                f"{pnl_amount},"          # 7. Kar/Zarar USD
                f"{pnl_percentage},"      # 8. Kar/Zarar %
                f"{cumulative_pnl},"     # 9. Toplam P&L
                f"{hold_time},"           # 10. Tutma Süresi (dk)
                f"{performance},"         # 11. Performans
                f"{short_reason}\n"       # 12. Sebep
            )
            
            csv_path = Path(settings.TRADES_CSV_LOG_PATH)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Dosyaya yazma
            with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                f.write(log_entry)
                f.flush()
                
        except Exception as e:
            logger.error(f"Trade logging error: {e}")
            # Fallback to simple logging
            try:
                fallback_path = Path("logs") / "trades_simple.csv"
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                with open(fallback_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()},{action},{position.position_id[:8]},{reason},{pnl_usd:.2f}\n")
            except:
                pass

    def get_performance_summary(self, current_price: float) -> Dict[str, Any]:
        """📊 Get comprehensive performance summary"""
        try:
            # Basic metrics
            initial_capital = self.initial_capital_usdt
            current_value = self.get_total_portfolio_value_usdt(current_price)
            total_profit = current_value - initial_capital
            total_profit_pct = (total_profit / initial_capital) * 100
            
            # Trade statistics
            total_trades = len(self.closed_trades)
            if total_trades == 0:
                return {
                    "initial_capital": initial_capital,
                    "current_value": current_value,
                    "total_profit": total_profit,
                    "total_profit_pct": total_profit_pct,
                    "total_trades": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "avg_trade": 0,
                    "max_win": 0,
                    "max_loss": 0,
                    "current_exposure": 0,
                    "open_positions": len(self.positions)
                }
            
            # Trade analysis
            wins = [t for t in self.closed_trades if t["profit_usd"] > 0]
            losses = [t for t in self.closed_trades if t["profit_usd"] <= 0]
            
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = (win_count / total_trades) * 100
            
            total_wins = sum(t["profit_usd"] for t in wins)
            total_losses = abs(sum(t["profit_usd"] for t in losses))
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            avg_trade = total_profit / total_trades
            max_win = max((t["profit_usd"] for t in wins), default=0)
            max_loss = min((t["profit_usd"] for t in losses), default=0)
            
            # Current exposure
            current_exposure = sum(abs(pos.quantity_btc) * current_price for pos in self.positions)
            exposure_pct = (current_exposure / current_value) * 100 if current_value > 0 else 0
            
            return {
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
                "available_usdt": self.available_usdt
            }
            
        except Exception as e:
            logger.error(f"Performance summary error: {e}")
            return {"error": str(e)}

    def __repr__(self):
        btc_total = sum(abs(pos.quantity_btc) for pos in self.positions)
        return (f"Portfolio(USDT: ${self.available_usdt:.2f}, "
                f"BTC: {btc_total:.6f}, "
                f"Positions: {len(self.positions)}, "
                f"Trades: {len(self.closed_trades)})")