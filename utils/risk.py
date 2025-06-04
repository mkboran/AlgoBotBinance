# utils/risk.py
from typing import Any, Dict, Optional

from utils.logger import logger
from utils.config import settings

class RiskManager:
    """🛡️ Risk Management System"""
    
    def __init__(self):
        self.max_portfolio_drawdown_pct = getattr(settings, 'GLOBAL_MAX_PORTFOLIO_DRAWDOWN_PCT', -20.0)
        self.max_daily_loss_pct = getattr(settings, 'GLOBAL_MAX_DAILY_LOSS_PCT', -10.0)
        self.max_position_exposure_pct = getattr(settings, 'GLOBAL_MAX_POSITION_EXPOSURE_PCT', 90.0)
        
        logger.info(f"🛡️ Risk Manager initialized:")
        logger.info(f"   Max Portfolio Drawdown: {self.max_portfolio_drawdown_pct}%")
        logger.info(f"   Max Daily Loss: {self.max_daily_loss_pct}%") 
        logger.info(f"   Max Position Exposure: {self.max_position_exposure_pct}%")
    
    def check_global_risk_limits(self, portfolio, current_btc_price_for_value_calc: float) -> bool:
        """Check global risk limits"""
        try:
            initial_capital = portfolio.initial_capital_usdt
            current_value = portfolio.get_total_portfolio_value_usdt(current_btc_price_for_value_calc)
            
            # Portfolio drawdown check
            drawdown_pct = ((current_value - initial_capital) / initial_capital) * 100
            if drawdown_pct <= self.max_portfolio_drawdown_pct:
                logger.critical(f"🚨 PORTFOLIO DRAWDOWN LIMIT EXCEEDED: {drawdown_pct:.2f}% <= {self.max_portfolio_drawdown_pct}%")
                return False
            
            # Position exposure check
            total_exposure = sum(abs(pos.quantity_btc) * current_btc_price_for_value_calc for pos in portfolio.positions)
            exposure_pct = (total_exposure / current_value) * 100 if current_value > 0 else 0
            
            if exposure_pct > self.max_position_exposure_pct:
                logger.warning(f"⚠️ High position exposure: {exposure_pct:.1f}% > {self.max_position_exposure_pct}%")
                # Don't stop trading, just warn
            
            return True
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return True  # Don't block trading on risk check errors
    
    def check_position_risk(self, position_amount: float, portfolio_value: float) -> bool:
        """Check individual position risk"""
        try:
            position_pct = (position_amount / portfolio_value) * 100 if portfolio_value > 0 else 0
            max_single_position_pct = 25.0  # Max 25% per position
            
            if position_pct > max_single_position_pct:
                logger.warning(f"⚠️ Position size too large: {position_pct:.1f}% > {max_single_position_pct}%")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Position risk check error: {e}")
            return True  # Don't block trading on risk check errors

if __name__ == "__main__":
    # Test code
    pass