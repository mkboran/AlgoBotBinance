# utils/profit_taking.py - Yeni dosya oluştur

"""
💰 Advanced Profit Taking System
Kademeli kar alma, trailing stop, partial sell sistemi
"""

from typing import Dict, Optional, Tuple, List
from datetime import datetime, timezone
from utils.logger import logger
from utils.config import settings

class AdvancedProfitTaking:
    """Gelişmiş kar alma sistemi"""
    
    def __init__(self):
        # Config'den ayarları al
        self.partial_sell_enabled = getattr(settings, 'MOMENTUM_PARTIAL_SELL_ENABLED', True)
        self.trailing_stop_enabled = getattr(settings, 'MOMENTUM_TRAILING_STOP_ENABLED', True)
        
        # Partial sell thresholds
        self.partial_threshold_1 = getattr(settings, 'MOMENTUM_PARTIAL_SELL_THRESHOLD_1', 5.0)  # %5
        self.partial_percentage_1 = getattr(settings, 'MOMENTUM_PARTIAL_SELL_PERCENTAGE_1', 0.3)  # %30
        self.partial_threshold_2 = getattr(settings, 'MOMENTUM_PARTIAL_SELL_THRESHOLD_2', 10.0)  # %10
        self.partial_percentage_2 = getattr(settings, 'MOMENTUM_PARTIAL_SELL_PERCENTAGE_2', 0.5)  # %50
        
        # Trailing stop
        self.trailing_activation = getattr(settings, 'MOMENTUM_TRAILING_STOP_ACTIVATION', 3.0)  # %3
        self.trailing_distance = getattr(settings, 'MOMENTUM_TRAILING_STOP_DISTANCE', 1.5)  # %1.5
        
        logger.info(f"✅ Advanced Profit Taking initialized:")
        logger.info(f"   Partial Sell: {self.partial_sell_enabled} - Thresholds: {self.partial_threshold_1}%/{self.partial_threshold_2}%")
        logger.info(f"   Trailing Stop: {self.trailing_stop_enabled} - Activation: {self.trailing_activation}%")
    
    def calculate_profit_percentage(self, position, current_price: float) -> float:
        """Pozisyonun kar/zarar yüzdesini hesapla"""
        try:
            if position.entry_price <= 0:
                return 0.0
            
            profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            return profit_pct
            
        except Exception as e:
            logger.error(f"Profit percentage calculation error: {e}")
            return 0.0
    
    def should_partial_sell(self, position, current_price: float) -> Tuple[bool, str, float]:
        """
        Partial sell gerekli mi kontrol et
        
        Returns:
            (should_sell, reason, sell_percentage)
        """
        if not self.partial_sell_enabled:
            return False, "PARTIAL_SELL_DISABLED", 0.0
        
        try:
            profit_pct = self.calculate_profit_percentage(position, current_price)
            
            # Position'da partial sell history var mı kontrol et
            partial_sells = getattr(position, 'partial_sells', [])
            
            # %10+ kar - %50 sat (sadece bir kez)
            if (profit_pct >= self.partial_threshold_2 and 
                not any(ps.get('threshold') == 'THRESHOLD_2' for ps in partial_sells)):
                
                return True, f"PARTIAL_SELL_50PCT_AT_{profit_pct:.1f}PCT", self.partial_percentage_2
            
            # %5+ kar - %30 sat (sadece bir kez)
            elif (profit_pct >= self.partial_threshold_1 and 
                  not any(ps.get('threshold') == 'THRESHOLD_1' for ps in partial_sells)):
                
                return True, f"PARTIAL_SELL_30PCT_AT_{profit_pct:.1f}PCT", self.partial_percentage_1
            
            return False, "NO_PARTIAL_SELL_NEEDED", 0.0
            
        except Exception as e:
            logger.error(f"Partial sell check error: {e}")
            return False, "PARTIAL_SELL_ERROR", 0.0
    
    def update_trailing_stop(self, position, current_price: float) -> Optional[float]:
        """
        Trailing stop fiyatını güncelle
        
        Returns:
            Yeni trailing stop price veya None
        """
        if not self.trailing_stop_enabled:
            return None
        
        try:
            profit_pct = self.calculate_profit_percentage(position, current_price)
            
            # Trailing stop aktif edilmeli mi?
            if profit_pct < self.trailing_activation:
                return None
            
            # Mevcut trailing stop var mı?
            current_trailing = getattr(position, 'trailing_stop_price', None)
            
            # Yeni trailing stop fiyatını hesapla
            trailing_distance_decimal = self.trailing_distance / 100.0
            new_trailing_stop = current_price * (1 - trailing_distance_decimal)
            
            # İlk kez trailing stop set ediliyor mu?
            if current_trailing is None:
                position.trailing_stop_price = new_trailing_stop
                logger.info(f"🎯 Trailing Stop Activated: {position.position_id} at ${new_trailing_stop:.2f} "
                           f"(Profit: {profit_pct:.1f}%)")
                return new_trailing_stop
            
            # Mevcut trailing stop'u yukarı çek (sadece fiyat yükselirse)
            if new_trailing_stop > current_trailing:
                position.trailing_stop_price = new_trailing_stop
                logger.debug(f"📈 Trailing Stop Updated: {position.position_id} ${current_trailing:.2f} → ${new_trailing_stop:.2f}")
                return new_trailing_stop
            
            return current_trailing
            
        except Exception as e:
            logger.error(f"Trailing stop update error: {e}")
            return None
    
    def should_trailing_stop_sell(self, position, current_price: float) -> Tuple[bool, str]:
        """Trailing stop tetiklendi mi kontrol et"""
        if not self.trailing_stop_enabled:
            return False, "TRAILING_DISABLED"
        
        try:
            trailing_stop = getattr(position, 'trailing_stop_price', None)
            
            if trailing_stop is None:
                return False, "NO_TRAILING_STOP"
            
            if current_price <= trailing_stop:
                profit_pct = self.calculate_profit_percentage(position, current_price)
                return True, f"TRAILING_STOP_HIT_{profit_pct:.1f}PCT"
            
            return False, "TRAILING_STOP_ACTIVE"
            
        except Exception as e:
            logger.error(f"Trailing stop check error: {e}")
            return False, "TRAILING_STOP_ERROR"
    
    def get_advanced_sell_decision(self, position, current_price: float, 
                                 position_age_minutes: float) -> Dict[str, any]:
        """
        Gelişmiş satış kararı al
        
        Returns:
            {
                'action': 'HOLD' | 'PARTIAL_SELL' | 'FULL_SELL',
                'reason': str,
                'sell_percentage': float,  # 0.0-1.0 arası
                'priority': int  # 1-10 arası (10 en yüksek öncelik)
            }
        """
        try:
            profit_pct = self.calculate_profit_percentage(position, current_price)
            
            # 1. Trailing Stop kontrolü (en yüksek öncelik)
            trailing_triggered, trailing_reason = self.should_trailing_stop_sell(position, current_price)
            if trailing_triggered:
                return {
                    'action': 'FULL_SELL',
                    'reason': trailing_reason,
                    'sell_percentage': 1.0,
                    'priority': 10
                }
            
            # 2. Partial Sell kontrolü
            should_partial, partial_reason, partial_pct = self.should_partial_sell(position, current_price)
            if should_partial:
                return {
                    'action': 'PARTIAL_SELL',
                    'reason': partial_reason,
                    'sell_percentage': partial_pct,
                    'priority': 8
                }
            
            # 3. Trailing Stop güncelle (satış değil, sadece güncelleme)
            self.update_trailing_stop(position, current_price)
            
            # 4. Extreme profit - full sell (güvenlik için)
            if profit_pct >= 25.0:  # %25+ kar
                return {
                    'action': 'FULL_SELL',
                    'reason': f'EXTREME_PROFIT_{profit_pct:.1f}PCT',
                    'sell_percentage': 1.0,
                    'priority': 9
                }
            
            # 5. Hold
            return {
                'action': 'HOLD',
                'reason': f'PROFIT_{profit_pct:.1f}PCT_HOLD',
                'sell_percentage': 0.0,
                'priority': 0
            }
            
        except Exception as e:
            logger.error(f"Advanced sell decision error: {e}")
            return {
                'action': 'HOLD',
                'reason': 'ERROR_HOLD',
                'sell_percentage': 0.0,
                'priority': 0
            }
    
    def execute_partial_sell(self, position, current_price: float, sell_percentage: float, 
                           reason: str) -> Dict[str, any]:
        """
        Partial sell işlemini gerçekleştir
        
        Bu fonksiyon portfolio.py tarafından çağrılacak
        """
        try:
            # Partial sell history'yi güncelle
            if not hasattr(position, 'partial_sells'):
                position.partial_sells = []
            
            # Satılacak miktarı hesapla
            total_quantity = abs(position.quantity_btc)
            sell_quantity = total_quantity * sell_percentage
            
            # Partial sell kaydını ekle
            partial_sell_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'price': current_price,
                'quantity': sell_quantity,
                'percentage': sell_percentage,
                'reason': reason,
                'threshold': 'THRESHOLD_2' if '50PCT' in reason else 'THRESHOLD_1'
            }
            
            position.partial_sells.append(partial_sell_record)
            
            # Position quantity'sini güncelle
            position.quantity_btc = position.quantity_btc - sell_quantity
            
            logger.info(f"💰 Partial Sell Executed: {position.position_id} - "
                       f"{sell_percentage*100:.0f}% at ${current_price:.2f} - {reason}")
            
            return {
                'success': True,
                'sell_quantity': sell_quantity,
                'remaining_quantity': abs(position.quantity_btc),
                'sell_percentage': sell_percentage,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Partial sell execution error: {e}")
            return {
                'success': False,
                'sell_quantity': 0.0,
                'remaining_quantity': abs(position.quantity_btc) if hasattr(position, 'quantity_btc') else 0.0,
                'sell_percentage': 0.0,
                'reason': f'ERROR: {str(e)}'
            }