# utils/kelly_criterion.py - Yeni dosya oluştur

"""
🎯 Kelly Criterion Position Sizing Module
Optimal position sizing için Kelly formülü kullanır
"""

import numpy as np
from typing import List, Dict, Optional
from utils.logger import logger

class KellyCriterion:
    """Kelly Criterion ile optimal position sizing"""
    
    def __init__(self, lookback_trades: int = 30, max_kelly_fraction: float = 0.4):
        self.lookback_trades = lookback_trades
        self.max_kelly_fraction = max_kelly_fraction
        self.min_kelly_fraction = 0.05
        
    def calculate_kelly_fraction(self, trade_history: List[Dict]) -> float:
        """
        Kelly Fraction hesapla: f* = (bp - q) / b
        Burada:
        - b = kazanç oranı (ortalama kazanç / ortalama kayıp)
        - p = kazanma olasılığı
        - q = kaybetme olasılığı (1-p)
        """
        if len(trade_history) < 10:
            logger.debug("Kelly: Yetersiz trade history, varsayılan %10 kullanılıyor")
            return 0.1
        
        try:
            # Son N trade'i al
            recent_trades = trade_history[-self.lookback_trades:]
            
            # Kar/zarar ayır
            winning_trades = [t for t in recent_trades if t.get('pnl_pct', 0) > 0]
            losing_trades = [t for t in recent_trades if t.get('pnl_pct', 0) <= 0]
            
            if not winning_trades or not losing_trades:
                logger.debug("Kelly: Tüm kar veya tüm zarar, muhafazakar %8 kullanılıyor")
                return 0.08
            
            # Win rate hesapla
            win_rate = len(winning_trades) / len(recent_trades)
            
            # Ortalama kazanç/kayıp yüzdesi
            avg_win_pct = np.mean([abs(t['pnl_pct']) for t in winning_trades])
            avg_loss_pct = np.mean([abs(t['pnl_pct']) for t in losing_trades])
            
            if avg_loss_pct == 0:
                logger.warning("Kelly: Ortalama kayıp 0, muhafazakar %8 kullanılıyor")
                return 0.08
            
            # Kelly formülü
            b = avg_win_pct / avg_loss_pct  # Kazanç oranı
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Güvenlik sınırları
            kelly_fraction = max(self.min_kelly_fraction, 
                               min(self.max_kelly_fraction, kelly_fraction))
            
            logger.debug(f"Kelly Hesaplama: Win Rate={p:.3f}, Avg Win={avg_win_pct:.3f}%, "
                        f"Avg Loss={avg_loss_pct:.3f}%, B={b:.3f}, Kelly={kelly_fraction:.3f}")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Kelly hesaplama hatası: {e}")
            return 0.1  # Güvenli varsayılan
    
    def adjust_for_volatility(self, kelly_fraction: float, current_volatility: float) -> float:
        """Volatiliteye göre Kelly fraction'ı ayarla"""
        try:
            # Yüksek volatilite = daha düşük pozisyon
            # Düşük volatilite = daha yüksek pozisyon
            
            if current_volatility > 0.03:  # %3+ volatilite
                volatility_multiplier = 0.7
            elif current_volatility > 0.02:  # %2-3 volatilite
                volatility_multiplier = 0.85
            elif current_volatility < 0.01:  # %1 altı volatilite
                volatility_multiplier = 1.2
            else:  # Normal volatilite
                volatility_multiplier = 1.0
            
            adjusted_kelly = kelly_fraction * volatility_multiplier
            
            # Sınırları tekrar kontrol et
            adjusted_kelly = max(self.min_kelly_fraction, 
                               min(self.max_kelly_fraction, adjusted_kelly))
            
            logger.debug(f"Kelly Volatilite Ayarı: Vol={current_volatility:.4f}, "
                        f"Multiplier={volatility_multiplier:.2f}, "
                        f"Adjusted Kelly={adjusted_kelly:.3f}")
            
            return adjusted_kelly
            
        except Exception as e:
            logger.error(f"Kelly volatilite ayarı hatası: {e}")
            return kelly_fraction
    
    def calculate_position_size(self, 
                              available_capital: float, 
                              trade_history: List[Dict],
                              current_volatility: float = 0.02,
                              kelly_multiplier: float = 1.0) -> float:
        """
        Optimal position size hesapla
        
        Args:
            available_capital: Kullanılabilir sermaye
            trade_history: Geçmiş trade listesi
            current_volatility: Güncel piyasa volatilitesi
            kelly_multiplier: Kelly fraction çarpanı (0.5-2.0 arası)
        
        Returns:
            Optimal position size (USDT)
        """
        try:
            # Kelly fraction hesapla
            kelly_fraction = self.calculate_kelly_fraction(trade_history)
            
            # Volatilite ayarı
            kelly_fraction = self.adjust_for_volatility(kelly_fraction, current_volatility)
            
            # Multiplier uygula
            final_kelly = kelly_fraction * kelly_multiplier
            final_kelly = max(self.min_kelly_fraction, 
                            min(self.max_kelly_fraction, final_kelly))
            
            # Position size hesapla
            position_size = available_capital * final_kelly
            
            logger.debug(f"Kelly Position Size: Capital=${available_capital:.2f}, "
                        f"Kelly={final_kelly:.3f}, Size=${position_size:.2f}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Kelly position size hatası: {e}")
            # Hata durumunda güvenli varsayılan
            return available_capital * 0.1

# Kullanım örneği için test fonksiyonu
def test_kelly_criterion():
    """Kelly Criterion test fonksiyonu"""
    kelly = KellyCriterion()
    
    # Örnek trade history
    sample_trades = [
        {'pnl_pct': 2.5},   # %2.5 kar
        {'pnl_pct': -1.2},  # %1.2 zarar
        {'pnl_pct': 3.1},   # %3.1 kar
        {'pnl_pct': -1.8},  # %1.8 zarar
        {'pnl_pct': 1.9},   # %1.9 kar
        {'pnl_pct': -0.8},  # %0.8 zarar
        {'pnl_pct': 4.2},   # %4.2 kar
        {'pnl_pct': -2.1},  # %2.1 zarar
        {'pnl_pct': 2.8},   # %2.8 kar
        {'pnl_pct': -1.5},  # %1.5 zarar
    ]
    
    position_size = kelly.calculate_position_size(
        available_capital=1000.0,
        trade_history=sample_trades,
        current_volatility=0.025,
        kelly_multiplier=0.8
    )
    
    print(f"Test Kelly Position Size: ${position_size:.2f}")

if __name__ == "__main__":
    test_kelly_criterion()