# strategies/rsi_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Optional, Tuple
# import numpy as np # Kullanılmıyor gibi, kaldırılabilir

from strategies.base import BaseStrategy
from utils.portfolio import Portfolio, Position # Position import edilmiş, iyi.
from utils.config import settings
from utils.logger import logger
from utils.ai_signal_provider import AiSignalProvider, AiSignal # AiSignal Enum importu eklendi

class RsiStrategy(BaseStrategy):
    """
    Optimize Edilmiş RSI Stratejisi:
    - ALIM: RSI aşırı satım + Trend doğrulama + Hacim teyidi + Volatilite Kontrolü + AI Sinyal Düzeltmesi
    - SATIM: RSI aşırı alım veya Kâr hedefine/Stop-loss'a ulaşma + ATR Stop
    """
    def __init__(self, portfolio: Portfolio, symbol: str = settings.SYMBOL, ai_provider: Optional[AiSignalProvider] = None):
        super().__init__(strategy_name="RSI", portfolio=portfolio, symbol=symbol, ai_provider=ai_provider)
        
        # === Strateji Parametreleri (Analiz Dokümanlarına Göre Revize Edilmiş) ===
        # Bu değerlerin settings (config.py) üzerinden gelmesi beklenir.
        # PDF (kullanıcı analizi) sayfa 17'deki önerilere göre:

        # RSI parametreleri
        self.rsi_period: int = settings.RSI_STRATEGY_RSI_PERIOD # Örn: 14 (standart)
        self.rsi_oversold_threshold: float = settings.RSI_STRATEGY_RSI_OVERSOLD
        # Öneri: 40.0 (trend filtresi olduğu için daha fazla sinyal almak amacıyla hafif yükseltildi)
        self.rsi_overbought_threshold: float = settings.RSI_STRATEGY_RSI_OVERBOUGHT
        # Öneri: 70.0 (erken satışı önlemek için yükseltildi)
        
        # Trend parametreleri
        self.short_ema_period: int = settings.RSI_STRATEGY_EMA_SHORT # Örn: 8
        self.long_ema_period: int = settings.RSI_STRATEGY_EMA_LONG   # Örn: 21
        # min_trend_strength (0.002) _check_trend içinde kullanılabilir, ancak skorlama daha esnek.
        self.trend_score_threshold: int = settings.RSI_STRATEGY_TREND_SCORE_THRESHOLD # Örn: 6 (0-10 arası skorda)
        
        # Hacim parametreleri
        self.volume_ma_period: int = settings.RSI_STRATEGY_VOLUME_MA_PERIOD # Örn: 20
        self.min_volume_factor: float = settings.RSI_STRATEGY_MIN_VOLUME_FACTOR
        # Öneri: 1.1 (hacim filtresi bir miktar gevşetildi)
        
        # Risk yönetimi
        self.profit_target_percentage: float = settings.RSI_STRATEGY_TP_PERCENTAGE
        # Öneri: 0.02 (%2 kâr hedefi, trend yönünde işlem yaptığı için risk/ödül artırılıyor)
        self.stop_loss_percentage: float = settings.RSI_STRATEGY_SL_PERCENTAGE
        # Öneri: 0.01 (%1 zarar kesme)
        
        self.atr_stop_loss_multiplier: float = settings.RSI_STRATEGY_ATR_SL_MULTIPLIER
        # Öneri (genel): ATR tabanlı stop. Örn: 2.0 (config'den)

        self.max_positions: int = settings.RSI_STRATEGY_MAX_POSITIONS # Örn: 3
        
        # Volatilite (Quantile olarak)
        self.volatility_window_for_rank: int = settings.RSI_STRATEGY_VOLATILITY_WINDOW_RANK # Örn: 100 (persentil için daha uzun pencere)
        self.max_volatility_quantile: float = settings.RSI_STRATEGY_MAX_VOLATILITY_QUANTILE
        # Öneri: 0.8 (çok yüksek volatilitede girme) veya 0.9 (daha esnek)

        self.logger.info(
            f"[{self.strategy_name}] Strateji başlatıldı. RSI Eşikleri: {self.rsi_oversold_threshold}/{self.rsi_overbought_threshold}, "
            f"Kâr Hedefi: {self.profit_target_percentage*100:.1f}%, Stop-Loss: {self.stop_loss_percentage*100:.1f}%, "
            f"Hacim Faktörü: {self.min_volume_factor}, Trend Skoru Eşiği: {self.trend_score_threshold}"
        )
        
    async def _calculate_indicators(self, df_ohlcv: pd.DataFrame) -> Optional[pd.DataFrame]:
        # DataFrame kopyası üzerinde çalışmak daha güvenli
        df_indicators = df_ohlcv.copy()
        min_required_data = max(self.rsi_period, self.long_ema_period, self.volume_ma_period, 26, 14) # MACD slow, ATR için
        
        if len(df_indicators) < min_required_data:
            self.logger.debug(f"[{self.strategy_name}] İndikatör hesaplamak için yeterli veri yok: {len(df_indicators)}/{min_required_data}")
            return None
            
        try:
            df_indicators['rsi'] = ta.rsi(df_indicators['close'], length=self.rsi_period)
            df_indicators['ema_short'] = ta.ema(df_indicators['close'], length=self.short_ema_period)
            df_indicators['ema_long'] = ta.ema(df_indicators['close'], length=self.long_ema_period)
            
            macd = ta.macd(df_indicators['close'], fast=12, slow=26, signal=9) # Standart MACD periyotları
            if macd is not None and not macd.empty:
                df_indicators['macd'] = macd.iloc[:,0]
                df_indicators['macd_signal'] = macd.iloc[:,1]
                df_indicators['macd_hist'] = macd.iloc[:,2]
            else:
                df_indicators['macd'] = df_indicators['macd_signal'] = df_indicators['macd_hist'] = pd.NA
            
            df_indicators['volume_ma'] = ta.sma(df_indicators['volume'], length=self.volume_ma_period)
            df_indicators['volume_ratio'] = df_indicators['volume'] / (df_indicators['volume_ma'].replace(0, 1e-9))
            
            df_indicators['momentum_3p'] = df_indicators['close'].pct_change(3) # 3 periyotluk momentum
            df_indicators['volatility_10p_std_pct'] = df_indicators['close'].pct_change().rolling(window=10).std() * 100
            # Volatilite persentilini daha uzun bir pencere üzerinden hesapla
            df_indicators['volatility_quantile_rank'] = df_indicators['volatility_10p_std_pct'].rolling(window=self.volatility_window_for_rank, min_periods=20).rank(pct=True)

            df_indicators['atr'] = ta.atr(df_indicators['high'], df_indicators['low'], df_indicators['close'], length=14)

            return df_indicators.iloc[-2:] # current ve previous için
            
        except Exception as e:
            self.logger.error(f"[{self.strategy_name}] İndikatör hesaplama hatası: {e}", exc_info=True)
            return None

    def _check_trend(self, current_indicators: pd.Series, previous_indicators: pd.Series) -> Tuple[bool, str]:
        """Trend analizi. Güncel ve bir önceki satırı (Series) alır."""
        trend_score = 0
        conditions_met = []

        # Gerekli sütunların varlığını ve NaN olmadığını kontrol et
        required_cols_trend = ['ema_short', 'ema_long', 'macd', 'macd_signal', 'momentum_3p', 'volatility_10p_std_pct']
        if any(pd.isna(current_indicators.get(col)) for col in required_cols_trend) or \
           any(pd.isna(previous_indicators.get(col)) for col in ['ema_short', 'ema_long']): # EMA strengthening için prev de lazım
            return False, "Trend analizi için eksik veri"

        # EMA Trendi
        if current_indicators['ema_short'] > current_indicators['ema_long']:
            trend_score += 3
            conditions_met.append("EMA Pozitif")

        # EMA Güçlenmesi
        current_ema_diff = current_indicators['ema_short'] - current_indicators['ema_long']
        previous_ema_diff = previous_indicators['ema_short'] - previous_indicators['ema_long']
        if current_ema_diff > previous_ema_diff:
            trend_score += 2
            conditions_met.append("EMA Güçleniyor")

        # MACD Trendi
        if current_indicators['macd'] > current_indicators['macd_signal']:
            trend_score += 2
            conditions_met.append("MACD Pozitif")
            
        # Momentum (Pozitif ve anlamlı bir artış)
        # min_trend_strength %0.2 idi, bu momentum için direkt kullanılabilir.
        if current_indicators['momentum_3p'] > 0.0005: # %0.05 gibi küçük bir eşik
            trend_score += 2
            conditions_met.append("Momentum Pozitif")

        # Düşük Volatilite (Piyasa sakinse trend daha tutarlı olabilir varsayımı)
        # Bu koşul, ana volatilite filtresiyle çelişebilir. "Aşırı yüksek olmayan" volatilite daha mantıklı olabilir.
        # Şimdilik orijinal mantığı koruyalım: volatilite, kendi ortalamasının altındaysa.
        # Daha robust bir "low_volatility" tanımı:
        # if current_indicators['volatility_10p_std_pct'] < current_indicators['volatility_10p_std_pct'].rolling(20).mean().iloc[-1] :
        # Yukarıdaki `volatility_ma` hesaplaması _check_trend içinde hatalıydı, indikatörlerde hesaplanmalı.
        # Ya da basitçe volatilite persentilini kullan:
        if current_indicators.get('volatility_quantile_rank', 1.0) < 0.5: # Volatilite medyanın altındaysa "düşük" kabul edilebilir
            trend_score += 1
            conditions_met.append("Volatilite Düşük/Orta")
            
        trend_strength_msg = "Güçlü" if trend_score >= 8 else "Orta" if trend_score >= self.trend_score_threshold else "Zayıf"
        final_msg = f"{trend_strength_msg} trend (Skor: {trend_score}/10, Koşullar: {', '.join(conditions_met) if conditions_met else 'Yok'})"
        
        return trend_score >= self.trend_score_threshold, final_msg

    async def should_buy(self, df_ohlcv: pd.DataFrame) -> bool:
        # 0. Genel Alım Koşulları ve Filtreler
        if not await self.validate_trade_conditions(df_ohlcv, strategy_name=self.strategy_name):
            return False

        indicators = await self._calculate_indicators(df_ohlcv)
        if indicators is None or len(indicators) < 2: # current ve previous için
            return False
            
        current = indicators.iloc[-1]
        previous = indicators.iloc[-2] # _check_trend için
        
        # Gerekli indikatörlerin varlığını ve NaN olmadığını kontrol et
        required_cols_buy = ['rsi', 'volume_ratio', 'volatility_quantile_rank']
        if any(pd.isna(current.get(col)) for col in required_cols_buy):
            # self.logger.debug(f"[{self.strategy_name}] Alım için gerekli indikatörlerden bazıları NaN.")
            return False

        # Pozisyon limiti kontrolü
        open_positions = self.portfolio.get_open_positions(self.symbol, self.strategy_name)
        if len(open_positions) >= self.max_positions:
            return False
            
        # 1. RSI Aşırı Satım Kontrolü (Revize Edilmiş Eşik)
        rsi_is_oversold = current['rsi'] <= self.rsi_oversold_threshold
        if not rsi_is_oversold:
            return False # Temel alım koşulu sağlanmadı
            
        # 2. Trend Kontrolü (Revize Edilmiş Mantık)
        trend_is_supportive, trend_msg = self._check_trend(current, previous)
        if not trend_is_supportive:
            # self.logger.debug(f"[{self.strategy_name}] Trend desteklemiyor: {trend_msg}")
            return False
            
        # 3. Hacim Kontrolü (Revize Edilmiş Faktör)
        volume_is_ok = current['volume_ratio'] >= self.min_volume_factor
        if not volume_is_ok:
            # self.logger.debug(f"[{self.strategy_name}] Hacim yetersiz: {current['volume_ratio']:.2f}x < {self.min_volume_factor}x")
            return False
            
        # 4. Volatilite Kontrolü (Aşırı Yüksek Olmasın)
        if current.get('volatility_quantile_rank', 1.0) > self.max_volatility_quantile:
            # self.logger.debug(f"[{self.strategy_name}] Volatilite çok yüksek: {current['volatility_quantile_rank']:.2f} > {self.max_volatility_quantile}")
            return False
            
        # 5. AI Sinyal Kontrolü (KRİTİK HATA DÜZELTMESİ)
        if self.use_ai_assistance and self.ai_provider:
            # self.get_ai_signal(df) yerine ai_provider'dan sinyal alacağız.
            # PDF önerisi: standalone sinyal alıp, eğer SELL/STRONG_SELL ise alımı engelle.
            ai_standalone_signal = await self.ai_provider.get_standalone_signal(df_ohlcv=indicators) # indikatörleri yolla
            if ai_standalone_signal in [AiSignal.SELL, AiSignal.STRONG_SELL]:
                self.logger.info(f"[{self.strategy_name}] ALIM sinyali AI tarafından engellendi (AI Sinyali: {ai_standalone_signal.name}).")
                return False
            # Alternatif olarak get_ai_confirmation da kullanılabilir:
            # ai_confirmation = await self.ai_provider.get_ai_confirmation("BUY", indicators, context={...})
            # if not ai_confirmation: return False

        self.logger.info(
            f"[{self.strategy_name}] 🔵 ALIM Sinyali:\n"
            f"- Fiyat: {current['close']:.8f}, RSI: {current['rsi']:.2f} (Eşik: <={self.rsi_oversold_threshold})\n"
            f"- {trend_msg}\n"
            f"- Hacim Oranı: {current['volume_ratio']:.2f}x (Eşik: >={self.min_volume_factor})\n"
            f"- Volatilite Quantile: {current.get('volatility_quantile_rank', pd.NA):.2f} (Eşik: <={self.max_volatility_quantile})"
        )
        return True

    async def should_sell(self, df_ohlcv: pd.DataFrame, position: Position) -> bool:
        indicators = await self._calculate_indicators(df_ohlcv)
        if indicators is None or len(indicators) < 2: # current ve previous için (trend kontrolü)
            return False
            
        current = indicators.iloc[-1]
        previous = indicators.iloc[-2] # _check_trend için
        current_price = current['close']

        if pd.isna(current_price) or any(pd.isna(current.get(col)) for col in ['rsi', 'atr']):
            # self.logger.debug(f"[{self.strategy_name}] Satış için gerekli güncel indikatörlerden bazıları NaN.")
            return False
        
        gross_profit_percentage = (current_price / position.entry_price - 1)
        sell_reason = None

        # 1. Sabit Stop-Loss (Revize Edilmiş Yüzde)
        if gross_profit_percentage <= -self.stop_loss_percentage:
            sell_reason = f"Sabit Stop-Loss (-{self.stop_loss_percentage*100:.2f}%)"
        
        # 2. ATR Tabanlı Dinamik Stop-Loss
        if not sell_reason and 'atr' in current and not pd.isna(current['atr']):
            atr_stop_price = position.entry_price - (current['atr'] * self.atr_stop_loss_multiplier)
            if current_price <= atr_stop_price:
                sell_reason = f"ATR Stop-Loss (Fiyat: {current_price:.8f} <= Hedef: {atr_stop_price:.8f})"

        # 3. Kâr Realizasyonu (Revize Edilmiş Yüzde)
        if not sell_reason and gross_profit_percentage >= self.profit_target_percentage:
            # Kâr hedefine ulaşıldığında trende bakmadan çıkmak daha basit olabilir.
            # Orijinal kodda trend mesajı loglanıyordu.
            sell_reason = f"Brüt Kâr Hedefi (+{self.profit_target_percentage*100:.2f}%)"

        # 4. RSI Aşırı Alım + Trend Zayıflama
        if not sell_reason and current['rsi'] >= self.rsi_overbought_threshold:
            trend_is_still_strong, trend_msg = self._check_trend(current, previous)
            if not trend_is_still_strong: # Trend artık güçlü değilse (skor < threshold)
                sell_reason = f"RSI Aşırı Alım + Trend Zayıflaması (RSI: {current['rsi']:.2f}, {trend_msg})"
        
        if sell_reason:
            log_message = (
                f"[{self.strategy_name}] 🔴 SATIŞ: {sell_reason}\n"
                f"- Giriş: {position.entry_price:.8f}, Çıkış: {current_price:.8f}\n"
                f"- Brüt Kâr/Zarar: {gross_profit_percentage*100:.4f}%"
            )
            if "Stop-Loss" in sell_reason or gross_profit_percentage < 0:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            return True
            
        return False