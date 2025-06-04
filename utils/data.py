# utils/data.py
import asyncio
from typing import List, Any, Callable, Coroutine, Tuple, Optional, Dict # Dict eklendi

import ccxt.async_support as ccxt
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

from utils.config import settings
from utils.logger import logger

# ccxt.NetworkError, ccxt.ExchangeError gibi genel hatalar için
COMMON_CCXT_ERRORS = (
    ccxt.NetworkError,
    ccxt.ExchangeError,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
    ccxt.ExchangeNotAvailable,
    ccxt.OnMaintenance,
    ccxt.InvalidNonce, # Eklenebilecek diğer yaygın hatalar
    ccxt.AuthenticationError # AuthenticationError da yeniden denenebilir (geçici sorunlar için) ama dikkatli olunmalı
)

class BotError(Exception):
    """Bot operasyonları sırasında oluşan özel hatalar için temel sınıf."""
    pass

class DataFetchingError(BotError):
    """Veri çekme sırasında oluşan hatalar için."""
    pass

class BinanceFetcher:
    """
    Binance'ten asenkron olarak OHLCV ve gelecekte LOB verilerini çekmek için sınıf.
    Yeniden bağlanma ve hata yönetimi içerir.
    """
    def __init__(self, symbol: str = settings.SYMBOL, timeframe: str = settings.TIMEFRAME):
        self.symbol: str = symbol
        self.timeframe: str = timeframe # Stratejilerin kullandığı ana zaman dilimi
        self.ohlcv_limit: int = settings.OHLCV_LIMIT # Tek seferde çekilecek mum sayısı
        
        # API anahtarları config'den veya ortam değişkenlerinden okunabilir.
        # ccxt, anahtarlar ortam değişkenlerinde (BINANCE_API_KEY, BINANCE_SECRET_KEY) varsa otomatik kullanabilir.
        exchange_config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        }
        if settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET: # config.py'de bu değişkenler olmalı
            exchange_config['apiKey'] = settings.BINANCE_API_KEY
            exchange_config['secret'] = settings.BINANCE_API_SECRET
            logger.info("Binance API anahtarları yapılandırmadan yüklendi.")
        else:
            logger.warning("Binance API anahtarları yapılandırmada bulunamadı. Borsa limitli erişim modunda olabilir.")

        self.exchange: ccxt.binance = ccxt.binance(exchange_config)
        logger.info(f"BinanceFetcher {self.symbol} sembolü ve {self.timeframe} zaman dilimi için başlatıldı. OHLCV Limiti: {self.ohlcv_limit}")

    @retry(
        stop=stop_after_attempt(settings.DATA_FETCHER_RETRY_ATTEMPTS), # Maksimum deneme sayısı (config'den)
        wait=wait_exponential(
            multiplier=settings.DATA_FETCHER_RETRY_MULTIPLIER, # Çarpan (config'den)
            min=settings.DATA_FETCHER_RETRY_MIN_WAIT,         # Min bekleme (saniye, config'den)
            max=settings.DATA_FETCHER_RETRY_MAX_WAIT          # Maks bekleme (saniye, config'den)
        ),
        retry=retry_if_exception_type(COMMON_CCXT_ERRORS),
        reraise=True # Son denemeden sonra hata devam ederse tekrar yükselt
    )
    async def fetch_ohlcv_data(self, limit_override: Optional[int] = None, since_timestamp_ms: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Binance'ten OHLCV verilerini çeker ve pandas DataFrame olarak döndürür.
        Hata durumunda tenacity ile yeniden dener.

        Args:
            limit_override (Optional[int]): Çekilecek mum sayısını override eder.
            since_timestamp_ms (Optional[int]): Hangi zamandan itibaren veri çekileceğini belirtir (milisaniye Unix timestamp).

        Returns:
            Optional[pd.DataFrame]: OHLCV verilerini içeren DataFrame veya hata/başarısızlık durumunda None.
                                    Eğer reraise=True ise, son denemeden sonra hata yükseltilir.
        """
        current_limit = limit_override if limit_override is not None else self.ohlcv_limit
        try:
            logger.debug(
                f"OHLCV verisi çekiliyor: Sembol={self.symbol}, Zaman Dilimi={self.timeframe}, "
                f"Limit={current_limit}, Since={since_timestamp_ms if since_timestamp_ms else 'Yok'}"
            )
            
            ohlcv = await self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframe, 
                since=since_timestamp_ms, 
                limit=current_limit
            )
            
            if not ohlcv: # Boş liste geldiyse
                logger.warning(f"{self.symbol} için OHLCV verisi dönmedi (muhtemelen o aralıkta veri yok veya limit çok düşük).")
                return pd.DataFrame() # Boş DataFrame döndür, None yerine

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # UTC olarak ayarla
            df.set_index('timestamp', inplace=True)
            
            # Veri türlerini de kontrol edip dönüştürmek iyi bir pratik
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            
            logger.debug(f"{self.symbol} için {len(df)} adet mum başarıyla çekildi, son mum: {df.index[-1] if not df.empty else 'N/A'}")
            return df
            
        except ccxt.AuthenticationError as e:
            logger.critical(f"CCXT Kimlik Doğrulama Hatası: {e}. API anahtarlarını kontrol edin!", exc_info=True)
            # Bu hatada yeniden denemek anlamsız olabilir, bu yüzden COMMON_CCXT_ERRORS'dan çıkarılabilir
            # veya tenacity'e özel bir `retry_error_callback` ile bu hata loglanıp bot durdurulabilir.
            # Şimdilik COMMON_CCXT_ERRORS içinde kalırsa tenacity reraise edecek.
            raise DataFetchingError(f"Kimlik doğrulama başarısız: {e}") from e
        except COMMON_CCXT_ERRORS as e: # Tenacity bu hataları yakalayıp yeniden deneyecek
            logger.warning(f"{self.symbol} için OHLCV çekilirken CCXT Hatası ({type(e).__name__}): {e}. Yeniden deneniyor...")
            raise # Tenacity'nin yeniden denemesi için hatayı tekrar yükselt
        except Exception as e:
            logger.error(f"{self.symbol} için OHLCV çekilirken beklenmedik hata: {e}", exc_info=True)
            raise DataFetchingError(f"OHLCV çekilirken beklenmedik hata: {e}") from e

    async def fetch_order_book_data(self, depth: int = 20) -> Optional[Dict[str, Any]]:
        """
        Binance'ten Limit Emir Defteri (LOB) verilerini çeker.
        Bu fonksiyon, Gelişmiş ML Modelleri için (kaynak 1, bölüm 2.1, 3.1) gereklidir.
        Şu an için bir placeholder olup, gerçek implementasyon gerektirir.
        """
        logger.info(f"fetch_order_book_data çağrıldı (Sembol: {self.symbol}, Derinlik: {depth}). Bu özellik henüz tam olarak implemente edilmedi.")
        # Örnek ccxt çağrısı (yeniden deneme mekanizması eklenmeli):
        # try:
        #     order_book = await self.exchange.fetch_order_book(self.symbol, limit=depth)
        #     # Gelen order_book verisini (bid'ler, ask'ler) işleyip döndür
        #     # Örn: return {"bids": order_book['bids'], "asks": order_book['asks'], "timestamp": order_book['timestamp']}
        # except Exception as e:
        #     logger.error(f"{self.symbol} için Emir Defteri verisi çekilemedi: {e}")
        #     return None
        raise NotImplementedError("fetch_order_book_data henüz tam olarak implemente edilmedi.")

    async def fetch_ticker_data(self) -> Optional[Dict[str, Any]]:
        """Sembol için güncel ticker (fiyat, spread vb.) bilgilerini çeker."""
        try:
            logger.debug(f"Ticker verisi çekiliyor: Sembol={self.symbol}")
            ticker = await self.exchange.fetch_ticker(self.symbol)
            if not ticker:
                logger.warning(f"{self.symbol} için Ticker verisi dönmedi.")
                return None
            logger.debug(f"{self.symbol} için Ticker verisi: Son Fiyat={ticker.get('last')}")
            return ticker
        except COMMON_CCXT_ERRORS as e:
            logger.warning(f"{self.symbol} için Ticker çekilirken CCXT Hatası ({type(e).__name__}): {e}. Yeniden denenebilir (eğer retry ile sarılırsa).")
            raise DataFetchingError(f"Ticker çekilirken CCXT Hatası: {e}") from e
        except Exception as e:
            logger.error(f"{self.symbol} için Ticker çekilirken beklenmedik hata: {e}", exc_info=True)
            raise DataFetchingError(f"Ticker çekilirken beklenmedik hata: {e}") from e

    async def close_connection(self) -> None:
        """Exchange bağlantısını düzgün bir şekilde kapatır."""
        try:
            logger.info(f"Binance ({self.symbol}) exchange bağlantısı kapatılıyor.")
            await self.exchange.close()
        except Exception as e:
            logger.error(f"Exchange bağlantısı kapatılırken hata oluştu: {e}", exc_info=True)

if __name__ == "__main__":
    async def main_test_fetch():
        # Test için config ayarlarının doğru olduğundan emin olun
        # settings.SYMBOL = "BTC/USDT"
        # settings.TIMEFRAME = "5s" # Buranın "5s" olduğundan emin olun
        # settings.OHLCV_LIMIT = 100
        # settings.DATA_FETCHER_RETRY_ATTEMPTS = 3 
        # ... diğer fetcher retry ayarları ...

        fetcher = BinanceFetcher(symbol=settings.SYMBOL, timeframe=settings.TIMEFRAME)
        ohlcv_df = None
        try:
            ohlcv_df = await fetcher.fetch_ohlcv_data() # Son N mumu çek

            if ohlcv_df is not None and not ohlcv_df.empty:
                print("\n--- OHLCV Verisi ---")
                print(ohlcv_df.tail())
                print(f"\nVeri Tipleri:\n{ohlcv_df.dtypes}")
                print(f"\nSon Mum Kapanış Fiyatı: {ohlcv_df['close'].iloc[-1]}")
                print(f"Veri Aralığı: {ohlcv_df.index.min()} -> {ohlcv_df.index.max()}")
            elif ohlcv_df is not None and ohlcv_df.empty:
                print("OHLCV verisi çekildi ancak boş döndü (muhtemelen belirtilen aralıkta veri yok).")
            else: 
                print("OHLCV verisi çekilemedi (fetch_ohlcv_data None döndürdü).")

            # Ticker verisi çekme testi
            # ticker_data = await fetcher.fetch_ticker_data() # Bu satırı test için açabilirsiniz
            # if ticker_data:
            #     print("\n--- Ticker Verisi ---")
            #     print(f"Sembol: {ticker_data.get('symbol')}, Son Fiyat: {ticker_data.get('last')}")
            #     print(f"Bid: {ticker_data.get('bid')}, Ask: {ticker_data.get('ask')}")

        except DataFetchingError as e: 
            print(f"Veri çekme işlemi kalıcı olarak başarısız oldu: {e}")
        except Exception as e:
            print(f"Test sırasında beklenmedik bir hata oluştu: {e}")
        finally:
            await fetcher.close_connection()

    asyncio.run(main_test_fetch())