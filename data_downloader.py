# utils/data_downloader.py
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os
from typing import List, Optional, Tuple, Any # Tuple ve Any zaten vardı veya gerekebilir
from pathlib import Path
from argparse import ArgumentParser # Komut satırı argümanları için

# config ve logger importları (settings'den varsayılanlar alınacak)
try:
    from utils.config import settings
    from utils.logger import logger
except ImportError:
    # Eğer utils modülü bir üst dizindeyse ve script doğrudan çalıştırılıyorsa:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.config import settings
    from utils.logger import logger

# API'den tek seferde çekilebilecek maksimum mum sayısı (Binance için genellikle 1000)
API_LIMIT = 1000
# API istekleri arası varsayılan bekleme süresi (saniye)
DEFAULT_REQUEST_DELAY = 0.2 # 1m gibi daha uzun periyotlar için 0.1-0.2s yeterli olabilir.
                             # "5s" gibi çok kısa periyotlarda daha dikkatli olunmalı.

def parse_date_argument(date_str: Optional[str], default_date: datetime) -> datetime:
    """Verilen string tarihi parse eder, hata durumunda veya None ise default_date döner."""
    if date_str:
        try:
            # Farklı formatları deneyebiliriz, en yaygını YYYY-MM-DD
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.replace(tzinfo=timezone.utc) # Tarihi UTC yap
        except ValueError:
            logger.warning(f"Geçersiz tarih formatı: {date_str}. Varsayılan kullanılacak.")
    return default_date

def get_binance_client() -> ccxt.binance:
    """Binance exchange istemcisini başlatır."""
    # API anahtarları config'den veya ortam değişkenlerinden okunabilir.
    exchange_config = {'enableRateLimit': True}
    if settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET:
        exchange_config['apiKey'] = settings.BINANCE_API_KEY
        exchange_config['secret'] = settings.BINANCE_API_SECRET
    return ccxt.binance(exchange_config)

def download_historical_data(
    symbol: str, 
    timeframe: str, 
    start_date: datetime, 
    end_date: datetime,
    request_delay: float
) -> List[List]:
    """Belirtilen aralık için OHLCV verilerini indirir."""
    logger.info(f"\n=== {symbol} - {timeframe} Verisi İndiriliyor ===")
    logger.info(f"Başlangıç: {start_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"Bitiş    : {end_date.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    
    exchange = get_binance_client()
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    all_candles: List[List] = []
    current_ms = start_ms
    retries = 0
    max_retries = 5 # API hataları için basit bir yeniden deneme limiti

    while current_ms < end_ms:
        try:
            logger.debug(f"Veri çekiliyor: {symbol}, {timeframe}, since={current_ms}, limit={API_LIMIT}")
            candles = exchange.fetch_ohlcv(
                symbol, 
                timeframe,
                since=current_ms,
                limit=API_LIMIT
            )
            
            if not candles: # Eğer boş liste döndüyse (o aralıkta veri yok veya sonuna gelindi)
                logger.info(f"Daha fazla veri bulunamadı veya {datetime.fromtimestamp(current_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} sonrası için veri yok.")
                break 
                
            all_candles.extend(candles)
            
            # Son çekilen mumun zaman damgasını al ve bir sonraki istek için ayarla
            last_candle_timestamp_ms = candles[-1][0]
            current_ms = last_candle_timestamp_ms + exchange.parse_timeframe(timeframe) * 1000 # Bir sonraki mumun başlangıcı
            
            # İlerleme logu
            downloaded_up_to_dt = datetime.fromtimestamp(last_candle_timestamp_ms/1000, tz=timezone.utc)
            progress_pct = min(100, (last_candle_timestamp_ms - start_ms) / (end_ms - start_ms) * 100) if (end_ms - start_ms) > 0 else 100
            logger.info(
                f"İndirilen: {len(candles):4} mum (Toplam: {len(all_candles):6,}) | "
                f"Son Mum: {downloaded_up_to_dt.strftime('%Y-%m-%d %H:%M')} | "
                f"İlerleme: {progress_pct:.1f}%"
            )
            
            retries = 0 # Başarılı istek sonrası yeniden deneme sayacını sıfırla
            time.sleep(request_delay) # API rate limitlerine saygı göster
            
        except ccxt.NetworkError as e:
            retries += 1
            logger.warning(f"Ağ Hatası: {e}. Deneme {retries}/{max_retries}. {request_delay * (2**retries)} saniye bekleniyor...")
            if retries >= max_retries:
                logger.error("Maksimum yeniden deneme sayısına ulaşıldı. İndirme durduruluyor.")
                break
            time.sleep(request_delay * (2**retries)) # Üstel bekleme
        except ccxt.ExchangeError as e: # Diğer borsa hataları
            logger.error(f"Borsa Hatası: {e}. İndirme bu segment için durduruluyor.")
            # Bazı ExchangeError'lar kalıcı olabilir, bu yüzden kırılabilir.
            # Daha detaylı hata yönetimi eklenebilir.
            break
        except Exception as e:
            logger.error(f"Veri indirilirken beklenmedik hata: {e}", exc_info=True)
            # Beklenmedik hatalarda da bir süre bekleyip devam etmeyi deneyebilir veya durdurulabilir.
            break
            
    return all_candles

def save_data_to_csv(
    candles: List[List], 
    symbol: str, 
    timeframe: str, 
    start_date_obj: datetime, 
    end_date_obj: datetime, 
    output_dir: str
) -> Optional[str]:
    """İndirilen mum verilerini CSV dosyasına kaydeder."""
    if not candles:
        logger.warning("Kaydedilecek veri bulunamadı!")
        return None
        
    df = pd.DataFrame(
        candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    # Veriyi filtreleyerek sadece istenen tarih aralığındakileri sakla (API bazen fazladan verebilir)
    df = df[(df['timestamp'] >= start_date_obj) & (df['timestamp'] <= end_date_obj)]
    df = df.drop_duplicates(subset=['timestamp']) # Olası duplicate'leri temizle
    df = df.sort_values('timestamp')

    if df.empty:
        logger.warning("Filtreleme ve duplicate temizleme sonrası kaydedilecek veri kalmadı.")
        return None

    # Dosya adını oluştur (semboldeki '/' karakterini değiştir)
    safe_symbol = symbol.replace('/', '')
    filename = f"{safe_symbol}_{timeframe}_{start_date_obj.strftime('%Y%m%d')}_{end_date_obj.strftime('%Y%m%d')}.csv"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # Çıktı dizinini oluştur
    filepath = output_path / filename
    
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"\n✅ {len(df):,} kayıt başarıyla kaydedildi → {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Veri CSV'ye kaydedilirken hata: {e}", exc_info=True)
        return None

def main():
    parser = ArgumentParser(description="Binance Tarihsel OHLCV Veri İndirici")
    parser.add_argument(
        "--symbol", 
        type=str, 
        default=settings.SYMBOL, 
        help=f"İndirilecek sembol (varsayılan: {settings.SYMBOL})"
    )
    parser.add_argument(
        "--timeframe", 
        type=str, 
        default=settings.TIMEFRAME, 
        help=f"Zaman dilimi (örn: 1m, 5m, 1h, 1d; varsayılan: {settings.TIMEFRAME})"
    )
    parser.add_argument(
        "--startdate", 
        type=str, 
        help="Başlangıç tarihi (YYYY-AA-GG formatında). Varsayılan: 30 gün öncesi."
    )
    parser.add_argument(
        "--enddate", 
        type=str, 
        help="Bitiş tarihi (YYYY-AA-GG formatında). Varsayılan: Bugün."
    )
    parser.add_argument(
        "--outputdir", 
        type=str, 
        default="historical_data", 
        help="Verilerin kaydedileceği dizin (varsayılan: historical_data)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help=f"API istekleri arası bekleme süresi (saniye, varsayılan: {DEFAULT_REQUEST_DELAY})"
    )
    args = parser.parse_args()

    # Tarih aralığını belirle
    # Bitiş tarihi, gün sonunu (23:59:59) temsil etmesi için ayarlanabilir veya olduğu gibi bırakılabilir.
    # ccxt 'since' parametresini başlangıç olarak alır, o bar dahil.
    default_end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) # Bugünün başlangıcı
    default_start_date = default_end_date - timedelta(days=30)

    start_date_obj = parse_date_argument(args.startdate, default_start_date)
    end_date_obj = parse_date_argument(args.enddate, default_end_date)

    # Bitiş tarihinin başlangıç tarihinden sonra olduğundan emin ol
    if start_date_obj >= end_date_obj:
        logger.error(f"Başlangıç tarihi ({start_date_obj}) bitiş tarihinden ({end_date_obj}) sonra veya eşit olamaz.")
        return

    # Veriyi indir
    candles_data = download_historical_data(
        args.symbol, 
        args.timeframe, 
        start_date_obj, 
        end_date_obj,
        args.delay
    )
    
    # Veriyi kaydet
    if candles_data:
        save_data_to_csv(
            candles_data, 
            args.symbol, 
            args.timeframe, 
            start_date_obj, # Dosya adı için orijinal başlangıç ve bitiş kullanılır
            end_date_obj, 
            args.outputdir
        )
    else:
        logger.warning("Hiçbir veri indirilmedi veya kaydedilecek veri yok.")

if __name__ == "__main__":
    main()