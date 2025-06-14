# utils/logger.py
import logging
import logging.handlers
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

try:
    from utils.config import settings
except ImportError:
    class FallbackSettings:
        LOG_LEVEL = 'INFO'
        LOG_TO_FILE = True
        TRADES_CSV_LOG_PATH = 'logs/trades.csv'
        SYMBOL = 'BTC/USDT'
        INITIAL_CAPITAL_USDT = 1000.0
    settings = FallbackSettings()

class ColoredFormatter(logging.Formatter):
    """Konsol çıktısını renklendiren formatlayıcı."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        original_levelname = record.levelname
        log_color = self.COLORS.get(original_levelname, '')
        record.levelname = f"{log_color}{original_levelname:<8}{self.RESET}" # Hizalama için boşluk eklendi
        formatted_message = super().format(record)
        record.levelname = original_levelname # Diğer handler'lar için orijinal hali geri yükle
        return formatted_message

class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Hata durumunda programı durdurmayan güvenli dosya loglayıcı."""
    def emit(self, record):
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)

class TradingLogger:
    """Sadeleştirilmiş, iki seviyeli (genel ve hata) loglama sistemi."""
    def __init__(self):
        self.log_dir = Path("logs")
        self.setup_directories()
        self.setup_loggers()
        
    def setup_directories(self):
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[FATAL] Failed to create log directory '{self.log_dir}': {e}", file=sys.stderr)
            self.log_dir = Path(".")
        
    def setup_loggers(self):
        try:
            self.logger = logging.getLogger("algobot")
            self.logger.setLevel(logging.DEBUG) # En düşük seviyeyi ayarla, handler'lar filtrelesin
            self.logger.handlers.clear()
            self.logger.propagate = False

            # 1. Console Handler (Sadece INFO ve üzeri)
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter('%(asctime)s [%(name)s] %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
            self.logger.addHandler(console_handler)
            
            if settings.LOG_TO_FILE:
                # 2. Ana Log Dosyası (algobot.log - DEBUG ve üzeri)
                algobot_log_path = self.log_dir / "algobot.log"
                file_handler = SafeRotatingFileHandler(
                    algobot_log_path, maxBytes=20*1024*1024, backupCount=5, encoding='utf-8'
                )
                file_formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(logging.DEBUG)
                self.logger.addHandler(file_handler)

                # 3. Hata Log Dosyası (errors.log - Sadece ERROR ve üzeri)
                errors_log_path = self.log_dir / "errors.log"
                error_handler = SafeRotatingFileHandler(
                    errors_log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
                )
                error_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s\nFile: %(pathname)s:%(lineno)d\nFunction: %(funcName)s\n%(exc_text)s\n' + '-'*80 + '\n', datefmt='%Y-%m-%d %H:%M:%S')
                error_handler.setFormatter(error_formatter)
                error_handler.setLevel(logging.ERROR)
                self.logger.addHandler(error_handler)
                
                print(f"INFO: Logging to files is enabled. Main log: '{algobot_log_path}', Errors: '{errors_log_path}'")

        except Exception as e:
            print(f"[FATAL] Logger setup failed: {e}", file=sys.stderr)
            self.logger = logging.getLogger("algobot_fallback")
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler(sys.stdout))
            self.logger.setLevel(logging.INFO)

# --- Global Logger Instance ---
try:
    logger = TradingLogger().logger
except Exception as e:
    # Acil durum logger'ı
    print(f"CRITICAL: Could not initialize TradingLogger. Using emergency fallback. Error: {e}")
    logger = logging.getLogger("emergency_logger")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)


# --- CSV Header Yardımcı Fonksiyonu ---
def ensure_csv_header(csv_path: str):
    """Analiz için sadeleştirilmiş CSV başlığını oluşturur veya doğrular."""
    try:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists() and path.stat().st_size > 0:
            return 
            
        header_lines = [
            f"# AlgoBot Trade Log - Generated on {datetime.now(timezone.utc).isoformat()}",
            f"# Initial Capital (for context): ${getattr(settings, 'INITIAL_CAPITAL_USDT', 1000):.2f} USDT",
            f"# Symbol (default): {getattr(settings, 'SYMBOL', 'BTC/USDT')}",
            "# pnl_usdt_trade and hold_duration_min are populated for SELL actions.",
        ]
        
        csv_header = (
            "timestamp_utc,"
            "position_id,"
            "strategy_name,"
            "symbol,"
            "action_type,"
            "price,"
            "quantity_asset,"
            "gross_value_usdt,"
            "fee_usdt,"
            "net_value_usdt,"
            "reason_detailed,"
            "hold_duration_min,"
            "pnl_usdt_trade,"
            "cumulative_pnl_usdt"
        )
        header_lines.append(csv_header)
        
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write('\n'.join(header_lines) + '\n')
            
        logger.info(f"Trade log CSV file header created: {path.resolve()}")
    except Exception as e:
        logger.error(f"CSV header creation error: {e}", exc_info=True)


# --- Sistem Durumu Loglama Fonksiyonları ---
def log_system_startup(config_summary: Dict[str, Any]):
    logger.info("=" * 60)
    logger.info("🚀 ALGO BOT SYSTEM STARTUP")
    logger.info("=" * 60)
    for key, value in config_summary.items():
        logger.info(f"  {key:<25}: {value}")
    logger.info("-" * 60)

def log_system_shutdown(final_stats: Dict[str, Any]):
    logger.info("=" * 60)
    logger.info("🏁 ALGO BOT SYSTEM SHUTDOWN")
    logger.info("=" * 60)
    if final_stats:
        for key, value in final_stats.items():
            logger.info(f"  {key:<25}: {value}")
    else:
        logger.info("  No final statistics available.")
    logger.info("=" * 60)