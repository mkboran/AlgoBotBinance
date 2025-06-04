# utils/logger.py - Professional Multi-Level Logging System
import logging
import logging.handlers
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union

# Config import'u try-catch ile koruyalım
try:
    from utils.config import settings
except ImportError:
    # Fallback settings if config not available
    class FallbackSettings:
        LOG_LEVEL = 'INFO'
        LOG_TO_FILE = True
        SYMBOL = 'BTC/USDT'
        TIMEFRAME = '5m'
        INITIAL_CAPITAL_USDT = 1000
        FEE_BUY = 0.001
        FEE_SELL = 0.001
        TRADES_CSV_LOG_PATH = 'logs/trades.csv'
        
    settings = FallbackSettings()

class ColoredFormatter(logging.Formatter):
    """Colored console output formatter"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m'   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        try:
            log_color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{log_color}{record.levelname}{self.RESET}"
            return super().format(record)
        except Exception:
            # Fallback formatting
            return f"{record.levelname}: {record.getMessage()}"

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
                
            if hasattr(record, 'extra_data'):
                log_entry['extra'] = record.extra_data
                
            return json.dumps(log_entry, ensure_ascii=False)
        except Exception as e:
            # Fallback JSON
            return json.dumps({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': 'ERROR',
                'message': f"Logging error: {str(e)}",
                'original_message': str(record.getMessage()) if hasattr(record, 'getMessage') else 'N/A'
            })

class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Safe rotating file handler with error handling"""
    
    def emit(self, record):
        try:
            super().emit(record)
        except Exception as e:
            # Silent fallback - don't break application
            try:
                print(f"Logging error: {e}", file=sys.stderr)
            except:
                pass  # Ultimate fallback

class TradingLogger:
    """Professional Trading Bot Logger with multiple outputs"""
    
    def __init__(self):
        self.setup_directories()
        self.setup_loggers()
        
    def setup_directories(self):
        """Setup log directories with error handling"""
        try:
            self.log_dir = Path("logs")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.log_dir / "trades").mkdir(exist_ok=True)
            (self.log_dir / "system").mkdir(exist_ok=True)
            (self.log_dir / "performance").mkdir(exist_ok=True)
            (self.log_dir / "errors").mkdir(exist_ok=True)
        except Exception as e:
            print(f"Failed to create log directories: {e}", file=sys.stderr)
            # Use current directory as fallback
            self.log_dir = Path(".")
        
    def setup_loggers(self):
        """Setup multiple specialized loggers with error handling"""
        
        try:
            # Main application logger
            self.logger = logging.getLogger("algobot")
            self.logger.setLevel(getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO').upper()))
            
            # Clear existing handlers
            self.logger.handlers.clear()
            
            # Console handler with colors (always works)
            try:
                console_handler = logging.StreamHandler(sys.stdout)
                console_formatter = ColoredFormatter(
                    '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                console_handler.setLevel(logging.INFO)
                self.logger.addHandler(console_handler)
            except Exception as e:
                print(f"Console handler setup failed: {e}", file=sys.stderr)
            
            # File handlers (with error handling)
            if getattr(settings, 'LOG_TO_FILE', True):
                try:
                    file_handler = SafeRotatingFileHandler(
                        self.log_dir / "algobot.log",
                        maxBytes=50*1024*1024,  # 50MB
                        backupCount=10,
                        encoding='utf-8'
                    )
                    file_formatter = logging.Formatter(
                        '%(asctime)s [%(name)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    )
                    file_handler.setFormatter(file_formatter)
                    file_handler.setLevel(logging.DEBUG)
                    self.logger.addHandler(file_handler)
                except Exception as e:
                    print(f"File handler setup failed: {e}", file=sys.stderr)
            
            # JSON structured log handler (optional)
            try:
                json_handler = SafeRotatingFileHandler(
                    self.log_dir / "system" / "structured.jsonl",
                    maxBytes=100*1024*1024,  # 100MB
                    backupCount=5,
                    encoding='utf-8'
                )
                json_handler.setFormatter(JsonFormatter())
                json_handler.setLevel(logging.INFO)
                self.logger.addHandler(json_handler)
            except Exception as e:
                print(f"JSON handler setup failed: {e}", file=sys.stderr)
            
            # Error-only handler (optional)
            try:
                error_handler = SafeRotatingFileHandler(
                    self.log_dir / "errors" / "errors.log",
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=10,
                    encoding='utf-8'
                )
                error_formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(message)s\n'
                    'File: %(pathname)s:%(lineno)d\n'
                    'Function: %(funcName)s\n'
                    '%(exc_text)s\n' + '-'*80 + '\n',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                error_handler.setFormatter(error_formatter)
                error_handler.setLevel(logging.ERROR)
                self.logger.addHandler(error_handler)
            except Exception as e:
                print(f"Error handler setup failed: {e}", file=sys.stderr)
            
            # Performance logger (optional)
            try:
                self.perf_logger = logging.getLogger("algobot.performance")
                perf_handler = SafeRotatingFileHandler(
                    self.log_dir / "performance" / "performance.log",
                    maxBytes=20*1024*1024,  # 20MB
                    backupCount=5,
                    encoding='utf-8'
                )
                perf_formatter = logging.Formatter(
                    '%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                perf_handler.setFormatter(perf_formatter)
                self.perf_logger.addHandler(perf_handler)
                self.perf_logger.setLevel(logging.INFO)
            except Exception as e:
                print(f"Performance logger setup failed: {e}", file=sys.stderr)
                self.perf_logger = self.logger  # Fallback
            
            # Trade logger (optional)
            try:
                self.trade_logger = logging.getLogger("algobot.trades")
                trade_handler = SafeRotatingFileHandler(
                    self.log_dir / "trades" / "trade_details.log",
                    maxBytes=50*1024*1024,  # 50MB
                    backupCount=20,
                    encoding='utf-8'
                )
                trade_formatter = logging.Formatter(
                    '%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                trade_handler.setFormatter(trade_formatter)
                self.trade_logger.addHandler(trade_handler)
                self.trade_logger.setLevel(logging.INFO)
            except Exception as e:
                print(f"Trade logger setup failed: {e}", file=sys.stderr)
                self.trade_logger = self.logger  # Fallback
                
        except Exception as e:
            print(f"Logger setup failed: {e}", file=sys.stderr)
            # Create minimal fallback logger
            self.logger = logging.getLogger("algobot")
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
            self.perf_logger = self.logger
            self.trade_logger = self.logger
        
    def log_trade_execution(self, action: str, symbol: str, amount: float, price: float, 
                          reason: str, context: Dict[str, Any]):
        """Log detailed trade execution with error handling"""
        try:
            trade_info = {
                'action': action,
                'symbol': symbol,
                'amount_usdt': amount,
                'price': price,
                'reason': reason,
                'context': context,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.trade_logger.info(f"TRADE_EXECUTION: {json.dumps(trade_info, indent=2)}")
        except Exception as e:
            self.logger.error(f"Trade execution logging failed: {e}")
        
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics with error handling"""
        try:
            self.perf_logger.info(f"PERFORMANCE: {json.dumps(metrics, indent=2)}")
        except Exception as e:
            self.logger.error(f"Performance metrics logging failed: {e}")
        
    def log_strategy_decision(self, strategy: str, decision: str, reasoning: Dict[str, Any]):
        """Log strategy decisions with error handling"""
        try:
            decision_info = {
                'strategy': strategy,
                'decision': decision,
                'reasoning': reasoning,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"STRATEGY_DECISION: {strategy} -> {decision}")
            self.trade_logger.info(f"STRATEGY_DECISION: {json.dumps(decision_info, indent=2)}")
        except Exception as e:
            self.logger.error(f"Strategy decision logging failed: {e}")

# Initialize global logger instance with error handling
try:
    trading_logger = TradingLogger()
    logger = trading_logger.logger
except Exception as e:
    print(f"Trading logger initialization failed: {e}", file=sys.stderr)
    # Create emergency fallback logger
    logger = logging.getLogger("algobot")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Create dummy trading logger
    class DummyTradingLogger:
        def __init__(self):
            self.logger = logger
        def log_trade_execution(self, *args, **kwargs): pass
        def log_performance_metrics(self, *args, **kwargs): pass
        def log_strategy_decision(self, *args, **kwargs): pass
    
    trading_logger = DummyTradingLogger()

def ensure_csv_header(csv_path: str):
    """💎 SADELEŞTİRİLMİŞ CSV header - Anlaşılabilir format"""
    try:
        path = Path(csv_path)
        
        # Parent dizinleri oluştur
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dosya varsa ve içinde içerik varsa bir şey yapma
        if path.exists() and path.stat().st_size > 0:
            return
            
        # 🎨 GÜZEL CSV HEADER
        current_time = datetime.now(timezone.utc)
        header_lines = [
            "# ================================================================================",
            "# 🚀 ENHANCED ALGOBOT TRADING LOG - Clean & Simple Format",
            "# ================================================================================",
            f"# Başlangıç Tarihi: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"# Bot: Enhanced AlgoBot v2.0",
            f"# Başlangıç Sermayesi: ${getattr(settings, 'INITIAL_CAPITAL_USDT', 1000):.2f} USDT",
            f"# Sembol: {getattr(settings, 'SYMBOL', 'BTC/USDT')}",
            "# ================================================================================",
            "" # CSV başlıklarından önce boş satır
        ]
        
        # CSV sütun başlıklarını ekle
        header_lines.append("timestamp,action,symbol,price,amount,cost,fee,balance")
        
        # Başlıkları dosyaya yaz
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header_lines) + '\n')
            
        logger.info(f"Yeni işlem kaydı dosyası oluşturuldu: {csv_path}")
    except Exception as e:
        logger.error(f"CSV başlığı oluşturulurken hata: {e}")
        print(f"Error ensuring CSV header: {e}", file=sys.stderr)

# Hata ayıklama için bir test fonksiyonu ekleyelim:

def test_csv_logging(csv_path: str = None):
    """CSV log dosyası oluşturma işlevini test eder ve sonucu yazdırır"""
    if csv_path is None:
        csv_path = getattr(settings, 'TRADES_CSV_LOG_PATH', 'logs/trades_test.csv')
    
    try:
        logger.info(f"CSV log testi başlıyor: {csv_path}")
        
        # Log dizininin varlığını kontrol et
        log_dir = Path(csv_path).parent
        logger.info(f"Log dizini: {log_dir}, Var mı: {log_dir.exists()}")
        
        # Log dizini yaratma denemesi
        if not log_dir.exists():
            logger.info("Log dizini oluşturuluyor...")
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Log dizini oluşturuldu mu: {log_dir.exists()}")
        
        # CSV header oluşturma
        logger.info("CSV header oluşturma fonksiyonu çağrılıyor...")
        ensure_csv_header(csv_path)
        
        # Dosya varlığı ve içeriği kontrol edilir
        csv_file = Path(csv_path)
        if csv_file.exists():
            logger.info(f"CSV dosyası oluşturuldu: {csv_path}")
            logger.info(f"CSV dosyası boyutu: {csv_file.stat().st_size} bytes")
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                logger.info(f"CSV dosyası {len(lines)} satır içeriyor")
        else:
            logger.error(f"CSV dosyası oluşturulamadı: {csv_path}")
            
        return csv_file.exists()
    except Exception as e:
        logger.error(f"CSV log testi sırasında hata: {str(e)}")
        return False