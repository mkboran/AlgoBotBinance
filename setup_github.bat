@echo off
chcp 65001 >nul
echo 🚀 GitHub Repository Setup
echo ==========================

REM Git kullanıcı bilgilerini kontrol et
echo ✅ Git kullanıcı bilgileri kontrol ediliyor...
git config user.name >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Git kullanıcı bilgileri bulunamadı!
    echo.
    set /p git_name="Git kullanıcı adınızı girin: "
    set /p git_email="Git email adresinizi girin: "
    
    echo ✅ Git kullanıcı bilgileri ayarlanıyor...
    git config --global user.name "!git_name!"
    git config --global user.email "!git_email!"
    
    echo ✅ Git yapılandırması tamamlandı!
    echo.
)

REM Git repository başlat
echo ✅ Git repository başlatılıyor...
git init

REM .gitignore oluştur
echo ✅ .gitignore dosyası oluşturuluyor...
(
echo # AlgoBot Binance - Git Ignore File
echo # =================================
echo.
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo build/
echo develop-eggs/
echo dist/
echo downloads/
echo eggs/
echo .eggs/
echo lib/
echo lib64/
echo parts/
echo sdist/
echo var/
echo wheels/
echo *.egg-info/
echo .installed.cfg
echo *.egg
echo.
echo # Logs and Data
echo logs/
echo *.log
echo historical_data/
echo temp/
echo.
echo # Config Files
echo config.json
echo .env
echo secrets.txt
echo api_keys.txt
echo.
echo # IDE
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
echo.
echo # Backtest Results
echo backtest_results_*.txt
echo *.png
echo *.jpg
) > .gitignore

REM Dosyaları stage'e ekle
echo ✅ Dosyalar stage'e ekleniyor...
git add .

REM İlk commit
echo ✅ İlk commit yapılıyor...
git commit -m "🚀 Initial commit: Enhanced AlgoBot Trading System with Advanced Strategies"

echo.
echo 📋 ✅ Setup tamamlandı! Şimdi push_to_github.bat çalıştırın
echo.
pause
