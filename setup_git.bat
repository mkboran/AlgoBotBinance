@echo off
echo 🚀 GitHub Repository Setup Script
echo ================================

cd /d "C:\Projects\AlgoBotBinance"

echo ✅ Git repository başlatılıyor...
git init

echo ✅ .gitignore dosyası oluşturuluyor...
echo # AlgoBot Binance - Git Ignore File > .gitignore
echo # ================================= >> .gitignore
echo. >> .gitignore
echo # Python >> .gitignore
echo __pycache__/ >> .gitignore
echo *.py[cod] >> .gitignore
echo *$py.class >> .gitignore
echo *.so >> .gitignore
echo .Python >> .gitignore
echo build/ >> .gitignore
echo develop-eggs/ >> .gitignore
echo dist/ >> .gitignore
echo downloads/ >> .gitignore
echo eggs/ >> .gitignore
echo .eggs/ >> .gitignore
echo lib/ >> .gitignore
echo lib64/ >> .gitignore
echo parts/ >> .gitignore
echo sdist/ >> .gitignore
echo var/ >> .gitignore
echo wheels/ >> .gitignore
echo *.egg-info/ >> .gitignore
echo .installed.cfg >> .gitignore
echo *.egg >> .gitignore
echo. >> .gitignore
echo # Logs and Data >> .gitignore
echo logs/ >> .gitignore
echo *.log >> .gitignore
echo historical_data/ >> .gitignore
echo temp/ >> .gitignore
echo. >> .gitignore
echo # Config Files >> .gitignore
echo config.json >> .gitignore
echo .env >> .gitignore
echo secrets.txt >> .gitignore
echo api_keys.txt >> .gitignore
echo. >> .gitignore
echo # IDE >> .gitignore
echo .vscode/ >> .gitignore
echo .idea/ >> .gitignore
echo *.swp >> .gitignore
echo *.swo >> .gitignore
echo. >> .gitignore
echo # OS >> .gitignore
echo .DS_Store >> .gitignore
echo Thumbs.db >> .gitignore

echo ✅ İlk commit hazırlanıyor...
git add .
git commit -m "🚀 Initial commit: Enhanced AlgoBot Trading System"

echo.
echo 📋 Şimdi GitHub'da yeni repository oluşturun:
echo    1. https://github.com/new adresine gidin
echo    2. Repository adı: AlgoBotBinance
echo    3. Description: Advanced Cryptocurrency Trading Bot with Enhanced Strategies
echo    4. Private/Public seçin
echo    5. README.md, .gitignore eklemeyin (zaten var)
echo.
echo 📋 GitHub repository URL'sini aldıktan sonra:
echo    git remote add origin https://github.com/USERNAME/AlgoBotBinance.git
echo    git branch -M main
echo    git push -u origin main
echo.
pause
