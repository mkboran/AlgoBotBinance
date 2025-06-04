@echo off
chcp 65001 >nul
echo 🔄 Git Update Script
echo ===================

echo ✅ Değişiklikleri staging...
git add .

echo ✅ Commit mesajı girin:
set /p commit_msg="Commit mesajı (boş bırakırsanız otomatik): "

if "%commit_msg%"=="" (
    set commit_msg=📊 Enhanced trading system improvements and optimizations
)

git commit -m "%commit_msg%"

echo ✅ GitHub'a push ediliyor...
git push origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 🎉 Güncelleme başarıyla tamamlandı!
    echo 🔗 Repository: https://github.com/mkboran/AlgoBotBinance
    echo.
    echo 📋 Son değişiklikleriniz GitHub'a yüklendi!
    echo 🔍 Değişiklikleri görmek için repository'yi ziyaret edin.
    echo.
    echo 💡 Unutmayın:
    echo    - historical_data klasörü yapısı GitHub'da
    echo    - CSV dosyaları boyut nedeniyle Git'e dahil değil
    echo    - Veri indirmek için: python data_downloader.py
) else (
    echo.
    echo ❌ Push işlemi başarısız oldu!
    echo 💡 Olası nedenler:
    echo    - İnternet bağlantısı sorunu
    echo    - GitHub authentication süresi dolmuş
    echo.
    echo 🔧 Manuel olarak deneyin:
    echo    git status
    echo    git push origin main
    echo.
)

echo.
pause
