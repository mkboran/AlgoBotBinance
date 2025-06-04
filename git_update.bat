@echo off
echo 🔄 Git Update Script
echo ===================

echo ✅ Değişiklikleri staging...
git add .

echo ✅ Commit mesajı girin:
set /p commit_msg="Commit mesajı: "

if "%commit_msg%"=="" (
    set commit_msg=📊 Enhanced trading system improvements and optimizations
)

git commit -m "%commit_msg%"

echo ✅ GitHub'a push ediliyor...
git push origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 🎉 Güncelleme başarıyla tamamlandı!
    echo 🔗 Repository: https://github.com/YOUR_USERNAME/AlgoBotBinance
    echo.
) else (
    echo.
    echo ❌ Push işlemi başarısız oldu!
    echo 💡 Olası nedenler:
    echo    - İnternet bağlantısı sorunu
    echo    - GitHub authentication gerekli
    echo    - Repository henüz oluşturulmamış
    echo.
    echo 🔧 Manuel olarak deneyin:
    echo    git status
    echo    git push origin main
    echo.
)

echo.
pause
