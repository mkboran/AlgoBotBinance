@echo off
chcp 65001 >nul
echo 🔍 Git Status Check
echo ==================

echo ✅ Git durumu kontrol ediliyor...
git status

echo.
echo ✅ Son commit'ler:
git log --oneline -5

echo.
echo ✅ Remote repository durumu:
git remote -v

echo.
echo 📋 Kullanılabilir komutlar:
echo    .\git_pull.bat - GitHub'dan değişiklikleri çek
echo    .\git_update.bat - Değişiklikleri GitHub'a gönder
echo    .\git_status.bat - Bu durumu tekrar göster

echo.
pause
