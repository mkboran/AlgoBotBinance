@echo off
chcp 65001 >nul
echo 🔄 GitHub'dan Değişiklikleri Çekme
echo =================================

echo ✅ Mevcut durumu kontrol ediliyor...
git status

echo.
echo ✅ GitHub'dan son değişiklikler çekiliyor...
git pull origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 🎉 Değişiklikler başarıyla çekildi!
    echo 🔄 VSCode'da dosyalar otomatik güncellenecek
    echo.
    echo 📋 Çekilen değişiklikleri görmek için:
    echo    git log --oneline -5
    echo.
) else (
    echo.
    echo ❌ Pull işlemi başarısız oldu!
    echo 💡 Olası nedenler:
    echo    - Yerel değişiklikler var (commit edilmemiş)
    echo    - İnternet bağlantısı sorunu
    echo    - Merge conflict
    echo.
    echo 🔧 Çözüm önerileri:
    echo    git status (durumu kontrol et)
    echo    git stash (yerel değişiklikleri kaydet)
    echo    git pull origin main (tekrar dene)
    echo    git stash pop (kaydedilen değişiklikleri geri al)
)

echo.
pause
