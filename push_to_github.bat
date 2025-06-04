@echo off
chcp 65001 >nul
echo 🔼 GitHub'a Push İşlemi
echo =======================

echo ⚠️  ÖNCE GitHub'da repository oluşturduğunuzdan emin olun!
echo    Repository URL: https://github.com/YOUR_USERNAME/AlgoBotBinance.git
echo.

set /p github_username="GitHub kullanıcı adınızı girin (örn: mkboran): "

if "%github_username%"=="" (
    echo ❌ Kullanıcı adı boş olamaz!
    pause
    exit /b 1
)

set github_url=https://github.com/%github_username%/AlgoBotBinance.git

echo ✅ Remote origin ekleniyor: %github_url%
git remote add origin %github_url%

echo ✅ Ana branch main olarak ayarlanıyor...
git branch -M main

echo ✅ GitHub'a push ediliyor...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 🎉 BAŞARILI! Projeniz GitHub'a yüklendi!
    echo 🔗 Repository URL: %github_url%
    echo.
    echo 📋 Gelecekteki güncellemeler için:
    echo    .\git_update.bat
    echo.
    echo 🌟 Repository'yi GitHub'da görüntülemek için:
    echo    %github_url%
) else (
    echo.
    echo ❌ Push işlemi başarısız oldu!
    echo 💡 Olası nedenler:
    echo    - GitHub repository henüz oluşturulmamış
    echo    - Yanlış kullanıcı adı: %github_username%
    echo    - Authentication gerekli (GitHub login)
    echo.
    echo 🔧 Manuel push için:
    echo    git push -u origin main
    echo.
    echo 🔐 Authentication gerekiyorsa:
    echo    GitHub → Settings → Developer settings → Personal access tokens
    echo    Veya: gh auth login (GitHub CLI ile)
)

echo.
pause
