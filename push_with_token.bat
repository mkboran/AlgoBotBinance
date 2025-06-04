@echo off
echo 🔐 GitHub Token ile Push
echo ========================

set /p github_username="GitHub kullanıcı adınız: "
set /p github_token="GitHub Personal Access Token: "

if "%github_username%"=="" (
    echo ❌ Kullanıcı adı gerekli!
    pause
    exit /b 1
)

if "%github_token%"=="" (
    echo ❌ Token gerekli!
    echo 💡 GitHub → Settings → Developer settings → Personal access tokens
    pause
    exit /b 1
)

set github_url=https://%github_token%@github.com/%github_username%/AlgoBotBinance.git

echo ✅ Token ile push ediliyor...
git remote add origin %github_url%
git branch -M main
git push -u origin main

echo ✅ İşlem tamamlandı!
pause
