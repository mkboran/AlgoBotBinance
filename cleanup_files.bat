@echo off
chcp 65001 >nul
echo 🧹 Gereksiz Dosyaları Temizleme
echo ==============================

echo ⚠️  Aşağıdaki gereksiz dosyalar silinecek:
echo    - setup_git.bat (duplicate)
echo    - quick_push.bat (gereksiz)
echo    - push_with_token.bat (gereksiz)
echo    - historical_data/.gitkeep (gereksiz - README.md var)
echo.

set /p confirm="Bu dosyaları silmek istediğinizden emin misiniz? (y/N): "

if /i "%confirm%"=="y" (
    echo.
    echo ✅ Gereksiz dosyalar siliniyor...
    
    if exist "setup_git.bat" (
        del "setup_git.bat"
        echo    ❌ setup_git.bat silindi
    )
    
    if exist "quick_push.bat" (
        del "quick_push.bat" 
        echo    ❌ quick_push.bat silindi
    )
    
    if exist "push_with_token.bat" (
        del "push_with_token.bat"
        echo    ❌ push_with_token.bat silindi
    )
    
    if exist "historical_data\.gitkeep" (
        del "historical_data\.gitkeep"
        echo    ❌ historical_data/.gitkeep silindi (README.md yeterli)
    )
    
    echo.
    echo ✅ 🎉 Temizlik tamamlandı!
    echo.
    echo 📋 Kalan önemli dosyalar:
    echo    ✅ setup_github.bat - İlk kurulum
    echo    ✅ push_to_github.bat - İlk push  
    echo    ✅ git_update.bat - Güncellemeler
    echo    ✅ cleanup_files.bat - Temizlik
    echo    ✅ README.md - Ana dokümantasyon
    echo    ✅ historical_data/README.md - Veri klasörü rehberi
    echo.
) else (
    echo ❌ İşlem iptal edildi.
)

echo.
pause
