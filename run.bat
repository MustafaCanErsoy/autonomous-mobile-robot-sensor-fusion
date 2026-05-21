@echo off
chcp 65001 > nul
echo ============================================
echo  FrostBot - Pizza Fabrikasi Simulasyonu
echo ============================================
echo.

python --version > nul 2>&1
if errorlevel 1 (
    echo HATA: Python bulunamadi. python.org/downloads adresinden yukleyin.
    pause
    exit /b 1
)

echo [1/2] Bagimliliklar yukleniyor...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo HATA: Bagimliliklar yuklenemedi.
    pause
    exit /b 1
)

echo [2/2] Simulasyon baslatiliyor...
echo.
python main.py

echo.
echo Tum ciktilar 'results\' klasorune kaydedildi.
pause
