@echo off
cd /d "%~dp0"

:: Environment
set "PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"
set "PY_CMD=python"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [ENV] .venv
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [ENV] venv
) else if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
    echo [ENV] env
) else (
    :: Prefer Anaconda (PyTorch+CUDA) over Python 3.14 (PyTorch CPU-only)
    py -V:ContinuumAnalytics/Anaconda39-64 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1 && (
        set "PY_CMD=py -V:ContinuumAnalytics/Anaconda39-64"
        echo [ENV] Anaconda GPU
    ) || (
        echo [ENV] system python - may be CPU-only PyTorch
    )
)

echo.
echo ============================================
echo   Restart Semantic Cluster WebUI (port 7860)
echo ============================================
echo.

set PORT=7860

:: Kill process on port
echo [1/3] Finding process on port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo       PID %%a
    taskkill /PID %%a /F >nul 2>&1
    echo       Killed
    goto :killed
)

echo       No process found
:killed

:: Wait for port release
echo       Waiting...
timeout /t 2 /nobreak >nul

:: Repair: remove corrupted -umpy, fix numpy/pandas binary compat
echo.
echo [2/3] Check dependencies...
set "PYTHONUTF8=1"
%PY_CMD% -c "import site,shutil,os; p=site.getsitepackages()[0]; d=os.path.join(p,'-umpy'); shutil.rmtree(d,ignore_errors=True) if os.path.exists(d) else None"
%PY_CMD% -m pip install -q "numpy<2" 2>nul
%PY_CMD% -m pip install -q "numexpr>=2.8.4" "bottleneck>=1.3.6" 2>nul
%PY_CMD% -m pip install -q -r requirements.txt 2>nul || echo       [WARN] pip had errors, continuing...

:: Start WebUI
echo.
echo [3/3] Starting WebUI...
%PY_CMD% -m ui.app

pause
