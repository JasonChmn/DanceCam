@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo === Installation de WebcamDance ===
echo.

rem --- GPU NVIDIA ---
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [!] Pilote NVIDIA introuvable. Installe le pilote GeForce depuis nvidia.com, puis relance.
    goto :fin_erreur
)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo.

rem --- uv : installe Python 3.11 et les paquets, rien à installer à la main ---
where uv >nul 2>&1
if errorlevel 1 (
    echo Installation de uv...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :fin_erreur
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

if not exist ".venv\Scripts\python.exe" (
    echo Création de l'environnement Python 3.11...
    uv venv --seed --python 3.11 .venv
    if errorlevel 1 goto :fin_erreur
)
set "PY=.venv\Scripts\python.exe"

echo Installation de PyTorch CUDA, quelques Go à télécharger...
uv pip install --python %PY% torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :fin_erreur
echo Installation des dépendances du jeu...
uv pip install --python %PY% -r requirements.txt
if errorlevel 1 goto :fin_erreur

rem --- vérification CUDA ---
%PY% -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA :', torch.cuda.get_device_name(0) if ok else 'INDISPONIBLE'); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo [!] PyTorch ne voit pas le GPU : mets à jour le pilote NVIDIA puis relance installer.bat.
    goto :fin_erreur
)

rem --- modèles de pose : jeu et import, téléchargés une fois ---
%PY% -c "from ultralytics import YOLO; YOLO('yolo26s-pose.pt'); YOLO('yolo26x-pose.pt')"
if errorlevel 1 goto :fin_erreur

rem --- raccourci sur le Bureau ---
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\WebcamDance.lnk'); $s.TargetPath = '%~dp0jouer.bat'; $s.WorkingDirectory = '%~dp0'; $s.Save()"
echo Raccourci WebcamDance créé sur le Bureau.

if not exist "songs\" echo [i] Pas de dossier songs : copie celui du PC Linux ici, ou importe des chansons depuis le menu.

echo.
choice /c ON /m "Lancer les diagnostics webcam puis GPU, environ 45 s"
if errorlevel 2 goto :fin_ok
%PY% tools\diag_webcam.py --seconds 10
%PY% tools\diag_inference.py --players 1 --seconds 30

:fin_ok
echo.
echo Installation terminée. Lancer le jeu : jouer.bat ou le raccourci du Bureau.
pause
exit /b 0

:fin_erreur
echo.
echo Installation interrompue.
pause
exit /b 1
