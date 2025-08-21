@echo off
cd ..
:: Generate timestamp (dd-HHmm)
for /f %%i in ('powershell -NoLogo -NoProfile -Command "Get-Date -Format dd-HHmm"') do set datetime=%%i

:: Create run-specific subfolder inside reports
set runfolder=ComfyUI-Ultimate-Inspector\reports\%datetime%
if not exist %runfolder% mkdir %runfolder%

:: Define filename
set filename=Report_Comfy_%datetime%.md

:: Run the inspector, outputting both report + pip_freeze into same folder
python ComfyUI-Ultimate-Inspector\ComfyUI-Ultimate-Inspector.py --output %runfolder%\%filename% --verbose

pause
