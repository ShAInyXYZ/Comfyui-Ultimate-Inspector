@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================
echo   🐳 ComfyUI Docker Container Generator  
echo ========================================================
echo   Starting batch file execution...
echo ========================================================
echo.

REM Simple timestamp generation - fix for Windows date format
for /f "tokens=2 delims=/" %%a in ("%date%") do set day=%%a
for /f "tokens=1-2 delims=:" %%a in ("%time%") do set hour=%%a&set minute=%%b

REM Clean up and format
set day=%day: =%
set hour=%hour: =%
set minute=%minute: =%
set minute=%minute:~0,2%

REM Ensure proper formatting
if "%hour:~1,1%"=="" set hour=0%hour%
if "%minute:~1,1%"=="" set minute=0%minute%

set timestamp=%day%-%hour%%minute%
echo [DEBUG] Generated timestamp: %timestamp%

REM Create logs directory
if not exist "logs" (
    echo [*] Creating logs directory...
    mkdir logs
)

REM Set log file path
set "LOG_FILE=logs\DockerGenerator_%timestamp%.log"
echo [DEBUG] Log file will be: %LOG_FILE%

REM Initialize log file
echo ======================================================== > "%LOG_FILE%"
echo   ComfyUI Docker Container Generator - Session Log >> "%LOG_FILE%"
echo ======================================================== >> "%LOG_FILE%"
echo   Started: %date% %time% >> "%LOG_FILE%"
echo   Working Directory: %cd% >> "%LOG_FILE%"
echo ======================================================== >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo [*] Log file initialized: %LOG_FILE%

REM Check current directory
echo [*] Current directory: %cd%
echo [*] Current directory: %cd% >> "%LOG_FILE%"

REM Check if main script exists
if not exist "ComfyUI-Docker-Generator.py" (
    echo [!] ERROR: ComfyUI-Docker-Generator.py not found!
    echo [!] ERROR: ComfyUI-Docker-Generator.py not found! >> "%LOG_FILE%"
    echo [!] You are in: %cd%
    echo [!] You are in: %cd% >> "%LOG_FILE%"
    echo [!] Please run this from the Comfyui-Docker-Generator directory
    echo [!] Please run this from the Comfyui-Docker-Generator directory >> "%LOG_FILE%"
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo [+] Found ComfyUI-Docker-Generator.py
echo [+] Found ComfyUI-Docker-Generator.py >> "%LOG_FILE%"

REM Check if database exists
if not exist "docker_images_db.json" (
    echo [!] ERROR: docker_images_db.json not found!
    echo [!] ERROR: docker_images_db.json not found! >> "%LOG_FILE%"
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo [+] Found docker_images_db.json
echo [+] Found docker_images_db.json >> "%LOG_FILE%"

REM Navigate to parent directory to look for reports
cd ..
echo [*] Moved to parent directory: %cd%
echo [*] Moved to parent directory: %cd% >> "%LOG_FILE%"

REM Check for reports directory
if not exist "reports" (
    echo [!] ERROR: No reports directory found!
    echo [!] ERROR: No reports directory found! >> "%LOG_FILE%"
    echo [!] Please run ComfyUI Ultimate Inspector first
    echo [!] Please run ComfyUI Ultimate Inspector first >> "%LOG_FILE%"
    echo [!] Expected: reports\XX-XXXX\ directories
    echo [!] Expected: reports\XX-XXXX\ directories >> "%LOG_FILE%"
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo [+] Found reports directory
echo [+] Found reports directory >> "%LOG_FILE%"

REM Count report subdirectories
set count=0
for /d %%i in (reports\*) do set /a count+=1

if %count%==0 (
    echo [!] ERROR: No report subdirectories found!
    echo [!] ERROR: No report subdirectories found! >> "%LOG_FILE%"
    echo [!] Found reports directory but no XX-XXXX subdirectories
    echo [!] Found reports directory but no XX-XXXX subdirectories >> "%LOG_FILE%"
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo [+] Found %count% report directories
echo [+] Found %count% report directories >> "%LOG_FILE%"

REM Find Python executable
set "python_path="

REM Check embedded Python (most common)
if exist "python_embeded\python.exe" (
    set "python_path=python_embeded\python.exe"
    echo [+] Using embedded Python: !python_path!
    echo [+] Using embedded Python: !python_path! >> "%LOG_FILE%"
) else if exist "python_embedded\python.exe" (
    set "python_path=python_embedded\python.exe"
    echo [+] Using embedded Python: !python_path!
    echo [+] Using embedded Python: !python_path! >> "%LOG_FILE%"
) else (
    REM Try system Python
    python --version >nul 2>&1
    if !errorlevel!==0 (
        set "python_path=python"
        echo [+] Using system Python
        echo [+] Using system Python >> "%LOG_FILE%"
    ) else (
        echo [!] ERROR: Python not found!
        echo [!] ERROR: Python not found! >> "%LOG_FILE%"
        echo [!] Checked for embedded and system Python
        echo [!] Checked for embedded and system Python >> "%LOG_FILE%"
        echo.
        echo Press any key to exit...
        pause > nul
        exit /b 1
    )
)

echo.
echo [*] All prerequisites found! Starting Docker Generator...
echo [*] All prerequisites found! Starting Docker Generator... >> "%LOG_FILE%"
echo.

REM Navigate back to Docker Generator directory
cd Comfyui-Docker-Generator
echo [*] Back in Docker Generator directory: %cd%
echo [*] Back in Docker Generator directory: %cd% >> "%LOG_FILE%"

REM Run the Python script
echo [*] Running Docker Generator...
echo [*] Running Docker Generator... >> "%LOG_FILE%"
echo [*] Command: %python_path% ComfyUI-Docker-Generator.py --path ..
echo [*] Command: %python_path% ComfyUI-Docker-Generator.py --path .. >> "%LOG_FILE%"
echo.

"%python_path%" ComfyUI-Docker-Generator.py --path ..

REM Capture exit code
set "exit_code=%errorlevel%"
echo.
echo [*] Docker Generator finished with exit code: %exit_code%
echo [*] Docker Generator finished with exit code: %exit_code% >> "%LOG_FILE%"

if %exit_code%==0 (
    echo [+] SUCCESS! Docker container generated successfully
    echo [+] SUCCESS! Docker container generated successfully >> "%LOG_FILE%"
    echo.
    echo Your Docker package is ready in docker_build_XX-XXXX folder
    echo Run: bash startup.sh to deploy
    echo Or: docker-compose up -d
) else (
    echo [!] FAILED! Docker generation failed with errors
    echo [!] FAILED! Docker generation failed with errors >> "%LOG_FILE%"
    echo.
    echo Check the output above for error details
)

echo.
echo Session completed at %date% %time% >> "%LOG_FILE%"
echo Complete log saved to: %LOG_FILE%
echo.
echo Press any key to exit...
pause > nul