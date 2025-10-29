@echo off
REM Simple build script for RayWhisper executable
REM This is a simpler alternative to build_exe.py

echo ========================================
echo RayWhisper - Simple Build Script
echo ========================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Error: Virtual environment not activated
    echo Please run: venv\Scripts\activate
    pause
    exit /b 1
)

REM Verify required config files exist
echo Verifying required files...
if not exist "config\config.yaml" (
    echo Error: config\config.yaml not found!
    echo Please ensure config\config.yaml exists before building.
    pause
    exit /b 1
)
echo   OK: config\config.yaml found

if exist "config\config.example.yaml" (
    echo   OK: config\config.example.yaml found
) else (
    echo   Warning: config\config.example.yaml not found (optional^)
)

REM Install PyInstaller if not present
echo Checking for PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Clean previous builds
echo.
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist raywhisper.spec del raywhisper.spec

REM Build the executable
echo.
echo Building executable...
echo This may take several minutes...
echo.

pyinstaller ^
    --name=raywhisper ^
    --console ^
    --onedir ^
    --clean ^
    --paths=src ^
    --add-data "config\config.yaml;config" ^
    --add-data "config\config.example.yaml;config" ^
    --add-data "README.md;." ^
    --add-data "LICENSE;." ^
    --hidden-import=raywhisper ^
    --hidden-import=raywhisper.presentation ^
    --hidden-import=raywhisper.presentation.cli ^
    --hidden-import=raywhisper.presentation.app ^
    --hidden-import=chromadb ^
    --hidden-import=sentence_transformers ^
    --hidden-import=sentence_transformers.backend ^
    --hidden-import=sentence_transformers.util ^
    --hidden-import=transformers ^
    --hidden-import=transformers.models ^
    --hidden-import=tqdm ^
    --hidden-import=tqdm.auto ^
    --hidden-import=faster_whisper ^
    --hidden-import=ctranslate2 ^
    --hidden-import=onnxruntime ^
    --hidden-import=sounddevice ^
    --hidden-import=pynput ^
    --hidden-import=pynput.keyboard._win32 ^
    --hidden-import=torch ^
    --hidden-import=scipy ^
    --hidden-import=scipy.sparse ^
    --hidden-import=scipy.spatial ^
    --hidden-import=sklearn ^
    --hidden-import=sklearn.metrics ^
    --hidden-import=sklearn.metrics.pairwise ^
    --hidden-import=sklearn.utils ^
    --hidden-import=sklearn.utils._cython_blas ^
    --exclude-module=matplotlib ^
    --exclude-module=pandas ^
    --exclude-module=pytest ^
    --copy-metadata=tqdm ^
    --copy-metadata=transformers ^
    --copy-metadata=tokenizers ^
    --copy-metadata=huggingface_hub ^
    --copy-metadata=safetensors ^
    --copy-metadata=regex ^
    --copy-metadata=filelock ^
    --copy-metadata=numpy ^
    --copy-metadata=packaging ^
    --copy-metadata=pyyaml ^
    --copy-metadata=requests ^
    src/raywhisper/__main__.py

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

REM Create data and logs directories
echo.
echo Creating data directories...
mkdir dist\raywhisper\data\chroma 2>nul
mkdir dist\raywhisper\logs 2>nul

REM Copy existing chroma data if it exists
if exist data\chroma (
    echo Copying existing ChromaDB data...
    xcopy /E /I /Y data\chroma dist\raywhisper\data\chroma
)

REM Create README
echo Creating README...
(
echo RayWhisper - Voice to Text with RAG
echo ====================================
echo.
echo Quick Start:
echo   1. raywhisper.exe populate ^<path-to-docs^> --clear
echo   2. raywhisper.exe run
echo   3. Enable Caps Lock to record, disable to transcribe
echo.
echo Configuration: Edit config/config.yaml
echo Logs: Check logs/raywhisper.log
echo.
echo For more info: https://github.com/Fredrik-C/RayWhisper2
) > dist\raywhisper\README.txt

REM Create run batch file
echo Creating launcher...
(
echo @echo off
echo echo Starting RayWhisper... Be patient, first launch may take time
echo raywhisper.exe run
echo pause
) > dist\raywhisper\run.bat

echo.
echo ========================================
echo BUILD COMPLETE!
echo ========================================
echo.
echo Distribution package: dist\raywhisper\
echo.
echo Test it:
echo   cd dist\raywhisper
echo   raywhisper.exe --version
echo.
pause

