# RayWhisper2 Installation Script for Windows
# This script installs dependencies in the correct order to avoid build issues

Write-Host "Installing RayWhisper2 dependencies..." -ForegroundColor Green

# Upgrade pip first
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install numpy from pre-built wheel first
# Note: Using numpy 2.x for Python 3.14 compatibility
Write-Host "`nInstalling NumPy (pre-built wheel)..." -ForegroundColor Yellow
pip install "numpy>=2.0.0" --only-binary=:all:

# Install other core dependencies
Write-Host "`nInstalling core dependencies..." -ForegroundColor Yellow
pip install pydantic>=2.6.0
pip install pydantic-settings>=2.3.0
pip install python-dotenv>=1.0.1
pip install loguru>=0.7.2
pip install click>=8.1.0

# Install audio dependencies
Write-Host "`nInstalling audio dependencies..." -ForegroundColor Yellow
pip install sounddevice>=0.4.7

# Install keyboard control
Write-Host "`nInstalling keyboard control..." -ForegroundColor Yellow
pip install pynput>=1.7.7
pip install pyperclip>=1.8.2

# Install ML dependencies (these may take a while)
Write-Host "`nInstalling ML dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers>=3.1.0
pip install faster-whisper>=1.2.0

# Install ChromaDB
Write-Host "`nInstalling ChromaDB..." -ForegroundColor Yellow
pip install chromadb>=0.5.0

# Install development dependencies
Write-Host "`nInstalling development dependencies..." -ForegroundColor Yellow
pip install pytest>=8.3.0
pip install pytest-asyncio>=0.23.0
pip install pytest-cov>=5.0.0
pip install pytest-benchmark>=4.0.0
pip install mypy>=1.11.0
pip install ruff>=0.6.0

# Install the package in editable mode
Write-Host "`nInstalling RayWhisper2 in editable mode..." -ForegroundColor Yellow
pip install -e .

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Run tests: pytest" -ForegroundColor White
Write-Host "  2. Populate embeddings: raywhisper populate ./docs --clear" -ForegroundColor White
Write-Host "  3. Run the app: raywhisper run" -ForegroundColor White

