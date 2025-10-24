# Script to set up a new virtual environment with Python 3.11 or 3.12

Write-Host "RayWhisper2 - Virtual Environment Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if current venv is activated
if ($env:VIRTUAL_ENV) {
    Write-Host "Deactivating current virtual environment..." -ForegroundColor Yellow
    deactivate
}

# Remove old venv if it exists
if (Test-Path "venv") {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

Write-Host ""
Write-Host "Checking available Python versions..." -ForegroundColor Cyan

# Try to find Python 3.11 or 3.12
$python311 = $null
$python312 = $null

try {
    $python311 = & py -3.11 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Found: $python311" -ForegroundColor Green
    }
} catch {
    # Python 3.11 not found
}

try {
    $python312 = & py -3.12 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Found: $python312" -ForegroundColor Green
    }
} catch {
    # Python 3.12 not found
}

Write-Host ""

# Choose which Python to use
if ($python312) {
    Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Yellow
    py -3.12 -m venv venv
} elseif ($python311) {
    Write-Host "Creating virtual environment with Python 3.11..." -ForegroundColor Yellow
    py -3.11 -m venv venv
} else {
    Write-Host "ERROR: Python 3.11 or 3.12 not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Python 3.14 is too new for most ML packages." -ForegroundColor Yellow
    Write-Host "Please install Python 3.11 or 3.12 from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Recommended: Python 3.11.9 or Python 3.12.7" -ForegroundColor Cyan
    exit 1
}

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Virtual environment created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Current Python version:" -ForegroundColor Cyan
python --version

Write-Host ""
Write-Host "Next step: Run the installation script" -ForegroundColor Cyan
Write-Host "  .\install.ps1" -ForegroundColor White

