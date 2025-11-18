"""
Build script for creating a redistributable RayWhisper executable.

This script uses PyInstaller to create a standalone .exe file with:
- All Python dependencies bundled
- ChromaDB data directory kept external
- Configuration files included
- ML models downloaded at runtime (not embedded)

Usage:
    python build_exe.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("✗ PyInstaller not found")
        print("\nInstalling PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed")
        return True


def clean_build_dirs():
    """Clean previous build artifacts."""
    print("\nCleaning previous build artifacts...")
    dirs_to_clean = ["build", "dist", "__pycache__"]
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")
    
    # Remove spec file if exists
    if os.path.exists("raywhisper.spec"):
        os.remove("raywhisper.spec")
        print("  Removed raywhisper.spec")
    
    print("✓ Cleanup complete")


def create_build_script():
    """Create the PyInstaller spec file for better control."""
    spec_content = """# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata, collect_data_files
import os

block_cipher = None

# Collect package metadata for transformers and its dependencies
metadata_packages = [
    'tqdm',
    'transformers',
    'tokenizers',
    'huggingface_hub',
    'safetensors',
    'regex',
    'filelock',
    'numpy',
    'packaging',
    'pyyaml',
    'requests',
]

# Collect all metadata
datas_metadata = []
for pkg in metadata_packages:
    try:
        datas_metadata.extend(copy_metadata(pkg))
    except Exception:
        pass  # Skip if package not found

# Collect faster_whisper assets (ONNX models for VAD)
datas_faster_whisper = []
try:
    import faster_whisper
    fw_path = os.path.dirname(faster_whisper.__file__)
    assets_path = os.path.join(fw_path, 'assets')

    if os.path.exists(assets_path):
        for file in os.listdir(assets_path):
            file_path = os.path.join(assets_path, file)
            if os.path.isfile(file_path):
                datas_faster_whisper.append((file_path, 'faster_whisper/assets'))
        print(f"Found {len(datas_faster_whisper)} faster_whisper asset files")
except Exception as e:
    print(f"Warning: Could not collect faster_whisper assets: {e}")
    # Fallback to collect_data_files
    try:
        datas_faster_whisper = collect_data_files('faster_whisper')
    except Exception:
        pass

# Collect all source files
a = Analysis(
    ['src/raywhisper/__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('config/config.yaml', 'config'),
        ('config/config.example.yaml', 'config'),
        ('README.md', '.'),
        ('LICENSE', '.'),
    ] + datas_metadata + datas_faster_whisper,
    hiddenimports=[
        'raywhisper',
        'raywhisper.presentation',
        'raywhisper.presentation.cli',
        'raywhisper.presentation.app',
        'raywhisper.application',
        'raywhisper.application.services',
        'raywhisper.application.use_cases',
        'raywhisper.infrastructure',
        'raywhisper.infrastructure.audio',
        'raywhisper.infrastructure.transcription',
        'raywhisper.infrastructure.vector_db',
        'raywhisper.infrastructure.embeddings',
        'raywhisper.infrastructure.keyboard',
        'raywhisper.infrastructure.output',
        'raywhisper.infrastructure.parsers',
        'raywhisper.domain',
        'raywhisper.config',
        # Third-party dependencies
        'chromadb',
        'chromadb.config',
        'sentence_transformers',
        'sentence_transformers.backend',
        'sentence_transformers.util',
        'transformers',
        'transformers.models',
        'tqdm',
        'tqdm.auto',
        'faster_whisper',
        'ctranslate2',
        'onnxruntime',
        'sounddevice',
        'pynput',
        'pynput.keyboard',
        'pynput.keyboard._win32',
        'pyperclip',
        'click',
        'loguru',
        'pydantic',
        'pydantic_settings',
        'torch',
        'numpy',
        # SciPy and sklearn (required by sentence_transformers)
        'scipy',
        'scipy.sparse',
        'scipy.sparse.csgraph',
        'scipy.special',
        'scipy.spatial',
        'scipy.spatial.distance',
        'sklearn',
        'sklearn.metrics',
        'sklearn.metrics.pairwise',
        'sklearn.utils',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors',
        'sklearn.tree',
        # ChromaDB dependencies
        'sqlite3',
        'onnxruntime',
        'tokenizers',
        'huggingface_hub',
    ],
    hookspath=['.pyinstaller-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'mypy',
        'ruff',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='raywhisper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='raywhisper',
)
"""
    
    with open("raywhisper.spec", "w") as f:
        f.write(spec_content)
    
    print("✓ Created raywhisper.spec")


def build_executable():
    """Run PyInstaller to build the executable."""
    print("\nBuilding executable with PyInstaller...")
    print("This may take several minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable,
            "-m", "PyInstaller",
            "raywhisper.spec",
            "--clean",
        ])
        print("\n✓ Build complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed: {e}")
        return False


def create_distribution_package():
    """Create the final distribution package."""
    print("\nCreating distribution package...")

    dist_dir = Path("dist/raywhisper")
    if not dist_dir.exists():
        print("✗ Distribution directory not found")
        return False

    # Verify config files were copied
    config_dir = dist_dir / "config"
    config_yaml = config_dir / "config.yaml"
    config_example = config_dir / "config.example.yaml"

    if not config_yaml.exists():
        print("⚠ Warning: config.yaml not found in distribution, copying...")
        config_dir.mkdir(exist_ok=True)
        source_config = Path("config/config.yaml")
        if source_config.exists():
            shutil.copy(source_config, config_yaml)
            print("  ✓ Copied config.yaml")
        else:
            print("  ✗ Error: config/config.yaml not found in source!")
            return False
    else:
        print("  ✓ config.yaml present")

    if not config_example.exists():
        print("  Copying config.example.yaml...")
        source_example = Path("config/config.example.yaml")
        if source_example.exists():
            shutil.copy(source_example, config_example)
            print("  ✓ Copied config.example.yaml")
    else:
        print("  ✓ config.example.yaml present")

    # Create data directory structure
    data_dir = dist_dir / "data"
    data_dir.mkdir(exist_ok=True)

    chroma_dir = data_dir / "chroma"
    chroma_dir.mkdir(exist_ok=True)

    # Create logs directory
    logs_dir = dist_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Copy existing chroma data if it exists
    source_chroma = Path("data/chroma")
    if source_chroma.exists() and any(source_chroma.iterdir()):
        print("  Copying existing ChromaDB data...")
        shutil.copytree(source_chroma, chroma_dir, dirs_exist_ok=True)
    
    # Copy the comprehensive distribution README
    source_readme = Path("DISTRIBUTION_README.txt")
    if source_readme.exists():
        shutil.copy(source_readme, dist_dir / "README.txt")
        print("  Copied distribution README")
    else:
        # Fallback: create a basic README
        readme_content = """# RayWhisper - Voice to Text with RAG

## Quick Start

1. **First Run - Populate Embeddings:**
    raywhisper.exe populate <path-to-docs2ingest> --clear

2. **Run the Application:**
   raywhisper.exe run

3. **Use Voice-to-Text:**
   - Hold Ctrl+Shift+Space to start recording
   - Speak while holding the keys
   - Release to transcribe and type the text

For more information, visit: https://github.com/Fredrik-C/RayWhisper2
"""
        with open(dist_dir / "README.txt", "w") as f:
            f.write(readme_content)
        print("  Created basic README")
    
    # Create a batch file for easy launching
    batch_content = """@echo off
echo RayWhisper - Voice to Text with RAG
echo ====================================
echo.
echo Starting RayWhisper... Be patient, first launch may take time
echo.
raywhisper.exe run
pause
"""
    
    with open(dist_dir / "run.bat", "w") as f:
        f.write(batch_content)
    
    print("✓ Distribution package created")
    return True


def print_summary():
    """Print build summary and next steps."""
    print("\n" + "="*60)
    print("BUILD SUCCESSFUL!")
    print("="*60)
    print("\nDistribution package location:")
    print(f"  {Path('dist/raywhisper').absolute()}")
    print("\nPackage contents:")
    print("  ├── raywhisper.exe       - Main executable")
    print("  ├── run.bat              - Quick launch script")
    print("  ├── README.txt           - User instructions")
    print("  ├── config/              - Configuration files")
    print("  ├── data/chroma/         - Vector database (external)")
    print("  ├── logs/                - Application logs")
    print("  └── _internal/           - Python runtime and dependencies")
    print("\nNext steps:")
    print("  1. Test the executable:")
    print("     cd dist/raywhisper")
    print("     raywhisper.exe --version")
    print()
    print("  2. Populate embeddings:")
    print("     raywhisper.exe populate <path-to-docs2ingest> --clear")
    print()
    print("  3. Run the application:")
    print("     raywhisper.exe run")
    print()
    print("  4. Distribute the entire 'dist/raywhisper' folder")
    print("\nNote: ChromaDB data is stored in data/chroma/ (not embedded in .exe)")
    print("="*60)


def main():
    """Main build process."""
    print("="*60)
    print("RayWhisper - Build Redistributable Executable")
    print("="*60)

    # Check if we're in the right directory
    if not Path("src/raywhisper").exists():
        print("\n✗ Error: Must run from project root directory")
        print("  Current directory:", Path.cwd())
        sys.exit(1)

    # Verify required config files exist
    print("\nVerifying required files...")
    config_yaml = Path("config/config.yaml")
    if not config_yaml.exists():
        print("✗ Error: config/config.yaml not found!")
        print("  Please ensure config/config.yaml exists before building.")
        sys.exit(1)
    print("  ✓ config/config.yaml found")

    config_example = Path("config/config.example.yaml")
    if config_example.exists():
        print("  ✓ config/config.example.yaml found")
    else:
        print("  ⚠ Warning: config/config.example.yaml not found (optional)")
    
    # Step 1: Check PyInstaller
    if not check_pyinstaller():
        sys.exit(1)
    
    # Step 2: Clean previous builds
    clean_build_dirs()
    
    # Step 3: Create spec file
    create_build_script()
    
    # Step 4: Build executable
    if not build_executable():
        sys.exit(1)
    
    # Step 5: Create distribution package
    if not create_distribution_package():
        sys.exit(1)
    
    # Step 6: Print summary
    print_summary()


if __name__ == "__main__":
    main()

