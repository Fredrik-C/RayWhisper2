"""Entry point for RayWhisper when run as a module or executable."""

import sys
from pathlib import Path

# Add the parent directory to the path to ensure imports work
# This is needed for PyInstaller builds
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = Path(sys.executable).parent
else:
    # Running as script
    application_path = Path(__file__).parent.parent

# Ensure the package is importable
if str(application_path) not in sys.path:
    sys.path.insert(0, str(application_path))

from raywhisper.presentation.cli import cli

if __name__ == "__main__":
    cli()

