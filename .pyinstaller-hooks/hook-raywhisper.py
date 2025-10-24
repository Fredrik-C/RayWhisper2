"""
PyInstaller hook for RayWhisper package.

This hook ensures all necessary modules and data files are included.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all raywhisper modules
datas, binaries, hiddenimports = collect_all('raywhisper')

# Ensure all submodules are included
hiddenimports += collect_submodules('raywhisper')
hiddenimports += collect_submodules('raywhisper.presentation')
hiddenimports += collect_submodules('raywhisper.application')
hiddenimports += collect_submodules('raywhisper.infrastructure')
hiddenimports += collect_submodules('raywhisper.domain')
hiddenimports += collect_submodules('raywhisper.config')

