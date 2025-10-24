"""
PyInstaller hook for faster-whisper.

Ensures faster-whisper and its ONNX model assets are properly included.
The silero VAD model files are required for voice activity detection.
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

# Collect all faster_whisper modules and data
datas, binaries, hiddenimports = collect_all('faster_whisper')

# Collect data files (ONNX models for VAD)
datas += collect_data_files('faster_whisper')

# Try to find and include the assets directory explicitly
try:
    import faster_whisper
    fw_path = os.path.dirname(faster_whisper.__file__)
    assets_path = os.path.join(fw_path, 'assets')
    
    if os.path.exists(assets_path):
        # Add all files in the assets directory
        for file in os.listdir(assets_path):
            file_path = os.path.join(assets_path, file)
            if os.path.isfile(file_path):
                datas.append((file_path, 'faster_whisper/assets'))
except Exception:
    pass  # If we can't find it, collect_data_files should have gotten it

# Add dependencies
hiddenimports += [
    'ctranslate2',
    'onnxruntime',
]

