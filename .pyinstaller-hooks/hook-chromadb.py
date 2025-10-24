"""
PyInstaller hook for ChromaDB.

Ensures all ChromaDB dependencies are properly included.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all chromadb modules and data
datas, binaries, hiddenimports = collect_all('chromadb')

# Add specific submodules that might be missed
hiddenimports += collect_submodules('chromadb')
hiddenimports += collect_submodules('chromadb.config')
hiddenimports += collect_submodules('chromadb.api')
hiddenimports += collect_submodules('chromadb.db')

# Add dependencies
hiddenimports += [
    'sqlite3',
    'onnxruntime',
    'tokenizers',
    'huggingface_hub',
]

