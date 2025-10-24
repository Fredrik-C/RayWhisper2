"""
PyInstaller hook for transformers (HuggingFace).

Ensures transformers and all its model implementations are properly included.
Required by sentence-transformers and reranker models.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

# Collect all transformers modules and data
datas, binaries, hiddenimports = collect_all('transformers')

# Add specific submodules
hiddenimports += collect_submodules('transformers')
hiddenimports += collect_submodules('transformers.models')

# Collect metadata for transformers and its dependencies
# This is required because transformers checks package versions at runtime
metadata_packages = [
    'transformers',
    'tqdm',
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

for pkg in metadata_packages:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass  # Skip if package not found

# Add dependencies
hiddenimports += [
    'tokenizers',
    'huggingface_hub',
    'safetensors',
    'regex',
    'tqdm',
]

