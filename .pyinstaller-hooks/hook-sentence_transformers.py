"""
PyInstaller hook for sentence-transformers.

Ensures sentence-transformers and all its dependencies are properly included.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all sentence_transformers modules and data
datas, binaries, hiddenimports = collect_all('sentence_transformers')

# Add specific submodules
hiddenimports += collect_submodules('sentence_transformers')
hiddenimports += collect_submodules('sentence_transformers.backend')
hiddenimports += collect_submodules('sentence_transformers.util')
hiddenimports += collect_submodules('sentence_transformers.models')

# Add dependencies that might be missed
hiddenimports += [
    'torch',
    'transformers',
    'tokenizers',
    'huggingface_hub',
    'scipy',
    'scipy.sparse',
    'scipy.spatial',
    'scipy.spatial.distance',
    'sklearn',
    'sklearn.metrics',
    'sklearn.metrics.pairwise',
]

