"""
PyInstaller hook for scikit-learn (sklearn).

Ensures sklearn and its dependencies are properly included.
Required by sentence-transformers for similarity calculations.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all sklearn submodules
hiddenimports = collect_submodules('sklearn')

# Add specific sklearn modules that are commonly needed
hiddenimports += [
    'sklearn.utils',
    'sklearn.utils._cython_blas',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    'sklearn.metrics',
    'sklearn.metrics.pairwise',
    'sklearn.neighbors',
    'sklearn.tree',
    'sklearn.ensemble',
]

# Collect data files (sklearn has some data files)
datas = collect_data_files('sklearn')

