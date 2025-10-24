"""
PyInstaller hook for SciPy.

Ensures SciPy and its dependencies are properly included.
Required by scikit-learn which is used by sentence-transformers.
"""

from PyInstaller.utils.hooks import collect_submodules

# Collect all scipy submodules
hiddenimports = collect_submodules('scipy')

# Add specific scipy modules that are commonly needed
hiddenimports += [
    'scipy.sparse',
    'scipy.sparse.csgraph',
    'scipy.sparse.linalg',
    'scipy.special',
    'scipy.spatial',
    'scipy.spatial.distance',
    'scipy.spatial.transform',
    'scipy.linalg',
    'scipy.integrate',
    'scipy.optimize',
]

