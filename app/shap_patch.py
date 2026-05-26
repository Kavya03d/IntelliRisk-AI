"""
Monkey-patch NumPy to restore deprecated attributes that SHAP 0.45.0 needs.
Import this BEFORE importing shap anywhere.
"""
import numpy as np

# Restore np.float_ removed in NumPy 2.0
if not hasattr(np, 'float_'):
    np.float_ = np.float64

# Restore np.int_ removed in NumPy 2.0
if not hasattr(np, 'int_'):
    np.int_ = np.int64

# Restore np.bool removed in NumPy 1.24
if not hasattr(np, 'bool'):
    np.bool = np.bool_

# Restore np.object removed in NumPy 1.24
if not hasattr(np, 'object'):
    np.object = object

# Restore np.str removed in NumPy 1.24
if not hasattr(np, 'str'):
    np.str = str

# Restore np.complex_ removed in NumPy 2.0
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

# Restore np.floating used by SHAP color module
if not hasattr(np, 'floating'):
    np.floating = np.floating  # already exists but patch obj2sctype below

# Restore np.obj2sctype removed in NumPy 2.0
if not hasattr(np, 'obj2sctype'):
    def _obj2sctype(rep):
        try:
            return np.dtype(rep).type
        except Exception:
            return None
    np.obj2sctype = _obj2sctype
