import os
import sys
import ctypes
import numpy as np
from typing import Optional

# Pick the right shared‐lib name
_lib_name = "clifford_sigmoid.dll" if sys.platform == "win32" else "libclifford_sigmoid.so"
# We assume you built it into csrc/build/
_lib_path = os.path.join(os.path.dirname(__file__), "..", "csrc", "build", _lib_name)
_lib = ctypes.CDLL(_lib_path)

# Define the C function signature
_lib.clifford_sigmoid_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.c_int,                    # B
    ctypes.c_int,                    # C
    ctypes.c_int,                    # blade
    ctypes.POINTER(ctypes.c_float),  # weights 
    ctypes.POINTER(ctypes.c_float),  # bias
]

def c_sigmoid_forward(
    x: np.ndarray,  # (B, C, D, blade)
    weights: np.ndarray,
    bias: np.ndarray
) -> np.ndarray:
    """
    Run the C implementation of Clifford sigmoid activation and return output of shape (B, C, D, blade).
    
    Args:
        x: Input array of shape (B, C, D, blade) where:
           B is batch size
           C is number of channels
           D is spatial dimension
           blade is number of blades in the Clifford algebra
    """
    # Ensure input is contiguous and float32
    x = np.ascontiguousarray(x, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)

    B, C, n_blades = x.shape
    
    # Allocate output array
    out = np.zeros((B, C, n_blades), dtype=np.float32)
    
    # Call C function
    _lib.clifford_sigmoid_forward(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
        ctypes.c_int(C),
        ctypes.c_int(n_blades),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    
    return out