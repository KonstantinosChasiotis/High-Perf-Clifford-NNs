import os, ctypes, sys, numpy as np

# Pick the right shared‐lib name for the platform
_lib_name = "libmv_act.dll" if sys.platform == "win32" else "libmv_act.so"

# We assume you built it into csrc/build/
_lib_path = os.path.join(
    os.path.dirname(__file__),
    "..", "csrc", "build",
    _lib_name
)

# Load it via ctypes
_lib = ctypes.CDLL(_lib_path)
# signature
_lib.mv_act_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # B, C, L, I
    ctypes.POINTER(ctypes.c_float),  # weight
    ctypes.c_int,                   # K
    ctypes.POINTER(ctypes.c_int),   # kernel_blades
    ctypes.POINTER(ctypes.c_float), # out
]
_lib.mv_act_forward.restype = None


def act_forward(x, weight, kernel_blades):
    """
    Run the C implementation of multivector activation (linear aggregation only).

    Accepts inputs of shape (B, C, I) or (B, C, L, I).
    """
    # Ensure numpy arrays with correct dtypes
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    kernel_blades = np.asarray(kernel_blades, dtype=np.int32)

    # Handle 3D (no spatial dim) by inserting L=1
    if x.ndim == 3:
        B, C, I = x.shape
        L = 1
        x4 = x.reshape(B, C, L, I)
    elif x.ndim == 4:
        B, C, L, I = x.shape
        x4 = x
    else:
        raise ValueError(f"act_forward: expected 3D or 4D input, got {x.ndim}D")

    # Weight has shape (C, K)
    K = weight.shape[1]

    # Prepare output buffer
    out4 = np.empty((B, C, L, I), dtype=np.float32)

    # Call into C
    _lib.mv_act_forward(
        x4.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(L), ctypes.c_int(I),
        weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(K),
        kernel_blades.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out4.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    # Remove added dimension for 3D inputs
    if x.ndim == 3:
        return out4.reshape(B, C, I)
    return out4
