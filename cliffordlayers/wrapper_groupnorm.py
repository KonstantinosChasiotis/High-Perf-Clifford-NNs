# cliffordlayers/wrappers_groupnorm.py

import os
import sys
import ctypes
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union

# Pick the right shared‐lib name
_lib_name = "clifford_groupnorm.dll" if sys.platform == "win32" else "libclifford_groupnorm.so"
# We assume you built it into csrc/build/
_lib_path = os.path.join(os.path.dirname(__file__), "..", "csrc", "build", _lib_name)
_lib = ctypes.CDLL(_lib_path)

def c_groupnorm_forward(
    x: torch.Tensor,            # (B, C, *D, I)
    n_blades: int,              # = I
    num_groups: int,  
    running_mean: Optional[torch.Tensor] = None,
    running_cov: Optional[torch.Tensor] = None,
    weight: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    bias: Optional[Union[torch.Tensor, nn.Parameter]] = None,      
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,    
) -> torch.Tensor:
    
    #assert user does not want to use unimplemented futures
    assert running_mean is None, "running_mean is not None, unimplemented future"
    assert running_cov is None, "running_cov is not None, unimplemented future"
    assert weight is None, "weight is not None, unimplemented future"
    assert training == True, "training False, unimplemented future"

    #tensor -> np.array
    x = x.detach().cpu().numpy().astype(np.float32)

    #shape checks
    B, C, *D, I = x.shape
    assert n_blades == I, "n_blades not equal to I"
    assert C % num_groups == 0, "num_groups must divide C"

    #*D can just be treated as a single flattened dimension
    D_flat = 1
    for dim in D:
        D_flat *= dim

    #setup in and out put
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros((B,C,*D,I), dtype=np.float32)

    # call into C
    _lib.clifford_groupnorm_forward(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
        ctypes.c_int(C),
        ctypes.c_int(D_flat),
        ctypes.c_int(I),
        ctypes.c_int(num_groups),
        ctypes.c_float(eps),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    return torch.from_numpy(out) #return a tensor cast (wrapper provides exactly the same interface)
