# tests/test_groupnorm_c_vs_py.py

import numpy as np
import torch
import pytest
import numpy as np

from cliffordlayers.nn.functional.groupnorm import (
    clifford_group_norm,
)
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.wrapper_groupnorm   import c_clifford_group_norm

B = 2
C = 8
D = 1
I = 4

x = torch.randn(B, C, D, I)


x_norm_c = c_clifford_group_norm(
    x,
    I,
    num_groups=4,
    running_mean=None,
    running_cov=None,
    weight=None,
    bias=None,
)
