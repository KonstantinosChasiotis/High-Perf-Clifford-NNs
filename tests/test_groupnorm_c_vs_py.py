# tests/test_groupnorm_c_vs_py.py

import numpy as np
import torch
import pytest
import numpy as np


from cliffordlayers.nn.functional.groupnorm import (
    clifford_group_norm,
)
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.wrapper_groupnorm   import c_groupnorm_forward

@pytest.mark.parametrize("g", [
    [1.0], [-1.0], [1.0,1.0], [-1.0,-1.0], [1.0,1.0,1.0]
])
@pytest.mark.parametrize("B,C,D,num_groups", [
    (2,4,3,2), (1,4,3,2), (7,9,5,3), (16, 40, 9, 8)
])
def test_groupnorm_c_matches_python(g, B, C, D, num_groups):

    eps = 1e-5
    if(CliffordSignature(g).n_blades > 4):
        eps = 1e-1 #for 3D numerically very unstable with low eps pertubation

    x = torch.randn(B, C, D, CliffordSignature(g).n_blades)

    x_norm = clifford_group_norm(
            x,
            CliffordSignature(g).n_blades,
            num_groups=num_groups,
            eps=eps,
        )

    x_norm_c = c_groupnorm_forward(
        x,
        CliffordSignature(g).n_blades,
        num_groups=num_groups,
        eps=eps,
    )

    torch.testing.assert_close(x_norm, x_norm_c)
    