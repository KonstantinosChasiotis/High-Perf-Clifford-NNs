# tests/test_act_c_vs_py.py
import itertools
import pytest

import numpy as np
import torch

from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.wrapper_act import act_forward
from cliffordlayers.nn.modules.gcan import MultiVectorAct

# ────────────────────────────────────────────────────────────────────────────────
# (batch, channels, input_blades) cases:
#  - batch ∈ {1, 2, 4}
#  - channels ∈ {1, 2, 3}
#  - input_blades: all combinations of length 1,2,3 from blade IDs 0..7
# Then truncate to the first 80 for speed.
# ────────────────────────────────────────────────────────────────────────────────
_batches  = [1, 2, 4]
_channels = [1, 2, 3]
_blade_ids = range(8)

_cases = []
for B in _batches:
    for C in _channels:
        for length in [1, 2, 3]:
            for blades in itertools.combinations(_blade_ids, length):
                _cases.append((B, C, blades))
# keep only the first 80 - we just chose to have 80 test cases
_cases = _cases[:80]

@pytest.mark.parametrize("batch, channels, input_blades", _cases)
def test_shape_and_values(batch, channels, input_blades):
    # 1) Setup algebra and dimensions
    algebra = CliffordAlgebra([0, 1, 1, 1])
    I_in = len(input_blades)

    # 2) Prepare random input of shape (B, C, I_in)
    x_torch = torch.randn(batch, channels, I_in)
    x_np     = x_torch.cpu().detach().numpy().astype(np.float32)

    # 3) Determine full algebra dimension I_full
    v_full = algebra.embed(x_torch, input_blades)   # (B, C, I_full)
    I_full = v_full.size(-1)

    # 4) Python MultiVectorAct (linear, bias=False)
    mva_py = MultiVectorAct(
        channels=channels,
        algebra=algebra,
        input_blades=input_blades,
        agg="linear"
    )
    with torch.no_grad():
        # Randomize conv weight, zero bias
        weight_np = np.random.randn(channels, I_in).astype(np.float32)
        mva_py.conv.weight.copy_(torch.from_numpy(weight_np[:, None, :]))
        if mva_py.conv.bias is not None:
            mva_py.conv.bias.zero_() # we remove the bias - SEE IF FULLY REMOVED 
    out_py = mva_py(x_torch).cpu().detach().numpy()  # (B, C, I_in)

    # 5) C wrapper forward
    ib_np = np.array(input_blades, dtype=np.int32)
    out_c = act_forward(
        x_np,
        weight_np,
        ib_np,   # input_blades
        ib_np,   # kernel_blades == input_blades for "linear"
        I_full
    )

    # 6) Verify
    assert out_c.shape == out_py.shape
    np.testing.assert_allclose(
        out_c, out_py,
        rtol=1e-5, atol=1e-5,
        err_msg=(
            f"Mismatch for batch={batch}, channels={channels}, "
            f"input_blades={input_blades}"
        )
    )
