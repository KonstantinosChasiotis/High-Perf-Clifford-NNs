#!/usr/bin/env python3
import numpy as np
import torch
from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.nn.modules.gcan import MultiVectorAct
from cliffordlayers.wrapper_act import act_forward

def main():
    # ── Problem dimensions ─────────────────────────
    B, C = 2, 3
    input_blades  = (0, 1, 2, 3)   # I_in = 4
    kernel_blades = (1, 2)         # K     = 2

    I_in = len(input_blades)
    K    = len(kernel_blades)

    # ── Random test data ──────────────────────────
    np.random.seed(0)
    x_np      = np.random.randn(B, C, I_in).astype(np.float32)
    weight_np = np.random.randn(C, K).astype(np.float32)

    # ── Python reference via MultiVectorAct ───────
    algebra = CliffordAlgebra([0,1,1,1])
    # Determine I_full by first embedding:
    tmp = torch.from_numpy(x_np)
    v_full = algebra.embed(tmp, input_blades)        # shape (B,C,I_full)
    I_full = v_full.size(-1)

    mva = MultiVectorAct(
      channels=C,
      algebra=algebra,
      input_blades=input_blades,
      kernel_blades=kernel_blades,
      agg="linear"
    )
    # Disable bias and overwrite weight
    mva.conv.bias = None
    with torch.no_grad():
        mva.conv.weight.copy_(torch.from_numpy(weight_np[:, None, :]))
    out_py = mva(torch.from_numpy(x_np)).detach().cpu().numpy()    # (B,C,I_in)

    # ── C implementation ───────────────────────────
    out_c = act_forward(
      x_np,
      weight_np,
      input_blades,
      kernel_blades,
      I_full
    )

    # ── Compare ────────────────────────────────────
    print("Python out:\n", out_py)
    print(" C    out:\n", out_c)
    np.testing.assert_allclose(
      out_c, out_py, rtol=1e-5, atol=1e-6,
      err_msg="3D MV-activation (embed→gate→get) mismatch"
    )
    print("✅ 3D MV-activation matches perfectly!")

if __name__ == "__main__":
    main()
