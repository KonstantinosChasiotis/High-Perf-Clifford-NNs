
import numpy as np
import torch

from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.wrapper_act import act_forward
from cliffordlayers.nn.modules.gcan import MultiVectorAct

###### CODE FROM test_gcan_layers.py #######
def test_multivectoract():
    algebra = CliffordAlgebra([0, 1, 1, 1])
    input = torch.randn(4, 3, 3)
    act = MultiVectorAct(channels=3, algebra=algebra, input_blades=(1, 2, 3))
    output = act(input)
    assert output.shape == (4, 3, 3)
############################################

def test_multivectoract_c_wrapper_shape():

    ### I feel like this test is not really telling anything  :(((

    # pick a 4D signature so we get 16 blades under the hood
    algebra = CliffordAlgebra([0, 1, 1, 1])
    # we only care about the 3 “active” blades in this example
    input_blades = (1, 2, 3)
    B, C, I = 4, 3, len(input_blades)

    # random input of shape (B, C, I)
    x = np.random.randn(B, C, I).astype(np.float32)

    #wrapper expects a weight matrix of shape (C, K)
    #K == number of kernel_blades
    K = len(input_blades)
    weight = np.random.randn(C, K).astype(np.float32)

    # and the array of blade‐indices we’re mixing over
    kernel_blades = np.array(input_blades, dtype=np.int32)

    out_c = act_forward(x, weight, kernel_blades)

    #same (B, C, I)
    assert out_c.shape == (B, C, I)
