import numpy as np
import torch
import pytest
from torch import nn

from cliffordlayers.nn.modules.gcan import MultiVectorAct
from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.wrapper_sigmoid import c_sigmoid_forward

# @pytest.mark.parametrize("B,C,D", [
#     (1,1,1),    # Minimal case
#     (2,3,4),    # Small case
#     (1,5,2),    # Different dimensions
#     (10,20,30)  # Larger case
# ])
# def test_c_matches_python(B, C, D):
#     # Generate random input
#     x = torch.randn(B, C, D)
    
#     # Run Python version (PyTorch sigmoid)
#     y_py = torch.sigmoid(x).detach().cpu().numpy()
    
#     # Run C version
#     y_c = c_sigmoid_forward(x.detach().cpu().numpy().astype(np.float32))
    
#     # Compare results
#     np.testing.assert_allclose(y_c, y_py, rtol=1e-5, atol=1e-6) 

# def test_multivectoract():
#     algebra = CliffordAlgebra([0, 1, 1, 1])
#     input = torch.randn(4, 3, 3)
#     act = MultiVectorAct(channels=3, algebra=algebra, input_blades=(1, 2, 3))
#     # print("input torch \n", input)
#     output = act(input)
#     # print("output torch \n", output)
#     assert output.shape == (4, 3, 3)

# def test_gcan_linear_forward():
#     algebra = CliffordAlgebra([0, 1, 1, 1])
#     input = torch.randn(4, 3, 3)
#     # print("input c \n", input)
#     output = c_sigmoid_forward(input)
#     # print("output c \n", output)
#     assert output.shape == (4, 3, 3)

def test_gcan_conv_c_vs_py():
    """
    Verify 1D blade convolution without embedding/get functionality
    """
    algebra = CliffordAlgebra([0, 1, 1, 1])
    
    batch = 4
    channels = 3
    n_blades = 4

    input = torch.randn(batch, channels, n_blades)
    
    # Manually set the weight and bias terms
    conv = nn.Conv1d(channels, channels, kernel_size=n_blades, groups=channels)
    conv.weight.data.uniform_(-2,2)
    conv.bias.data.uniform_(-2,2)
    
    # This splits the conv data into # of blades
    #TODO: Add support for kernel blades
    input = input * torch.sigmoid(conv(input))
    
    print("input torch \n", input)

    weights = conv.weight.data.detach().cpu().numpy().astype(np.float32)
    bias = conv.bias.data.detach().cpu().numpy().astype(np.float32)
    
    # Python version
    output_py = conv(input).detach().cpu().numpy()
    
    print("output torch \n", output_py)
    
    # C version
    output_c = c_sigmoid_forward(input.detach().cpu().numpy().astype(np.float32), weights, bias)
    print("output c \n", output_c)
    
    print("=======")
    print(f"weights: {weights}")
    print(f"bias: {bias}")
    
    print("input tensor shape: ", input.shape)
    print("output tensor shape: ", output_py.shape)
    print("output c shape: ", output_c.shape)

    # Compare results
    np.testing.assert_allclose(output_c, output_py, rtol=1e-5, atol=1e-6)

# def test_gcan_c_vs_py():
#     algebra = CliffordAlgebra([0, 1, 1, 1])
    
#     input = torch.randn(4, 3, 3) #Assume it's an embedded vector? No, since multivector act takes unembedded :/

#     act = MultiVectorAct(channels=3, algebra=algebra, input_blades=(1, 2, 3), agg="linear")
    
#     # Manually set the weight and bias terms
#     act.conv.weight.data.uniform_(-2,2)
#     act.conv.bias.data.uniform_(-2,2)

#     print("input tensor shape: ", input.shape)
#     print("input torch \n", input)

#     weights = act.conv.weight
#     bias = act.conv.bias
#     print(f"weights: {weights}")
#     print(f"bias: {bias}")

#     # Python version
#     output_py = act(input).detach().cpu().numpy()
#     print("output tensor shape: ", output_py.shape)
#     print("output torch \n", output_py)
    
#     # C version
#     output_c = c_sigmoid_forward(input.detach().cpu().numpy().astype(np.float32))
#     print("output c shape: ", output_c.shape)
#     print("output c \n", output_c)
#     # Compare results
#     np.testing.assert_allclose(output_c, output_py, rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    test_multivectoract()
    test_gcan_linear_forward()
