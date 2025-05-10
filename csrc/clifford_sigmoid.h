#ifndef CLIFFORD_SIGMOID_H
#define CLIFFORD_SIGMOID_H

#include <stdlib.h>
#include <stdbool.h>

/// Applies sigmoid activation to Clifford multivectors
/// 
/// @param x Input tensor of shape [B][C][D][blade] where:
///          B = batch size
///          C = number of channels
///          D = spatial dimension
///          blade = number of blades in the Clifford algebra
/// @param out Output tensor of same shape as x
/// @param B Batch size
/// @param C Number of channels
/// @param D Spatial dimension
/// @param n_blades Number of blades in the Clifford algebra
/// @param weight Weight tensor for the convolution
/// @param bias Bias tensor
void clifford_sigmoid_forward(
    const float *x,
    float *out,
    int B,
    int C,
    int D,
    int n_blades,
    const float *weight,
    const float *bias);

#endif // CLIFFORD_SIGMOID_H 