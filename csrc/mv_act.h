#ifndef MV_ACT_H
#define MV_ACT_H

#ifdef __cplusplus
extern "C" {
#endif

/// Apply “linear” multi-vector activation:
///  v = embed(input) * sigmoid(conv1d(embed(input)[..., kernel_blades]))
///
/// x:      [B][C][L][I]
/// weight: [C][K]           (depth-wise kernels)
/// kernel_blades: length K, indices into 0..I-1
/// out:    [B][C][L][I]      (may alias x)
void mv_act_forward(
    const float *x,
    int B, int C, int L, int _I,
    const float *weight,
    int K,
    const int *kernel_blades,
    float *out
);

#ifdef __cplusplus
}
#endif
#endif // MV_ACT_H
