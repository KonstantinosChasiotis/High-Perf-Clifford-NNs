#include <stdlib.h>
#include <math.h>
#include "mv_act.h"

// x:      [B][C][L][I]
// weight: [C][K]
// kernel_blades: length K, indices into the blade‐dimension I
// out:    [B][C][L][I]  (in-place overwrite is fine)
void mv_act_forward(
    const float *x,
    int B, int C, int L, int I,
    const float *weight,
    int K,
    const int *kernel_blades,
    float *out
) {
    // for each sample, channel, position, blade
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int p = 0; p < L; ++p) {
                // compute the 1D depthwise convolution at (b,c,p)
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int blade = kernel_blades[k];
                    // x[b,c,p+k,blade]
                    float v = x[
                        ((b * C + c) * L + (p + k)) * I
                        + blade
                    ];
                    acc += weight[c*K + k] * v;
                }
                // sigmoid activation
                float sig = 1.0f / (1.0f + expf(-acc));

                
                // now scale all blades at (b,c,p,*)
                for (int i = 0; i < I; ++i) {
                    float v = x[
                        ((b * C + c) * L + p) * I
                        + i
                    ];
                    out[
                        ((b * C + c) * L + p) * I
                        + i
                    ] = v * sig;
                }
            }
        }
    }
}

