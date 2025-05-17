#include "linear_bench.h"
#include "../clifford_linear.h"
#include "../flops.h"
#include <stdlib.h>    
#include <stdint.h> 

void baseline_linear(const CliffordLinear* L, const float* x, int B, float* out) {
    clifford_linear_forward(L, x, B, out);
}

void optimized_linear_v1(const CliffordLinear* L, const float* x, int B, float* out) {
    clifford_linear_forward(L, x, B, out);
}

void optimized_linear_v2(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;
    const int d_flat = C_in * I;      // input‐features per sample
    const int n_flat = C_out * I;     // output‐features per sample

    // 1) Build & cache the Clifford-kernel once
    static float *K       = NULL;
    static int    last_sz = 0;
    int           needed  = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        last_sz = needed;
        // tell the compiler it’s aligned (malloc is ≥16B on most platforms)
        K = (float *)__builtin_assume_aligned(K, 64);
    }

    // 2) Process each sample on a single core
    for (int b = 0; b < B; ++b) {
        //----------------------------------------
        // 2a) Flatten x[b] into a contiguous Xf[]
        //----------------------------------------
        float Xf[d_flat];                // C99 VLA: watch your stack size!
        const float *xb = x + b * (C_in * I);
        for (int i_in = 0; i_in < I; ++i_in) {
            int dst = i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[ ic*I + i_in ];
            }
        }

        //----------------------------------------
        // 2b) For each output-blade & channel, dot-product
        //----------------------------------------
        float *outb = out + b * (C_out * I);
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int   r    = i_out * C_out + oc;   // row in K & bias
                float acc  = 0.0f;
                if (L->bias) {
                    acc = L->bias[r];
                    FLOP(1);
                }
                float *Krow = K + (size_t)r * d_flat;

                // unroll by 4 for better ILP/vectorization
                int d = 0, limit = d_flat - 3;
                for (; d < limit; d += 4) {
                    acc += Krow[d  ] * Xf[d  ]
                         + Krow[d+1] * Xf[d+1]
                         + Krow[d+2] * Xf[d+2]
                         + Krow[d+3] * Xf[d+3];
                    FLOP(8);
                }
                // tail cleanup
                for (; d < d_flat; ++d) {
                    acc += Krow[d] * Xf[d];
                    FLOP(2);
                }

                // write back into exactly the same layout as baseline:
                // out[b*(C_out*I) + oc*I + i_out]
                outb[ oc*I + i_out ] = acc;
            }
        }
    }
}




void register_linear_functions() {
    add_linear_function(baseline_linear, "baseline_linear");
    add_linear_function(optimized_linear_v1, "optimized_v1");
    add_linear_function(optimized_linear_v2, "optimized_v2");
}
