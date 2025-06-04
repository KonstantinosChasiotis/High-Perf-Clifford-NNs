#include "linear_bench.h"
#include "../clifford_linear.h"
#include "../flops.h"
#include <stdlib.h>    
#include <stdint.h> 
#include <immintrin.h> // For AVX2 intrinsics
#include <stdio.h>     // For fprintf, and for commented-out debug prints
#include <stdio.h>     // For fprintf in case of errors (though not strictly used in final proposal)
#include <math.h>      // For fmaf

void baseline_linear(const CliffordLinear* L, const float* x, int B, float* out) {
    clifford_linear_forward(L, x, B, out);
}

void optimized_linear_simple(
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

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

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
        // tell the compiler it's aligned (malloc is ≥16B on most platforms)
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
                    // BYTES_READ(sizeof(float));
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
                    // BYTES_READ(sizeof(float) * 8); // 4 from Krow, 4 from Xf
                }
                // tail cleanup
                for (; d < d_flat; ++d) {
                    acc += Krow[d] * Xf[d];
                    FLOP(2);
                    // BYTES_READ(sizeof(float) * 2); // 1 from Krow, 1 from Xf
                }

                // write back into exactly the same layout as baseline:
                // out[b*(C_out*I) + oc*I + i_out]
                outb[ oc*I + i_out ] = acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

// Implwemented a 4-accumulator which leads to signifcant speedup
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

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

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
        // tell the compiler it's aligned (malloc is ≥16B on most platforms)
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
                
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;

                if (L->bias) {
                    acc0 = L->bias[r];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                }
                
                float *Krow = K + (size_t)r * d_flat;

                int d = 0;
                // Calculate limit for the main unrolled loop to handle multiples of 4
                int limit_unrolled = (d_flat / 4) * 4; 
                for (; d < limit_unrolled; d += 4) { // Unroll by 4, using 4 accumulators
                    acc0 += Krow[d+0] * Xf[d+0];
                    acc1 += Krow[d+1] * Xf[d+1];
                    acc2 += Krow[d+2] * Xf[d+2];
                    acc3 += Krow[d+3] * Xf[d+3];
                    FLOP(8); // 4 muls, 4 adds
                    // BYTES_READ(sizeof(float) * 8); // 4 from Krow, 4 from Xf
                }
                
                // Sum the k-accumulators
                float total_acc = acc0 + acc1 + acc2 + acc3;
                FLOP(3); // 3 adds

                // Clean-up remaining elems
                for (; d < d_flat; ++d) {
                    total_acc += Krow[d] * Xf[d];
                    FLOP(2); // 1 mul,1 add
                    // BYTES_READ(sizeof(float) * 2); // 1 from Krow, 1 from Xf
                }

                // write back into exactly the same layout as baseline:
                // out[b*(C_out*I) + oc*I + i_out]
                outb[ oc*I + i_out ] = total_acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

// Added float fma ops from math.h
void optimized_linear_v3(
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

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

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
        K = (float *)__builtin_assume_aligned(K, 64);
    }

    // 2) Process each sample on a single core
    for (int b = 0; b < B; ++b) {
        // 2a) Flatten x[b] into a contiguous Xf[]
        float Xf[d_flat];
        const float *xb = x + b * (C_in * I);
        for (int i_in = 0; i_in < I; ++i_in) {
            int dst = i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[ ic*I + i_in ];
            }
        }

        // 2b) For each output-blade & channel, dot-product
        float *outb = out + b * (C_out * I);
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int   r    = i_out * C_out + oc;
                
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;

                if (L->bias) {
                    acc0 = L->bias[r];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                }
                
                float *Krow = K + (size_t)r * d_flat;

                int d = 0;
                int limit_unrolled = (d_flat / 4) * 4; 
                for (; d < limit_unrolled; d += 4) {
                    //Apply the fma op for the accumulators
                    acc0 = fmaf(Krow[d+0], Xf[d+0], acc0);
                    acc1 = fmaf(Krow[d+1], Xf[d+1], acc1);
                    acc2 = fmaf(Krow[d+2], Xf[d+2], acc2);
                    acc3 = fmaf(Krow[d+3], Xf[d+3], acc3);
                    FLOP(8); // 4 FMAs, so 4 muls and 4adds
                    // BYTES_READ(sizeof(float) * 8); // 4 from Krow, 4 from Xf
                }
                
                float total_acc = acc0 + acc1 + acc2 + acc3;
                FLOP(3);

                for (; d < d_flat; ++d) {
                    total_acc = fmaf(Krow[d], Xf[d], total_acc);
                    FLOP(2); // 1 FMA (i.e. 1 mul & 1 add)
                    // BYTES_READ(sizeof(float) * 2); // 1 from Krow, 1 from Xf
                }

                outb[ oc*I + i_out ] = total_acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

// TODO:  Tried tiling matrix mult, but doesn't lead to much speed-up (probably reuqires tuning, but maybe TA suggests us to use BLAS?)
void optimized_linear_v4(
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

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

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
        K = (float *)__builtin_assume_aligned(K, 64);
    }

    // 2) Process each sample on a single core
    for (int b = 0; b < B; ++b) {
        // 2a) Flatten x[b] into a contiguous Xf[]
        float Xf[d_flat]; // C99 VLA: watch your stack size!
        const float *xb = x + b * (C_in * I);
        for (int i_in = 0; i_in < I; ++i_in) {
            int dst = i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[ ic*I + i_in ];
            }
        }

        // 2b) For each output-blade & channel, dot-product
        float *outb = out + b * (C_out * I);
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int   r    = i_out * C_out + oc;   // row in K & bias
                float *Krow = K + (size_t)r * d_flat;

                float final_acc;
                if (L->bias) {
                    final_acc = L->bias[r];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                } else {
                    final_acc = 0.0f;
                }

                const int D_BLOCK_SIZE = 256; // Trying a tile size of 256

                // Try using blocking to try improve cache perofmance
                for (int d_block_base = 0; d_block_base < d_flat; d_block_base += D_BLOCK_SIZE) {
                    float acc0_tile = 0.0f;
                    float acc1_tile = 0.0f;
                    float acc2_tile = 0.0f;
                    float acc3_tile = 0.0f;

                    int d_block_end = d_block_base + D_BLOCK_SIZE;
                    if (d_block_end > d_flat) {
                        d_block_end = d_flat;
                    }
                    
                    int current_block_num_elems = d_block_end - d_block_base;
                    if (current_block_num_elems <= 0) continue; 

                    int d_tile_idx = 0; 
                    
                    float *Krow_tile_ptr = Krow + d_block_base;
                    float *Xf_tile_ptr   = Xf   + d_block_base;

                    // Keep using the fma + k accumulators like before
                    int limit_unrolled_tile = (current_block_num_elems / 4) * 4;
                    for (; d_tile_idx < limit_unrolled_tile; d_tile_idx += 4) {
                        acc0_tile = fmaf(Krow_tile_ptr[d_tile_idx+0], Xf_tile_ptr[d_tile_idx+0], acc0_tile);
                        acc1_tile = fmaf(Krow_tile_ptr[d_tile_idx+1], Xf_tile_ptr[d_tile_idx+1], acc1_tile);
                        acc2_tile = fmaf(Krow_tile_ptr[d_tile_idx+2], Xf_tile_ptr[d_tile_idx+2], acc2_tile);
                        acc3_tile = fmaf(Krow_tile_ptr[d_tile_idx+3], Xf_tile_ptr[d_tile_idx+3], acc3_tile);
                        FLOP(8);
                        // BYTES_READ(sizeof(float) * 8); // 4 from Krow_tile_ptr, 4 from Xf_tile_ptr
                    }

                    float total_acc_tile = 0.0f;
                    if (limit_unrolled_tile > 0) { 
                        total_acc_tile = acc0_tile + acc1_tile + acc2_tile + acc3_tile;
                        FLOP(3);
                    }
                    
                    // Process whatevers remaining
                    for (; d_tile_idx < current_block_num_elems; ++d_tile_idx) {
                        total_acc_tile = fmaf(Krow_tile_ptr[d_tile_idx], Xf_tile_ptr[d_tile_idx], total_acc_tile);
                        FLOP(2);
                        // BYTES_READ(sizeof(float) * 2); // 1 from Krow_tile_ptr, 1 from Xf_tile_ptr
                    }

                    final_acc += total_acc_tile; //final result
                    FLOP(1); 
                }
                outb[ oc*I + i_out ] = final_acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

// Duplicate of v4 for trying things
void optimized_linear_v5(
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

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

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
        K = (float *)__builtin_assume_aligned(K, 64);
    }

    // 2) Process each sample on a single core
    for (int b = 0; b < B; ++b) {
        // 2a) Flatten x[b] into a contiguous Xf[]
        float Xf[d_flat]; // C99 VLA: watch your stack size!
        const float *xb = x + b * (C_in * I);
        for (int i_in = 0; i_in < I; ++i_in) {
            int dst = i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[ ic*I + i_in ];
            }
        }

        // 2b) For each output-blade & channel, dot-product
        float *outb = out + b * (C_out * I);
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int   r    = i_out * C_out + oc;   // row in K & bias
                float *Krow = K + (size_t)r * d_flat;

                float final_acc;
                if (L->bias) {
                    final_acc = L->bias[r];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                } else {
                    final_acc = 0.0f;
                }

                const int D_BLOCK_SIZE = 256; // Trying a tile size of 256

                for (int d_block_base = 0; d_block_base < d_flat; d_block_base += D_BLOCK_SIZE) {
                    float acc0_tile = 0.0f;
                    float acc1_tile = 0.0f;
                    float acc2_tile = 0.0f;
                    float acc3_tile = 0.0f;

                    int d_block_end = d_block_base + D_BLOCK_SIZE;
                    if (d_block_end > d_flat) {
                        d_block_end = d_flat;
                    }
                    
                    int current_block_num_elems = d_block_end - d_block_base;
                    if (current_block_num_elems <= 0) continue; 

                    int d_tile_idx = 0; 
                    
                    float *Krow_tile_ptr = Krow + d_block_base;
                    float *Xf_tile_ptr   = Xf   + d_block_base;

                    // Using fma with accumulators like before
                    int limit_unrolled_tile = (current_block_num_elems / 4) * 4;
                    for (; d_tile_idx < limit_unrolled_tile; d_tile_idx += 4) {
                        acc0_tile = fmaf(Krow_tile_ptr[d_tile_idx+0], Xf_tile_ptr[d_tile_idx+0], acc0_tile);
                        acc1_tile = fmaf(Krow_tile_ptr[d_tile_idx+1], Xf_tile_ptr[d_tile_idx+1], acc1_tile);
                        acc2_tile = fmaf(Krow_tile_ptr[d_tile_idx+2], Xf_tile_ptr[d_tile_idx+2], acc2_tile);
                        acc3_tile = fmaf(Krow_tile_ptr[d_tile_idx+3], Xf_tile_ptr[d_tile_idx+3], acc3_tile);
                        FLOP(8);
                        // BYTES_READ(sizeof(float) * 8); // 4 from Krow_tile_ptr, 4 from Xf_tile_ptr
                    }

                    float total_acc_tile = 0.0f;
                    if (limit_unrolled_tile > 0) { 
                        total_acc_tile = acc0_tile + acc1_tile + acc2_tile + acc3_tile;
                        FLOP(3); 
                    }
                    
                    // Process whatevers remaining
                    for (; d_tile_idx < current_block_num_elems; ++d_tile_idx) {
                        total_acc_tile = fmaf(Krow_tile_ptr[d_tile_idx], Xf_tile_ptr[d_tile_idx], total_acc_tile);
                        FLOP(2); // 1 FMA (i.e. 1 mul & 1 add)
                    }

                    final_acc += total_acc_tile;
                    FLOP(1); 
                }
                outb[ oc*I + i_out ] = final_acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

// Used to sum horizontally using AVX
static inline float horizontal_sum_m256(__m256 acc_vec) {
    __m128 vlow  = _mm256_castps256_ps128(acc_vec);
    __m128 vhigh = _mm256_extractf128_ps(acc_vec, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(vsum);
}

// Added avx optimizations on top of v3 function
void optimized_linear_v6_avx(
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

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    // 1) Build & cache the Clifford-kernel once
    static float *K       = NULL;
    static int    last_sz = 0;
    int           needed  = d_flat * n_flat;
    if (last_sz != needed) {
        if (K) free(K); // Ensure K is freed only if it was allocated
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        last_sz = needed;
        // The following is a compiler hint. Actual alignment depends on malloc behavior from get_kernel
        // or if _mm_malloc was used inside get_kernel.
        K = (float *)__builtin_assume_aligned(K, 64); 
    }

    // 2) Process each sample on a single core
    for (int b = 0; b < B; ++b) {
        // 2a) Flatten x[b] into a contiguous Xf[]
        // For AVX, Xf should ideally be aligned. Stack VLA alignment is compiler-dependent.
        // Using _mm_loadu_ps for Xf is safer.
        float Xf[d_flat]; 
        const float *xb = x + b * (C_in * I);
        for (int i_in = 0; i_in < I; ++i_in) {
            int dst = i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[ ic*I + i_in ];
            }
        }

        // 2b) For each output-blade & channel, dot-product
        float *outb = out + b * (C_out * I);
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int   r    = i_out * C_out + oc;
                float *Krow = K + (size_t)r * d_flat;
                
                float total_acc;
                if (L->bias) {
                    total_acc = L->bias[r];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                } else {
                    total_acc = 0.0f;
                }

                __m256 vec_acc = _mm256_setzero_ps();
                int d = 0;
                //Process 8 floats at a time (since 256/8 is32 bytes, 1 float is 4 bytes -> 8))
                int limit_vec = (d_flat / 8) * 8;

                for (; d < limit_vec; d += 8) {
                    __m256 k_chunk = _mm256_load_ps(Krow + d); 
                    __m256 x_chunk = _mm256_load_ps(Xf + d);
                    vec_acc = _mm256_fmadd_ps(k_chunk, x_chunk, vec_acc);
                    FLOP(16); // 8 FMAs = 8 muls + 8 adds so 16 flops total right?
                    // BYTES_READ(sizeof(float) * 16); // 8 from Krow, 8 from Xf
                }
                
                // Horzintally sum the floats (is there a more optimal way for this?)
                float sum_scalar_avx = horizontal_sum_m256(vec_acc);
                FLOP(7); // 7 adds
                total_acc += sum_scalar_avx;
                FLOP(1);

                // Handle any remaining elements
                for (; d < d_flat; ++d) {
                    total_acc = fmaf(Krow[d], Xf[d], total_acc);
                    FLOP(2); // 1 FMA (i.e. 2 flops => 1 mul and 1 add)
                    // BYTES_READ(sizeof(float) * 2); // 1 from Krow, 1 from Xf
                }
                outb[ oc*I + i_out ] = total_acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

// Tried using accumulators but across vectors to break dependencies there
void optimized_linear_v7_avx_k_accum(
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

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    // 1) Build & cache the Clifford-kernel once
    static float *K       = NULL;
    static int    last_sz = 0;
    int           needed  = d_flat * n_flat;
    if (last_sz != needed) {
        if (K) free(K); 
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        last_sz = needed;
        K = (float *)__builtin_assume_aligned(K, 64); 
    }

    // 2) Process each sample on a single core
    for (int b = 0; b < B; ++b) {
        // 2a) Flatten x[b] into a contiguous Xf[]
        float Xf[d_flat]; 
        const float *xb = x + b * (C_in * I);
        for (int i_in = 0; i_in < I; ++i_in) {
            int dst = i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[ ic*I + i_in ];
            }
        }

        // 2b) For each output-blade & channel, dot-product
        float *outb = out + b * (C_out * I);
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int   r    = i_out * C_out + oc;
                float *Krow = K + (size_t)r * d_flat;
                
                float total_acc;
                if (L->bias) {
                    total_acc = L->bias[r];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                } else {
                    total_acc = 0.0f;
                }

                // Try vector accumulators
                __m256 vec_acc0 = _mm256_setzero_ps();
                __m256 vec_acc1 = _mm256_setzero_ps();

                int d = 0;
                
                // Process 16 at a time (since 2 * 8 floats )
                int limit_vec_unrolled = (d_flat / 16) * 16; 

                for (; d < limit_vec_unrolled; d += 16) {
                    __m256 k_chunk0 = _mm256_load_ps(Krow + d); 
                    __m256 x_chunk0 = _mm256_load_ps(Xf + d);
                    vec_acc0 = _mm256_fmadd_ps(k_chunk0, x_chunk0, vec_acc0);
                    FLOP(16);
                    // BYTES_READ(sizeof(float) * 16); // 8 from Krow, 8 from Xf

                    __m256 k_chunk1 = _mm256_load_ps(Krow + d + 8); 
                    __m256 x_chunk1 = _mm256_load_ps(Xf + d + 8);
                    vec_acc1 = _mm256_fmadd_ps(k_chunk1, x_chunk1, vec_acc1);
                    FLOP(16);
                    // BYTES_READ(sizeof(float) * 16); // 8 from Krow, 8 from Xf
                }
                
                // Handle remaining AVX vector if d_flat was not a multiple of 16 but >= 8
                if (d < (d_flat / 8) * 8) { // Check if there's at least one more full AVX vector
                     __m256 k_chunk = _mm256_load_ps(Krow + d); 
                     __m256 x_chunk = _mm256_load_ps(Xf + d);
                     vec_acc0 = _mm256_fmadd_ps(k_chunk, x_chunk, vec_acc0); // Add to vec_acc0
                     FLOP(16);
                     // BYTES_READ(sizeof(float) * 16); // 8 from Krow, 8 from Xf
                     d += 8;
                }

                float sum_scalar_avx = 0.0f;
                //check avx was used
                if (limit_vec_unrolled > 0 || d > (d_flat / 16) * 16) {
                    vec_acc0 = _mm256_add_ps(vec_acc0, vec_acc1); // Combine the avx accumulators
                    FLOP(8);
                    sum_scalar_avx = horizontal_sum_m256(vec_acc0);
                    FLOP(7); 
                    
                    total_acc += sum_scalar_avx;
                    FLOP(1); 
                }
                //make sure to process whatevers remaining
                for (; d < d_flat; ++d) {
                    total_acc = fmaf(Krow[d], Xf[d], total_acc);
                    FLOP(2); 
                    // BYTES_READ(sizeof(float) * 2); // 1 from Krow, 1 from Xf
                }

                outb[ oc*I + i_out ] = total_acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

//Try using 4 vector accumulators instead of just 2 (doesn't lead to speed-up)
void optimized_linear_v8_avx_more_unroll( 
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    
    int                    B,
    float                * __restrict out)  
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;
    const int d_flat = C_in * I;      
    const int n_flat = C_out * I;     

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K       = NULL;
    static int    last_sz = 0;
    int           needed  = d_flat * n_flat;
    if (last_sz != needed) {
        if (K) free(K); 
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        last_sz = needed;
        K = (float *)__builtin_assume_aligned(K, 64); 
    }

    for (int b = 0; b < B; ++b) {
        float Xf[d_flat]; 
        const float *xb = x + b * (C_in * I);
        for (int i_in = 0; i_in < I; ++i_in) {
            int dst = i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[ ic*I + i_in ];
            }
        }

        float *outb = out + b * (C_out * I);
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int   r    = i_out * C_out + oc;
                float *Krow = K + (size_t)r * d_flat;
                
                float total_acc;
                if (L->bias) {
                    total_acc = L->bias[r];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                } else {
                    total_acc = 0.0f;
                }

                __m256 vec_acc0 = _mm256_setzero_ps();
                __m256 vec_acc1 = _mm256_setzero_ps();
                __m256 vec_acc2 = _mm256_setzero_ps();
                __m256 vec_acc3 = _mm256_setzero_ps();
                int d = 0;
                int limit_d_total = d_flat;
                int initial_d = d; 
                int limit_unroll32 = (limit_d_total / 32) * 32;
                
                for (; d < limit_unroll32; d += 32) {
                    vec_acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Krow + d +  0), _mm256_loadu_ps(Xf + d +  0), vec_acc0); FLOP(16);
                    vec_acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(Krow + d +  8), _mm256_loadu_ps(Xf + d +  8), vec_acc1); FLOP(16);
                    vec_acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(Krow + d + 16), _mm256_loadu_ps(Xf + d + 16), vec_acc2); FLOP(16);
                    vec_acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(Krow + d + 24), _mm256_loadu_ps(Xf + d + 24), vec_acc3); FLOP(16);
                    // BYTES_READ(sizeof(float) * 64); // 32 from Krow, 32 from Xf
                }

                // Handling the remaining vectors
                int limit_unroll16 = (limit_d_total / 16) * 16;
                if (d < limit_unroll16) {
                    vec_acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Krow + d +  0), _mm256_loadu_ps(Xf + d +  0), vec_acc0); FLOP(16);
                    vec_acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(Krow + d +  8), _mm256_loadu_ps(Xf + d +  8), vec_acc1); FLOP(16);
                    // BYTES_READ(sizeof(float) * 32); // 16 from Krow, 16 from Xf
                    
                    d += 16;
                }
                // if theres 8 remaining, process that
                int limit_unroll8 = (limit_d_total / 8) * 8;
                if (d < limit_unroll8) {
                    vec_acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Krow + d +  0), _mm256_loadu_ps(Xf + d +  0), vec_acc0); FLOP(16);
                    // BYTES_READ(sizeof(float) * 16); // 8 from Krow, 8 from Xf
                    d += 8;
                }
                
                bool avx_ran = (d > initial_d); //(make sure avx was used)
                if (avx_ran) { 
                    vec_acc0 = _mm256_add_ps(vec_acc0, vec_acc1); 
                    FLOP(8); 
                    vec_acc2 = _mm256_add_ps(vec_acc2, vec_acc3); 
                    FLOP(8); 
                    
                    vec_acc0 = _mm256_add_ps(vec_acc0, vec_acc2); 
                    FLOP(8); 
                    float sum_scalar_avx = horizontal_sum_m256(vec_acc0); 
                    FLOP(7); 
                    
                    
                    total_acc += sum_scalar_avx; 
                    FLOP(1); 
                }

                // Process remaining scalars
                for (; d < limit_d_total; ++d) { 
                    total_acc = fmaf(Krow[d], Xf[d], total_acc);
                    FLOP(2); 
                    // BYTES_READ(sizeof(float) * 2); // 1 from Krow, 1 from Xf
                }

                outb[ oc*I + i_out ] = total_acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

//Skip flattening/unflattening, just directly access the data in memory
void optimized_linear_v9_scalar_no_xf(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c_in*I + i_in]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c_out*I + i_out]
{
    const int I_in   = L->sig.n_blades; // Number of input blades (same as output blades, I)
    const int I_out  = L->sig.n_blades; // Number of output blades
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;
    
    const int K_cols = C_in * I_in; 
    const int K_rows = C_out * I_out;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I_in * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I_out * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    // 1) Build & cache the Clifford-kernel once
    static float *K_kernel = NULL;
    static int    last_sz_K = 0;
    int           needed_K  = K_cols * K_rows;

    if (last_sz_K != needed_K) {
        if (K_kernel) free(K_kernel); K_kernel = NULL;
        if (needed_K > 0) {
            int nb_unused;
            L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K_kernel);
            if (K_kernel) K_kernel = (float *)__builtin_assume_aligned(K_kernel, 64);
        }
        last_sz_K = needed_K;
    }

    // 2) Process each sample
    for (int b = 0; b < B; ++b) {
        const float *x_batch_ptr = x + b * (C_in * I_in);
        float       *out_batch_ptr = out + b * (C_out * I_out);

        // Multiply without flattening/unflattening
        // For each output blade
        for (int i_out_blade = 0; i_out_blade < I_out; ++i_out_blade) {
            // For each output channel
            for (int oc_out = 0; oc_out < C_out; ++oc_out) {
                int K_row_idx = i_out_blade * C_out + oc_out;
                float final_acc = 0.0f;
                if (L->bias) {
                    final_acc = L->bias[K_row_idx];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                }

                const float *K_row_ptr = K_kernel + (size_t)K_row_idx * K_cols;

                // Using accumulators
                for (int ic_in = 0; ic_in < C_in; ++ic_in) {
                    float acc_blade_sum0 = 0.0f;
                    float acc_blade_sum1 = 0.0f;
                    float acc_blade_sum2 = 0.0f;
                    float acc_blade_sum3 = 0.0f;
                    
                    const float *x_channel_ptr = x_batch_ptr + ic_in * I_in;
                    int i_in_blade = 0;
                    int limit_blade_unrolled = (I_in / 4) * 4;

                    for (; i_in_blade < limit_blade_unrolled; i_in_blade += 4) {
                        // K access: K_row_ptr[ (i_in_blade+0)*C_in + ic_in ]
                        // X access: x_channel_ptr[ i_in_blade+0 ]
                        acc_blade_sum0 = fmaf(K_row_ptr[(i_in_blade+0)*C_in + ic_in], x_channel_ptr[i_in_blade+0], acc_blade_sum0);
                        acc_blade_sum1 = fmaf(K_row_ptr[(i_in_blade+1)*C_in + ic_in], x_channel_ptr[i_in_blade+1], acc_blade_sum1);
                        acc_blade_sum2 = fmaf(K_row_ptr[(i_in_blade+2)*C_in + ic_in], x_channel_ptr[i_in_blade+2], acc_blade_sum2);
                        acc_blade_sum3 = fmaf(K_row_ptr[(i_in_blade+3)*C_in + ic_in], x_channel_ptr[i_in_blade+3], acc_blade_sum3);
                        FLOP(8); // 4 FMAs, so 4 muls and 4 adds
                        // BYTES_READ(sizeof(float) * 8); // 4 from K_row_ptr, 4 from x_channel_ptr
                    }
                    
                    float total_blade_sum_for_channel = acc_blade_sum0 + acc_blade_sum1 + acc_blade_sum2 + acc_blade_sum3;
                    FLOP(3); //summing the 4 accs

                    for (; i_in_blade < I_in; ++i_in_blade) {
                        total_blade_sum_for_channel = fmaf(K_row_ptr[i_in_blade*C_in + ic_in], x_channel_ptr[i_in_blade], total_blade_sum_for_channel);
                        FLOP(2); // 1 fma
                        // BYTES_READ(sizeof(float) * 2); // 1 from K_row_ptr, 1 from x_channel_ptr
                    }
                    final_acc += total_blade_sum_for_channel;
                    FLOP(1);
                }
                
                // Write to output: out[b*(C_out*I_out) + oc_out*I_out + i_out_blade]
                out_batch_ptr[oc_out * I_out + i_out_blade] = final_acc;
                BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

//Vectorize the v10 (no flatting) implementation
void optimized_linear_v10_avx_no_xf(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c_in*I + i_in]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c_out*I + i_out]
{
    const int I_in   = L->sig.n_blades; 
    const int I_out  = L->sig.n_blades; 
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;
    
    const int K_cols = C_in * I_in; 
    const int K_rows = C_out * I_out;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I_in * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I_out * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K_kernel = NULL;
    static int    last_sz_K = 0;
    int           needed_K  = K_cols * K_rows;

    if (last_sz_K != needed_K) {
        if (K_kernel) free(K_kernel); K_kernel = NULL;
        if (needed_K > 0) {
            int nb_unused;
            L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K_kernel);
            if (K_kernel) K_kernel = (float *)__builtin_assume_aligned(K_kernel, 64);
        }
        last_sz_K = needed_K;
    }

    if (needed_K > 0 && !K_kernel) { 
        for(int i=0; i < B * C_out * I_out; ++i) out[i] = 0.0f; 
        return;
    }

    for (int b = 0; b < B; ++b) {
        const float *x_batch_ptr = x + b * (C_in * I_in);
        float       *out_batch_ptr = out + b * (C_out * I_out);

        for (int i_out_blade = 0; i_out_blade < I_out; ++i_out_blade) {
            for (int oc_out = 0; oc_out < C_out; ++oc_out) {
                int K_row_idx = i_out_blade * C_out + oc_out;
                float final_acc = 0.0f;
                if (L->bias) {
                    final_acc = L->bias[K_row_idx];
                    FLOP(1);
                    // BYTES_READ(sizeof(float));
                }
                const float *K_row_ptr = K_kernel + (size_t)K_row_idx * K_cols;

                for (int ic_in = 0; ic_in < C_in; ++ic_in) {
                    float total_blade_sum_for_channel = 0.0f;
                    const float *x_channel_ptr = x_batch_ptr + ic_in * I_in;
                    
                    __m256 vec_acc_blades = _mm256_setzero_ps();
                    int i_in_blade = 0;
                    int limit_blade_avx = (I_in / 8) * 8;

                    for (; i_in_blade < limit_blade_avx; i_in_blade += 8) {
                        __m256 x_vec = _mm256_loadu_ps(x_channel_ptr + i_in_blade);
                        
                        __m256i k_indices_vec = _mm256_set_epi32(
                            (i_in_blade+7)*C_in + ic_in, (i_in_blade+6)*C_in + ic_in,
                            (i_in_blade+5)*C_in + ic_in, (i_in_blade+4)*C_in + ic_in,
                            (i_in_blade+3)*C_in + ic_in, (i_in_blade+2)*C_in + ic_in,
                            (i_in_blade+1)*C_in + ic_in, (i_in_blade+0)*C_in + ic_in
                        );                        
                        __m256 k_vec = _mm256_i32gather_ps(K_row_ptr, k_indices_vec, 1);
                        vec_acc_blades = _mm256_fmadd_ps(k_vec, x_vec, vec_acc_blades);
                        FLOP(16);
                        // BYTES_READ(sizeof(float) * 16); // 8 from K_row_ptr (via gather), 8 from x_channel_ptr
                    }
                    
                    //horizontally sum across the acc blades
                    total_blade_sum_for_channel = horizontal_sum_m256(vec_acc_blades);
                    FLOP(7);

                    // handle remaininig values
                    for (; i_in_blade < I_in; ++i_in_blade) {
                        total_blade_sum_for_channel = fmaf(K_row_ptr[i_in_blade*C_in + ic_in], x_channel_ptr[i_in_blade], total_blade_sum_for_channel);
                        FLOP(2); 
                        // BYTES_READ(sizeof(float) * 2); // 1 from K_row_ptr, 1 from x_channel_ptr
                    }

                    final_acc += total_blade_sum_for_channel;
                    FLOP(1);
                }
                out_batch_ptr[oc_out * I_out + i_out_blade] = final_acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
}

void register_linear_functions() {
    add_linear_function(baseline_linear, "baseline_linear");
    add_linear_function(optimized_linear_simple, "optimized_simple");
    add_linear_function(optimized_linear_v2, "optimized_v2_k_accum");
    add_linear_function(optimized_linear_v3, "optimized_v3_fma_k_accum");
    add_linear_function(optimized_linear_v4, "optimized_v4_tiled_fma_k_accum");
    
    add_linear_function(optimized_linear_v5, "optimized_v5_scalar_peak");
    add_linear_function(optimized_linear_v6_avx, "optimized_v6_avx");
    add_linear_function(optimized_linear_v7_avx_k_accum, "optimized_v7_avx_k_accum");
    add_linear_function(optimized_linear_v8_avx_more_unroll, "optimized_v8_avx_more_unroll");
    add_linear_function(optimized_linear_v9_scalar_no_xf, "optimized_v9_scalar_no_xf");
    add_linear_function(optimized_linear_v10_avx_no_xf, "optimized_v10_avx_no_xf");
}
