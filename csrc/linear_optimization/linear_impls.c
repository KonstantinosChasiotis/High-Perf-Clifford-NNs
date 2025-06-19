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

void optimized_no_kernel_v0(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    // TODO: Will only work for 1D Clifford algebra (I=2)
    const float g0 = L->sig.g[0];

    for (int b = 0; b < B; ++b) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            float b0_out = 0.0f;
            float b1_out = 0.0f;
        if (L->bias) {
                b0_out = L->bias[c_out * I + 0];
                b1_out = L->bias[c_out * I + 1];
            }

            for (int c_in = 0; c_in < C_in; ++c_in) {
                // Weight layout: (I, C_out, C_in)
                const float w0 = L->weight[(0 * C_out + c_out) * C_in + c_in];
                const float w1 = L->weight[(1 * C_out + c_out) * C_in + c_in];

                // Input layout: (B, C_in, I)
                const float *x_ptr = x + b * (C_in * I) + c_in * I;
                const float x0 = x_ptr[0];
                const float x1 = x_ptr[1];
                
                // Clifford product, accumulate over input channels
                b0_out += w0 * x0 + g0 * w1 * x1;
                b1_out += w1 * x0 + w0 * x1;
            }

            // Output layout: (B, C_out, I)
            float *out_ptr = out + b * (C_out * I) + c_out * I;
            out_ptr[0] = b0_out;
            out_ptr[1] = b1_out;
        }
    }
}

void optimized_linear_basic(
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

    // 1) Build the Clifford-kernel fresh every time
        int nb_unused;
    float *K;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        // tell the compiler it's aligned (malloc is ≥16B on most platforms)
        K = (float *)__builtin_assume_aligned(K, 64);

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

                // Implement -1, 0, 1 based on g value (i.e. no kernel)
                // Don't malloc the kernel, use static registers or direct?
                for (int d = 0; d < C_in * I; d++){
                    acc += Krow[d] * Xf[d];
                    FLOP(2);
                }

                // write back into exactly the same layout as baseline:
                // out[b*(C_out*I) + oc*I + i_out]
                outb[ oc*I + i_out ] = acc;
                // BYTES_WRITTEN(sizeof(float));
            }
        }
    }
    
    // Clean up kernel memory
    free(K);
}

// Sufficient to implement for just one kernel
// void optimized_linear_1d_kernel_broken(
//     const CliffordLinear * __restrict L,
//     const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
//     int                    B,
//     float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
// {
//     const int I      = L->sig.n_blades;
//     const int C_in   = L->in_channels;
//     const int C_out  = L->out_channels;
//     const int d_flat = C_in * I;      // input‐features per sample
//     const int n_flat = C_out * I;     // output‐features per sample

//     // Tmp variables
//     float* weights[C_out][C_in][I];
    
//     int* weights_view[I][I] = {
//         {0, 1},
//         {1, 0}
//     };

//     int* g_view[I] = {
//         {1, L->sig.g[0]},
//         {1, 1}
//     };


//     float* tmp_accum[I];

//     if (I != 2){
//         printf("This implementation is only valid for #blades = 2\n");
//         return;
//     }

//     //Re-order the weights from [blade][out_ch][in_ch] into [out_ch][in_ch][blades]
//     for (int out_ch = 0; out_ch < C_out; ++out_ch){
//         for (int in_ch = 0; in_ch < C_in; ++in_ch){
//             for (int blade = 0; blade < I; ++blade){
//                 weights[out_ch][in_ch][blade] = L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
//             }
//         }
//     }

//     // Contiguous output access
//     for (int b = 0; b < B; ++b){
//         for (int out_ch = 0; out_ch < C_out; ++out_ch){
//             for (int b = 0; b < I; ++b){
//                 tmp_accum[b] = 0.0f; //Reset for each new blade

//                 for (int in_ch = 0; in_ch < C_in; ++in_ch){                
//                     for (int index_b = 0; index_b < I; ++index_b){
//                         float w = weights[out_ch][in_ch][weights_view[b][index_b]];
//                         float x = x[in_ch * I + index_b];

//                         tmp_accum[index_b] += w * x;
//                     }
//                 }

//                 for (int index_b = 0; index_b < I; ++index_b){
//                     float g = g_view[b][index_b];
//                     float acc = tmp_accum[index_b];

//                     out[out_ch * I + b] += acc * g;
//                 }
//             }
//         }
//     }
// }

void optimized_linear_1d_kernel(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    // weights_view[output_blade][input_blade] = which weight index to use
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    // g_view[output_blade][input_blade] = which g coefficient to apply
    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                float accumulator = 0.0f;

                // Add bias if present
                if (L->bias) {
                    accumulator = L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Accumulate across all input channels and input blades
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        // Get the appropriate weight using lookup table
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = L->weight[weight_idx * C_out * C_in + out_ch * C_in + in_ch];
                        
                        // Get input value: x[batch * (C_in * I) + in_ch * I + in_blade]
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Get the appropriate g coefficient using lookup table
                        float g_coeff = g_view[out_blade][in_blade];
                        
                        // Accumulate: weight * input * g_coefficient
                        accumulator += weight * input_val * g_coeff;
                        FLOP(3); // 2 multiplies + 1 add
                    }
                }

                // Store result: out[batch * (C_out * I) + out_ch * I + out_blade]
                out[batch * (C_out * I) + out_ch * I + out_blade] = accumulator;
            }
        }
    }
}

void optimized_linear_1d_kernel_reordered_weights(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
                if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    // This gives us unit stride access when iterating through blades
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int out_ch = 0; out_ch < C_out; ++out_ch) {
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            for (int blade = 0; blade < I; ++blade) {
                // Original: L->weight[blade * C_out * C_in + out_ch * C_in + in_ch]
                // New:      weights_reorg[out_ch * C_in * I + in_ch * I + blade]
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                float accumulator = 0.0f;

                // Add bias if present
                if (L->bias) {
                    accumulator = L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Accumulate across all input channels and input blades
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    // Get base pointer for this out_ch/in_ch pair - unit stride access from here
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    const float *input_slice = x + batch * (C_in * I) + in_ch * I;
                    
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        // Get the appropriate weight using lookup table - now unit stride!
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                        
                        // Get input value - also unit stride
                        float input_val = input_slice[in_blade];
                        
                        // Get the appropriate g coefficient using lookup table
                        float g_coeff = g_view[out_blade][in_blade];
                        
                        // Accumulate: weight * input * g_coefficient
                        accumulator += weight * input_val * g_coeff;
                        FLOP(3); // 2 multiplies + 1 add
                    }
                }

                // Store result: out[batch * (C_out * I) + out_ch * I + out_blade]
                out[batch * (C_out * I) + out_ch * I + out_blade] = accumulator;
            }
        }
    }

    free(weights_reorg);
}

void optimized_linear_1d_kernel_reordered_weights_factored_g_accum(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int out_ch = 0; out_ch < C_out; ++out_ch) {
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            for (int blade = 0; blade < I; ++blade) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                
                // Accumulator per input blade - accumulate weight*input without g_coeff
                float blade_accumulator[I]; // For I=2, this is just 2 floats
                for (int i = 0; i < I; ++i) {
                    blade_accumulator[i] = 0.0f;
                }

                // Accumulate across all input channels (without g_coeff)
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    const float *input_slice = x + batch * (C_in * I) + in_ch * I;
                    
                    // Hottest loop - no g_coeff lookup here!
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                        float input_val = input_slice[in_blade];
                        
                        // Just accumulate weight * input (no g_coeff multiplication)
                        blade_accumulator[in_blade] += weight * input_val;
                        FLOP(2); // 1 multiply + 1 add
                    }
                }

                // Now apply g_coeff and reduce to final result
                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    float g_coeff = g_view[out_blade][in_blade];
                    final_accumulator += blade_accumulator[in_blade] * g_coeff;
                    FLOP(2); // 1 multiply + 1 add
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_1d_kernel_reordered_weights_factored_g_accum_fma(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int out_ch = 0; out_ch < C_out; ++out_ch) {
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            for (int blade = 0; blade < I; ++blade) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                
                // Accumulator per input blade - accumulate weight*input without g_coeff
                float blade_accumulator[I]; // For I=2, this is just 2 floats
                for (int i = 0; i < I; ++i) {
                    blade_accumulator[i] = 0.0f;
                }

                // Accumulate across all input channels (without g_coeff)
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    const float *input_slice = x + batch * (C_in * I) + in_ch * I;
                    
                    // Hottest loop - no g_coeff lookup here! - OPTIMIZED WITH FMAF
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                        float input_val = input_slice[in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    float g_coeff = g_view[out_blade][in_blade];
                    // Use fmaf for fused multiply-add operation
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_coeff, final_accumulator);
                    FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_1d_kernel_reordered_source_weights_factored_g_accum_fma(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                
                // Accumulator per input blade - accumulate weight*input without g_coeff
                float blade_accumulator[I]; // For I=2, this is just 2 floats
                for (int i = 0; i < I; ++i) {
                    blade_accumulator[i] = 0.0f;
                }

                // Accumulate across all input channels (without g_coeff)
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    const float *input_slice = x + batch * (C_in * I) + in_ch * I;
                    
                    // Hottest loop - no g_coeff lookup here! - OPTIMIZED WITH FMAF
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                        float input_val = input_slice[in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    float g_coeff = g_view[out_blade][in_blade];
                    // Use fmaf for fused multiply-add operation
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_coeff, final_accumulator);
                    FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    float blade_accumulator[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {

                // Accumulate across all input channels (without g_coeff)
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    // Likely heavily-strided weight matrix cache accesses
                    int weight_idx = weights_view[out_blade][in_blade];
                    
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                        float weight = weight_slice[weight_idx];
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    float g_coeff = g_view[out_blade][in_blade];
                    // Use fmaf for fused multiply-add operation
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_coeff, final_accumulator);
                    blade_accumulator[in_blade] = 0.0f;
                    FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

// Likely bad striding of in_channel for each input leading to far more cycles than one-time input/output flattening striding.
void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use

    // Given the matrix is symmetric, we can load in the weight once and multiply by the weights_view inverse? Some form of unrolling?
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // float blade_accumulator[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {

                float blade_acc0 = 0.0f, blade_acc1 = 0.0f, blade_acc2 = 0.0f, blade_acc3 = 0.0f;
                float blade_acc4 = 0.0f, blade_acc5 = 0.0f, blade_acc6 = 0.0f, blade_acc7 = 0.0f;

                // Accumulate across all input channels (without g_coeff)
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    // Likely heavily-strided weight matrix cache accesses
                    int weight_idx = weights_view[out_blade][in_blade];
                    
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                        float weight = weight_slice[weight_idx];
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        switch (in_blade) {
                            case 0: blade_acc0 = fmaf(weight, input_val, blade_acc0); break;
                            case 1: blade_acc1 = fmaf(weight, input_val, blade_acc1); break;
                            case 2: blade_acc2 = fmaf(weight, input_val, blade_acc2); break;
                            case 3: blade_acc3 = fmaf(weight, input_val, blade_acc3); break;
                            case 4: blade_acc4 = fmaf(weight, input_val, blade_acc4); break;
                            case 5: blade_acc5 = fmaf(weight, input_val, blade_acc5); break;
                            case 6: blade_acc6 = fmaf(weight, input_val, blade_acc6); break;
                            case 7: blade_acc7 = fmaf(weight, input_val, blade_acc7); break;
                        }
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator_1 = 0.0f;
                float final_accumulator_2 = 0.0f;
                float final_accumulator_3 = 0.0f;
                float final_accumulator_4 = 0.0f;
                final_accumulator_1 = fmaf(blade_acc0, g_view[out_blade][0], final_accumulator_1);
                final_accumulator_2 = fmaf(blade_acc1, g_view[out_blade][1], final_accumulator_2);
                final_accumulator_3 = fmaf(blade_acc2, g_view[out_blade][2], final_accumulator_3);
                final_accumulator_4 = fmaf(blade_acc3, g_view[out_blade][3], final_accumulator_4);
                final_accumulator_1 = fmaf(blade_acc4, g_view[out_blade][4], final_accumulator_1);
                final_accumulator_2 = fmaf(blade_acc5, g_view[out_blade][5], final_accumulator_2);
                final_accumulator_3 = fmaf(blade_acc6, g_view[out_blade][6], final_accumulator_3);
                final_accumulator_4 = fmaf(blade_acc7, g_view[out_blade][7], final_accumulator_4);

                float final_accumulator = final_accumulator_1 + final_accumulator_2 + final_accumulator_3 + final_accumulator_4;
                FLOP(16); // 8 FMAs, so 8 multiplies + 8 adds

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

// Likely bad striding of in_channel for each input leading to far more cycles than one-time input/output flattening striding.
void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use

    // Given the matrix is symmetric, we can load in the weight once and multiply by the weights_view inverse? Some form of unrolling?
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // float blade_accumulator[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {

                float blade_acc0 = 0.0f, blade_acc1 = 0.0f, blade_acc2 = 0.0f, blade_acc3 = 0.0f;
                float blade_acc4 = 0.0f, blade_acc5 = 0.0f, blade_acc6 = 0.0f, blade_acc7 = 0.0f;

                // Accumulate across all input channels (without g_coeff)
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        // Likely heavily-strided weight matrix cache accesses
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                    
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        switch (in_blade) {
                            case 0: blade_acc0 = fmaf(weight, input_val, blade_acc0); break;
                            case 1: blade_acc1 = fmaf(weight, input_val, blade_acc1); break;
                            case 2: blade_acc2 = fmaf(weight, input_val, blade_acc2); break;
                            case 3: blade_acc3 = fmaf(weight, input_val, blade_acc3); break;
                            case 4: blade_acc4 = fmaf(weight, input_val, blade_acc4); break;
                            case 5: blade_acc5 = fmaf(weight, input_val, blade_acc5); break;
                            case 6: blade_acc6 = fmaf(weight, input_val, blade_acc6); break;
                            case 7: blade_acc7 = fmaf(weight, input_val, blade_acc7); break;
                        }
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator_1 = 0.0f;
                float final_accumulator_2 = 0.0f;
                float final_accumulator_3 = 0.0f;
                float final_accumulator_4 = 0.0f;
                final_accumulator_1 = fmaf(blade_acc0, g_view[out_blade][0], final_accumulator_1);
                final_accumulator_2 = fmaf(blade_acc1, g_view[out_blade][1], final_accumulator_2);
                final_accumulator_3 = fmaf(blade_acc2, g_view[out_blade][2], final_accumulator_3);
                final_accumulator_4 = fmaf(blade_acc3, g_view[out_blade][3], final_accumulator_4);
                final_accumulator_1 = fmaf(blade_acc4, g_view[out_blade][4], final_accumulator_1);
                final_accumulator_2 = fmaf(blade_acc5, g_view[out_blade][5], final_accumulator_2);
                final_accumulator_3 = fmaf(blade_acc6, g_view[out_blade][6], final_accumulator_3);
                final_accumulator_4 = fmaf(blade_acc7, g_view[out_blade][7], final_accumulator_4);

                float final_accumulator = final_accumulator_1 + final_accumulator_2 + final_accumulator_3 + final_accumulator_4;
                FLOP(16); // 8 FMAs, so 8 multiplies + 8 adds

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

// Likely bad striding of in_channel for each input leading to far more cycles than one-time input/output flattening striding.
void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use

    // Given the matrix is symmetric, we can load in the weight once and multiply by the weights_view inverse? Some form of unrolling?
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    float blade_accumulator[8] = {0}; // For I=8, this is 8 floats
    float blade_accumulator_inverse[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            int half_I = I / 2;
            for (int out_blade = 0; out_blade < half_I; ++out_blade) {

                float blade_acc0 = 0.0f, blade_acc1 = 0.0f, blade_acc2 = 0.0f, blade_acc3 = 0.0f;
                float blade_acc4 = 0.0f, blade_acc5 = 0.0f, blade_acc6 = 0.0f, blade_acc7 = 0.0f;

                // Accumulate across all input channels (without g_coeff)
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    int weight_idx = weights_view[out_blade][in_blade];
                    int weight_idx_inverse = weights_view[out_blade][I - in_blade - 1];
                    
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                        
                        float weight = weight_slice[weight_idx];
                        float weight_inverse = weight_slice[weight_idx_inverse];
                    
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        blade_accumulator_inverse[in_blade] = fmaf(weight_inverse, input_val, blade_accumulator_inverse[in_blade]);

                        FLOP(4); // 2 FMAs, so 2 multiplies + 2 adds
                    }
                }


                // TODO: Fully-unroll this based on the in_blade and thus multiply only with g (otherwise just return or negate!)
                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator = 0.0f;
                float final_accumulator_inverse = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_view[out_blade][in_blade], final_accumulator);
                    final_accumulator_inverse = fmaf(blade_accumulator_inverse[in_blade], g_view[I - out_blade - 1][in_blade], final_accumulator_inverse);

                    blade_accumulator[in_blade] = 0.0f;
                    blade_accumulator_inverse[in_blade] = 0.0f;
                    FLOP(4); // 2 FMAs, so 2 multiplies + 2 adds
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    final_accumulator_inverse += L->bias[(I - out_blade - 1) * C_out + out_ch];
                    FLOP(2);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
                out[batch * (C_out * I) + out_ch * I + (I - out_blade - 1)] = final_accumulator_inverse;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse_unrolled_g(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use

    // Given the matrix is symmetric, we can load in the weight once and multiply by the weights_view inverse? Some form of unrolling?
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    // float g_view[8][8] = {
    //     {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
    //     {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
    //     {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
    //     {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
    //     {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
    //     {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
    //     {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
    //     {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    // };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    float blade_accumulator[8] = {0}; // For I=8, this is 8 floats
    float blade_accumulator_inverse[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            int half_I = I / 2;
            for (int out_blade = 0; out_blade < half_I; ++out_blade) {

                float blade_acc0 = 0.0f, blade_acc1 = 0.0f, blade_acc2 = 0.0f, blade_acc3 = 0.0f;
                float blade_acc4 = 0.0f, blade_acc5 = 0.0f, blade_acc6 = 0.0f, blade_acc7 = 0.0f;

                // Accumulate across all input channels (without g_coeff)
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    int weight_idx = weights_view[out_blade][in_blade];
                    int weight_idx_inverse = weights_view[out_blade][I - in_blade - 1];
                    
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                        
                        float weight = weight_slice[weight_idx];
                        float weight_inverse = weight_slice[weight_idx_inverse];
                    
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        blade_accumulator_inverse[in_blade] = fmaf(weight_inverse, input_val, blade_accumulator_inverse[in_blade]);

                        FLOP(4); // 2 FMAs, so 2 multiplies + 2 adds
                    }
                }

                // Replace the TODO section with explicit g-coefficient multiplications
                // Now apply g_coeff and reduce to final result - FULLY UNROLLED AND OPTIMIZED
                float final_accumulator = 0.0f;
                float final_accumulator_inverse = 0.0f;

                switch (out_blade) {
                    case 0: // blade 0: {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2}
                            // blade 7: {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f}
                        final_accumulator = blade_accumulator[0] +                          // 1.0f
                                        blade_accumulator[1] * g0 +                      // g0
                                        blade_accumulator[2] * g1 +                      // g1
                                        blade_accumulator[3] * g2 +                      // g2
                                        blade_accumulator[4] * (-g0g1) +                 // -g0g1
                                        blade_accumulator[5] * (-g0g2) +                 // -g0g2
                                        blade_accumulator[6] * (-g1g2) +                 // -g1g2
                                        blade_accumulator[7] * (-g0g1g2);                // -g0g1g2
                        
                        final_accumulator_inverse = blade_accumulator_inverse[0] +          // 1.0f
                                                blade_accumulator_inverse[1] +           // 1.0f (no multiply!)
                                                (-blade_accumulator_inverse[2]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[3] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[4] +           // 1.0f (no multiply!)
                                                (-blade_accumulator_inverse[5]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[6] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[7];            // 1.0f (no multiply!)
                        FLOP(6 + 0); // 6 multiplies for normal, 0 for inverse (only adds/negates)
                        break;

                    case 1: // blade 1: {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2}
                            // blade 6: {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0}
                        final_accumulator = blade_accumulator[0] +                          // 1.0f (no multiply!)
                                        blade_accumulator[1] +                           // 1.0f (no multiply!)
                                        blade_accumulator[2] * (-g1) +                   // -g1
                                        blade_accumulator[3] * (-g2) +                   // -g2
                                        blade_accumulator[4] * g1 +                      // g1
                                        blade_accumulator[5] * g2 +                      // g2
                                        blade_accumulator[6] * (-g1g2) +                 // -g1g2
                                        blade_accumulator[7] * (-g1g2);                  // -g1g2
                        
                        final_accumulator_inverse = blade_accumulator_inverse[0] +          // 1.0f (no multiply!)
                                                blade_accumulator_inverse[1] * g0 +      // g0
                                                blade_accumulator_inverse[2] +           // 1.0f (no multiply!)
                                                (-blade_accumulator_inverse[3]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[4] * (-g0) +   // -g0
                                                blade_accumulator_inverse[5] * g0 +      // g0
                                                blade_accumulator_inverse[6] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[7] * g0;       // g0
                        FLOP(6 + 4); // 6 multiplies for normal, 4 for inverse
                        break;

                    case 2: // blade 2: {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2}
                            // blade 5: {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1}
                        final_accumulator = blade_accumulator[0] +                          // 1.0f (no multiply!)
                                        blade_accumulator[1] * g0 +                      // g0
                                        blade_accumulator[2] +                           // 1.0f (no multiply!)
                                        blade_accumulator[3] * (-g2) +                   // -g2
                                        blade_accumulator[4] * (-g0) +                   // -g0
                                        blade_accumulator[5] * g0g2 +                    // g0g2
                                        blade_accumulator[6] * g2 +                      // g2
                                        blade_accumulator[7] * g0g2;                     // g0g2
                        
                        final_accumulator_inverse = blade_accumulator_inverse[0] +          // 1.0f (no multiply!)
                                                blade_accumulator_inverse[1] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[2] * (-g1) +   // -g1
                                                (-blade_accumulator_inverse[3]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[4] * g1 +      // g1
                                                blade_accumulator_inverse[5] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[6] * (-g1) +   // -g1
                                                blade_accumulator_inverse[7] * (-g1);    // -g1
                        FLOP(6 + 4); // 6 multiplies for normal, 4 for inverse
                        break;

                    case 3: // blade 3: {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1}
                            // blade 4: {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2}
                        final_accumulator = blade_accumulator[0] +                          // 1.0f (no multiply!)
                                        blade_accumulator[1] * g0 +                      // g0
                                        blade_accumulator[2] * g1 +                      // g1
                                        blade_accumulator[3] +                           // 1.0f (no multiply!)
                                        blade_accumulator[4] * (-g0g1) +                 // -g0g1
                                        blade_accumulator[5] * (-g0) +                   // -g0
                                        blade_accumulator[6] * (-g1) +                   // -g1
                                        blade_accumulator[7] * (-g0g1);                  // -g0g1
                        
                        final_accumulator_inverse = blade_accumulator_inverse[0] +          // 1.0f (no multiply!)
                                                blade_accumulator_inverse[1] +           // 1.0f (no multiply!)
                                                (-blade_accumulator_inverse[2]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[3] * g2 +      // g2
                                                blade_accumulator_inverse[4] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[5] * (-g2) +   // -g2
                                                blade_accumulator_inverse[6] * g2 +      // g2
                                                blade_accumulator_inverse[7] * g2;       // g2
                        FLOP(6 + 4); // 6 multiplies for normal, 4 for inverse
                        break;
                }

                // Clear accumulators for next iteration
                for (int i = 0; i < I; ++i) {
                    blade_accumulator[i] = 0.0f;
                    blade_accumulator_inverse[i] = 0.0f;
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    final_accumulator_inverse += L->bias[(I - out_blade - 1) * C_out + out_ch];
                    FLOP(2);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
                out[batch * (C_out * I) + out_ch * I + (I - out_blade - 1)] = final_accumulator_inverse;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
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

    // 1) Build the Clifford-kernel fresh every time
        int nb_unused;
    float *K;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        K = (float *)__builtin_assume_aligned(K, 64);

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
    
    // Clean up kernel memory
    free(K);
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

    // 1) Build the Clifford-kernel fresh every time
        int nb_unused;
    float *K;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        K = (float *)__builtin_assume_aligned(K, 64);

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
    
    // Clean up kernel memory
    free(K);
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

    // 1) Build the Clifford-kernel fresh every time
        int nb_unused;
    float *K;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        K = (float *)__builtin_assume_aligned(K, 64);

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
    
    // Clean up kernel memory
    free(K);
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

    // 1) Build the Clifford-kernel fresh every time
        int nb_unused;
    float *K;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        // The following is a compiler hint. Actual alignment depends on malloc behavior from get_kernel
        // or if _mm_malloc was used inside get_kernel.
        K = (float *)__builtin_assume_aligned(K, 64); 

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
    
    // Clean up kernel memory
    free(K);
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

    // 1) Build the Clifford-kernel fresh every time
        int nb_unused;
    float *K;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        K = (float *)__builtin_assume_aligned(K, 64); 

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
    
    // Clean up kernel memory
    free(K);
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

    // Build the Clifford-kernel fresh every time
        int nb_unused;
    float *K;
        L->get_kernel(L->weight, L->sig.g,
                      C_in, C_out,
                      &nb_unused, &K);
        K = (float *)__builtin_assume_aligned(K, 64); 

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
    
    // Clean up kernel memory
    free(K);
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

    // 1) Build the Clifford-kernel fresh every time
    float *K_kernel = NULL;
    int needed_K = K_cols * K_rows;
        if (needed_K > 0) {
            int nb_unused;
            L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K_kernel);
            if (K_kernel) K_kernel = (float *)__builtin_assume_aligned(K_kernel, 64);
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
    
    // Clean up kernel memory
    if (K_kernel) free(K_kernel);
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

    // Build the Clifford-kernel fresh every time
    float *K_kernel = NULL;
    int needed_K = K_cols * K_rows;
        if (needed_K > 0) {
            int nb_unused;
            L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K_kernel);
            if (K_kernel) K_kernel = (float *)__builtin_assume_aligned(K_kernel, 64);
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
    
    // Clean up kernel memory
    if (K_kernel) free(K_kernel);
}

void optimized_matrix_free_scalar(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    if (L->sig.dim != 3) {
        // fprintf(stderr, "Error: optimized_matrix_free_scalar only supports 3D Clifford algebra (dim=3, n_blades=8).\n");
        return;
    }

    const int I = 8;
    const int C_in = L->in_channels;
    const int C_out = L->out_channels;

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    for (int b = 0; b < B; ++b) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            
            float y_acc[8];
            const float* bias_ptr = L->bias ? (L->bias + c_out * I) : NULL;
            if (bias_ptr) {
                for (int i = 0; i < I; ++i) y_acc[i] = bias_ptr[i];
            } else {
                for (int i = 0; i < I; ++i) y_acc[i] = 0.0f;
            }

            for (int c_in = 0; c_in < C_in; ++c_in) {
                float w_slice[8];
                for(int i=0; i<I; ++i) {
                    w_slice[i] = L->weight[(i * C_out + c_out) * C_in + c_in];
                }
                const float* x_slice = x + (b * C_in + c_in) * I;

                // Use 2 accumulators per blade to break dependency chains.
                float c0_acc0=0, c0_acc1=0;
                float c1_acc0=0, c1_acc1=0;
                float c2_acc0=0, c2_acc1=0;
                float c3_acc0=0, c3_acc1=0;
                float c4_acc0=0, c4_acc1=0;
                float c5_acc0=0, c5_acc1=0;
                float c6_acc0=0, c6_acc1=0;
                float c7_acc0=0, c7_acc1=0;

                // Term 0
                c0_acc0 = fmaf(w_slice[0], x_slice[0], c0_acc0);
                c1_acc0 = fmaf(w_slice[1], x_slice[0], c1_acc0);
                c2_acc0 = fmaf(w_slice[2], x_slice[0], c2_acc0);
                c3_acc0 = fmaf(w_slice[3], x_slice[0], c3_acc0);
                c4_acc0 = fmaf(w_slice[4], x_slice[0], c4_acc0);
                c5_acc0 = fmaf(w_slice[5], x_slice[0], c5_acc0);
                c6_acc0 = fmaf(w_slice[6], x_slice[0], c6_acc0);
                c7_acc0 = fmaf(w_slice[7], x_slice[0], c7_acc0);

                // Term 1
                c0_acc1 = fmaf(g0 * w_slice[1], x_slice[1], c0_acc1);
                c1_acc1 = fmaf(w_slice[0], x_slice[1], c1_acc1);
                c2_acc1 = fmaf(g0 * w_slice[4], x_slice[1], c2_acc1);
                c3_acc1 = fmaf(g0 * w_slice[5], x_slice[1], c3_acc1);
                c4_acc1 = fmaf(w_slice[2], x_slice[1], c4_acc1);
                c5_acc1 = fmaf(w_slice[3], x_slice[1], c5_acc1);
                c6_acc1 = fmaf(g0 * w_slice[7], x_slice[1], c6_acc1);
                c7_acc1 = fmaf(w_slice[6], x_slice[1], c7_acc1);

                // Term 2
                c0_acc0 = fmaf(g1 * w_slice[2], x_slice[2], c0_acc0);
                c1_acc0 = fmaf(-g1 * w_slice[4], x_slice[2], c1_acc0);
                c2_acc0 = fmaf(w_slice[0], x_slice[2], c2_acc0);
                c3_acc0 = fmaf(g1 * w_slice[6], x_slice[2], c3_acc0);
                c4_acc0 = fmaf(-w_slice[1], x_slice[2], c4_acc0);
                c5_acc0 = fmaf(-g1 * w_slice[7], x_slice[2], c5_acc0);
                c6_acc0 = fmaf(w_slice[3], x_slice[2], c6_acc0);
                c7_acc0 = fmaf(-w_slice[5], x_slice[2], c7_acc0);

                // Term 3
                c0_acc1 = fmaf(g2 * w_slice[3], x_slice[3], c0_acc1);
                c1_acc1 = fmaf(-g2 * w_slice[5], x_slice[3], c1_acc1);
                c2_acc1 = fmaf(-g2 * w_slice[6], x_slice[3], c2_acc1);
                c3_acc1 = fmaf(w_slice[0], x_slice[3], c3_acc1);
                c4_acc1 = fmaf(g2 * w_slice[7], x_slice[3], c4_acc1);
                c5_acc1 = fmaf(-w_slice[1], x_slice[3], c5_acc1);
                c6_acc1 = fmaf(-w_slice[2], x_slice[3], c6_acc1);
                c7_acc1 = fmaf(w_slice[4], x_slice[3], c7_acc1);

                // Term 4
                c0_acc0 = fmaf(-g0 * g1 * w_slice[4], x_slice[4], c0_acc0);
                c1_acc0 = fmaf(g1 * w_slice[2], x_slice[4], c1_acc0);
                c2_acc0 = fmaf(-g0 * w_slice[1], x_slice[4], c2_acc0);
                c3_acc0 = fmaf(-g0 * g1 * w_slice[7], x_slice[4], c3_acc0);
                c4_acc0 = fmaf(w_slice[0], x_slice[4], c4_acc0);
                c5_acc0 = fmaf(g1 * w_slice[6], x_slice[4], c5_acc0);
                c6_acc0 = fmaf(-g0 * w_slice[5], x_slice[4], c6_acc0);
                c7_acc0 = fmaf(w_slice[3], x_slice[4], c7_acc0);

                // Term 5
                c0_acc1 = fmaf(-g0 * g2 * w_slice[5], x_slice[5], c0_acc1);
                c1_acc1 = fmaf(g2 * w_slice[3], x_slice[5], c1_acc1);
                c2_acc1 = fmaf(g0 * g2 * w_slice[7], x_slice[5], c2_acc1);
                c3_acc1 = fmaf(-g0 * w_slice[1], x_slice[5], c3_acc1);
                c4_acc1 = fmaf(-g2 * w_slice[6], x_slice[5], c4_acc1);
                c5_acc1 = fmaf(w_slice[0], x_slice[5], c5_acc1);
                c6_acc1 = fmaf(g0 * w_slice[4], x_slice[5], c6_acc1);
                c7_acc1 = fmaf(-w_slice[2], x_slice[5], c7_acc1);

                // Term 6
                c0_acc0 = fmaf(-g1 * g2 * w_slice[6], x_slice[6], c0_acc0);
                c1_acc0 = fmaf(-g1 * g2 * w_slice[7], x_slice[6], c1_acc0);
                c2_acc0 = fmaf(g2 * w_slice[3], x_slice[6], c2_acc0);
                c3_acc0 = fmaf(-g1 * w_slice[2], x_slice[6], c3_acc0);
                c4_acc0 = fmaf(g2 * w_slice[5], x_slice[6], c4_acc0);
                c5_acc0 = fmaf(-g1 * w_slice[4], x_slice[6], c5_acc0);
                c6_acc0 = fmaf(w_slice[0], x_slice[6], c6_acc0);
                c7_acc0 = fmaf(w_slice[1], x_slice[6], c7_acc0);

                // Term 7
                c0_acc1 = fmaf(-g0 * g1 * g2 * w_slice[7], x_slice[7], c0_acc1);
                c1_acc1 = fmaf(-g1 * g2 * w_slice[6], x_slice[7], c1_acc1);
                c2_acc1 = fmaf(g0 * g2 * w_slice[5], x_slice[7], c2_acc1);
                c3_acc1 = fmaf(-g0 * g1 * w_slice[4], x_slice[7], c3_acc1);
                c4_acc1 = fmaf(g2 * w_slice[3], x_slice[7], c4_acc1);
                c5_acc1 = fmaf(-g1 * w_slice[2], x_slice[7], c5_acc1);
                c6_acc1 = fmaf(g0 * w_slice[1], x_slice[7], c6_acc1);
                c7_acc1 = fmaf(w_slice[0], x_slice[7], c7_acc1);
                
                // Sum contributions and add to the main accumulator
                y_acc[0] += c0_acc0 + c0_acc1;
                y_acc[1] += c1_acc0 + c1_acc1;
                y_acc[2] += c2_acc0 + c2_acc1;
                y_acc[3] += c3_acc0 + c3_acc1;
                y_acc[4] += c4_acc0 + c4_acc1;
                y_acc[5] += c5_acc0 + c5_acc1;
                y_acc[6] += c6_acc0 + c6_acc1;
                y_acc[7] += c7_acc0 + c7_acc1;
            }

            float* out_ptr = out + (b * C_out + c_out) * I;
            for (int i=0; i<I; ++i) {
                out_ptr[i] = y_acc[i];
            }
        }
    }
}

void optimized_matrix_free_avx(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    if (L->sig.dim != 3) {
        // fprintf(stderr, "Error: optimized_matrix_free_avx only supports 3D Clifford algebra (dim=3, n_blades=8).\n");
        return;
    }

    const int I = 8;
    const int C_in = L->in_channels;
    const int C_out = L->out_channels;

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // P_MASKS are constant and can be initialized once.
    static __m256i P_MASKS[8];
    static bool masks_initialized = false;
    if (!masks_initialized) {
        P_MASKS[0] = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); // col 0
        P_MASKS[1] = _mm256_set_epi32(6, 7, 3, 2, 5, 4, 0, 1); // col 1
        P_MASKS[2] = _mm256_set_epi32(5, 3, 7, 1, 6, 0, 4, 2); // col 2
        P_MASKS[3] = _mm256_set_epi32(4, 2, 1, 7, 0, 6, 5, 3); // col 3
        P_MASKS[4] = _mm256_set_epi32(3, 5, 6, 0, 7, 1, 2, 4); // col 4
        P_MASKS[5] = _mm256_set_epi32(2, 4, 0, 6, 1, 7, 3, 5); // col 5
        P_MASKS[6] = _mm256_set_epi32(1, 0, 4, 5, 2, 3, 7, 6); // col 6
        P_MASKS[7] = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7); // col 7
        masks_initialized = true;
    }
    
    // C_VECS depend on g, so they are re-calculated on each call.
    __m256 C_VECS[8];
    C_VECS[0] = _mm256_set1_ps(1.0f);
    C_VECS[1] = _mm256_set_ps(1.f, g0, 1.f, 1.f, g0, g0, 1.f, g0);
    C_VECS[2] = _mm256_set_ps(-1.f, 1.f, -g1, -1.f, g1, 1.f, -g1, g1);
    C_VECS[3] = _mm256_set_ps(1.f, -1.f, -1.f, g2, 1.f, -g2, -g2, g2);
    C_VECS[4] = _mm256_set_ps(1.f, -g0, g1, 1.f, -g0*g1, -g0, g1, -g0*g1);
    C_VECS[5] = _mm256_set_ps(1.f, g0, 1.f, -g2, -g0, g0*g2, g2, -g0*g2);
    C_VECS[6] = _mm256_set_ps(1.f, 1.f, -g1, g2, -g1, g2, -g1*g2, -g1*g2);
    C_VECS[7] = _mm256_set_ps(1.f, g0, -g1, g2, -g0*g1, g0*g2, -g1*g2, -g0*g1*g2);

    for (int b = 0; b < B; ++b) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            
            __m256 y_acc;
            if (L->bias) {
                y_acc = _mm256_loadu_ps(L->bias + c_out * I);
            } else {
                y_acc = _mm256_setzero_ps();
            }

            for (int c_in = 0; c_in < C_in; ++c_in) {
                // Gather the 8 weights for this input/output channel pair.
                // A gather instruction would be ideal, but for 8 floats, this is fine.
                float w_buffer[8];
                for(int i=0; i<I; ++i) {
                    w_buffer[i] = L->weight[(i * C_out + c_out) * C_in + c_in];
                }
                __m256 w_vec = _mm256_loadu_ps(w_buffer);
                
                const float* x_slice = x + (b * C_in + c_in) * I;

                #pragma unroll
                for (int i_in = 0; i_in < 8; ++i_in) {
                    __m256 x_broadcast = _mm256_set1_ps(x_slice[i_in]);
                    __m256 w_shuffled  = _mm256_permutevar8x32_ps(w_vec, P_MASKS[i_in]);
                    __m256 term        = _mm256_mul_ps(C_VECS[i_in], w_shuffled);
                    y_acc              = _mm256_fmadd_ps(term, x_broadcast, y_acc);
                }
            }
            _mm256_storeu_ps(out + (b * C_out + c_out) * I, y_acc);
        }
    }
}

void register_linear_functions() {
    add_linear_function(baseline_linear, "baseline_linear");
    add_linear_function(optimized_linear_basic, "optimized_basic");
    // add_linear_function(optimized_linear_fused_kernel, "optimized_linear_fused_kernel");
    add_linear_function(optimized_linear_1d_kernel, "optimized_linear_1d_kernel");
    add_linear_function(optimized_linear_1d_kernel_reordered_weights, "optimized_linear_1d_kernel_reordered_weights");
    add_linear_function(optimized_linear_1d_kernel_reordered_weights_factored_g_accum, "optimized_linear_1d_kernel_reordered_weights_factored_g_accum");
    add_linear_function(optimized_linear_1d_kernel_reordered_weights_factored_g_accum_fma, "optimized_linear_1d_kernel_reordered_weights_factored_g_accum_fma");
    add_linear_function(optimized_linear_1d_kernel_reordered_source_weights_factored_g_accum_fma, "optimized_linear_1d_kernel_reordered_source_weights_factored_g_accum_fma");
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma, "optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma");
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum, "optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum");
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner, "optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner");
    // add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner_inverse, "optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner_inverse");
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse, "optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse");
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse_unrolled_g, "optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse_unrolled_g");
    // add_linear_function(optimized_linear_simple, "optimized_simple");
    // add_linear_function(optimized_no_kernel_v0, "optimized_no_kernel_v0");
    // add_linear_function(optimized_linear_v2, "optimized_v2_k_accum");
    add_linear_function(optimized_linear_v3, "optimized_v3_fma_k_accum");
    add_linear_function(optimized_linear_v4, "optimized_v4_tiled_fma_k_accum");
    
    add_linear_function(optimized_linear_v5, "optimized_v5_scalar_peak");
    add_linear_function(optimized_linear_v6_avx, "optimized_v6_avx");
    add_linear_function(optimized_linear_v7_avx_k_accum, "optimized_v7_avx_k_accum");
    add_linear_function(optimized_linear_v8_avx_more_unroll, "optimized_v8_avx_more_unroll");
    // add_linear_function(optimized_linear_v9_scalar_no_xf, "optimized_v9_scalar_no_xf");
    // add_linear_function(optimized_linear_v10_avx_no_xf, "optimized_v10_avx_no_xf");
    // add_linear_function(optimized_matrix_free_scalar, "optimized_matrix_free_scalar");
    // add_linear_function(optimized_matrix_free_avx, "optimized_matrix_free_avx");
}
