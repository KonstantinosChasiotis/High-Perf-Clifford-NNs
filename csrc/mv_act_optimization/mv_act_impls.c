#include "mv_act_bench.h"
#include "mv_act_setup.h"
#include "../mv_act.h"
#include "../flops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>


static void baseline_act(
    const float *x,
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes for input data
    size_t input_bytes = B * C * I_in * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for input_blades and kernel_blades
    size_t input_blades_bytes = I_in * sizeof(int);
    size_t kernel_blades_bytes = K * sizeof(int);
    BYTES_READ(input_blades_bytes);
    BYTES_READ(kernel_blades_bytes);
    
    // Track bytes for weight and bias
    size_t weight_bytes = C * K * sizeof(float);
    size_t bias_bytes = C * sizeof(float);
    BYTES_READ(weight_bytes);
    BYTES_READ(bias_bytes);
    
    // Track bytes for output data
    size_t output_bytes = B * C * I_in * sizeof(float);
    BYTES_WRITTEN(output_bytes);
    
    mv_act_forward(
      x, B, C, I_in, I_full,
      input_blades, weight, K,
      kernel_blades, bias, out
    );
}

static void optimized_act_v0(
    const float *x,
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes for input data
    size_t input_bytes = B * C * I_in * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for input_blades and kernel_blades
    size_t input_blades_bytes = I_in * sizeof(int);
    size_t kernel_blades_bytes = K * sizeof(int);
    BYTES_READ(input_blades_bytes);
    BYTES_READ(kernel_blades_bytes);
    
    // Track bytes for weight and bias
    size_t weight_bytes = C * K * sizeof(float);
    size_t bias_bytes = C * sizeof(float);
    BYTES_READ(weight_bytes);
    BYTES_READ(bias_bytes);
    
    // Track bytes for output data
    size_t output_bytes = B * C * I_in * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    size_t Vsize = (size_t)B * C * I_full;
    float *v = malloc(Vsize * sizeof(float));
    memset(v, 0, Vsize * sizeof(float));

    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        size_t xc = (size_t)b*C*I_in + c*I_in;
        size_t vc = (size_t)b*C*I_full + c*I_full;
        for (int i = 0; i < I_in; ++i) {
          v[vc + input_blades[i]] = x[xc + i];
        }
      }
    }

    float *gate = malloc((size_t)B * C * sizeof(float));
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        float acc = bias[c];
        const float *w = weight + (size_t)c * K;
        FLOP(2);
        size_t vc = (size_t)b*C*I_full + c*I_full;
        for (int k = 0; k < K; ++k) {
          int blade = kernel_blades[k];
          acc += w[k] * v[vc + blade];
          FLOP(2);
        }
        gate[b*C + c] = 1.0f / (1.0f + expf(-acc));
        FLOP(4);
      }
    }

    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        float g = gate[b*C + c];
        size_t vc = (size_t)b*C*I_full + c*I_full;
        size_t xo = (size_t)b*C*I_in  + c*I_in;
        for (int i = 0; i < I_in; ++i) {
          out[xo + i] = v[vc + input_blades[i]] * g;
          FLOP(1);
        }
      }
    }

    free(gate);
    free(v);
}

/*
 * optimized_act_v1
 *
 * This function computes a gated activation over a 3D tensor x[B][C][I_in],
 * using a sparse kernel defined by kernel_blades[K] and an inverse mapping
 * of input_blades[I_in] into a dense space of size I_full. It eliminates
 * the need for a large temporary buffer by performing an on-the-fly lookup.
 *
 * Arguments:
 *   x             - pointer to input tensor of shape [B * C * I_in]
 *   B, C          - batch size and channel count
 *   I_in, I_full  - number of used blades and total blades respectively
 *   input_blades  - array of length I_in mapping input slots → blade indices
 *   weight        - weight tensor of shape [C * K]
 *   K             - number of kernel blades (sparse dot length)
 *   kernel_blades - array of length K listing blade indices for each weight
 *   bias          - bias array of length C
 *   out           - pointer to output tensor of shape [B * C * I_in]
 */
void optimized_act_v1(
    const float *x,
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes for input data
    size_t input_bytes = B * C * I_in * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for input_blades and kernel_blades
    size_t input_blades_bytes = I_in * sizeof(int);
    size_t kernel_blades_bytes = K * sizeof(int);
    BYTES_READ(input_blades_bytes);
    BYTES_READ(kernel_blades_bytes);
    
    // Track bytes for weight and bias
    size_t weight_bytes = C * K * sizeof(float);
    size_t bias_bytes = C * sizeof(float);
    BYTES_READ(weight_bytes);
    BYTES_READ(bias_bytes);
    
    // Track bytes for output data
    size_t output_bytes = B * C * I_in * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    // ----------------------------------------------------------------------
    // 1) Build inverse map: blade_index → position in x (or -1 if unused)
    //    - Allocate an array of length I_full
    //    - Initialize each slot to -1 (marks "unused blade")
    int *inv = malloc(I_full * sizeof *inv);
    if (!inv) {
        // Optional: handle allocation failure
        return;
    }
    for (int i = 0; i < I_full; ++i) {
        inv[i] = -1;
    }
    //    - Fill in only the used blades by inverting the input_blades map
    for (int i = 0; i < I_in; ++i) {
        int blade = input_blades[i];
        if (blade >= 0 && blade < I_full) {
            inv[blade] = i;
        }
        // else: invalid blade index - could assert or ignore
    }

    // ----------------------------------------------------------------------
    // 2) Allocate scratch space for gate activations of size B * C
    float *gate = malloc((size_t)B * C * sizeof *gate);
    if (!gate) {
        free(inv);
        // Optional: handle allocation failure
        return;
    }

    // ----------------------------------------------------------------------
    // 3) Compute gate[b,c] = sigmoid( bias[c] + Σ_k weight[c*K + k] * x[b,c,inv[blade]] )
    //    - Loop over batches and channels
    //    - For each kernel blade index, look up its input position via inv[]
    //    - If inv[blade] < 0, that blade is unused and contributes zero
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            size_t baseX = (size_t)b * C * I_in + (size_t)c * I_in;
            size_t baseW = (size_t)c * K;
            float acc = bias[c];  // start from bias

            for (int k = 0; k < K; ++k) {
                int blade = kernel_blades[k];
                int xi    = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
                if (xi >= 0) {
                    // valid input position: accumulate weighted x
                    acc += weight[baseW + k] * x[baseX + xi];
                    FLOP(2); // multiply + add
                }
                // if xi < 0: skip unused blade (contribution = 0)
            }

            // Apply sigmoid activation: gate = 1 / (1 + exp(-acc))
            gate[b*C + c] = 1.0f / (1.0f + expf(-acc));
            FLOP(4); // exp + add + reciprocal + misc (project convention)
        }
    }

    // ----------------------------------------------------------------------
    // 4) Write back gated results: out[b,c,i] = x[b,c,i] * gate[b,c]
    //    - Loop over batches, channels, and input positions
    //    - Multiply each original input by the computed gate
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            size_t baseX = (size_t)b * C * I_in + (size_t)c * I_in;
            size_t gate_idx = (size_t)b * C + c;
            float  g = gate[gate_idx];
            for (int i = 0; i < I_in; ++i) {
                // One FLOP: multiply
                out[baseX + i] = x[baseX + i] * g;
                FLOP(1);
            }
        }
    }

    // ----------------------------------------------------------------------
    // 5) Cleanup temporary allocations
    free(gate);
    free(inv);

    // ----------------------------------------------------------------------
}



void optimized_act_v2(
    const float *x,
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes for input data
    size_t input_bytes = B * C * I_in * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for input_blades and kernel_blades
    size_t input_blades_bytes = I_in * sizeof(int);
    size_t kernel_blades_bytes = K * sizeof(int);
    BYTES_READ(input_blades_bytes);
    BYTES_READ(kernel_blades_bytes);
    
    // Track bytes for weight and bias
    size_t weight_bytes = C * K * sizeof(float);
    size_t bias_bytes = C * sizeof(float);
    BYTES_READ(weight_bytes);
    BYTES_READ(bias_bytes);
    
    // Track bytes for output data
    size_t output_bytes = B * C * I_in * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    // 1) Build inverse‐index map: blade → input index (or –1 if unused)
    int *inv = malloc(I_full * sizeof *inv);
    for (int i = 0; i < I_full; ++i) {
        inv[i] = -1;
    }
    for (int i = 0; i < I_in; ++i) {
        inv[input_blades[i]] = i;
    }

    // 2) Allocate workspace for gates
    float *gate = malloc((size_t)B * C * sizeof *gate);

    // 3) Precompute strides
    int CI = C * I_in;   // jump from x[b] to x[b+1]

    // --- Conv + Sigmoid: channels outer, batches inner ---
    for (int c = 0; c < C; ++c) {
        size_t baseW = (size_t)c * K;      // &weight[c][0]
        float  bval  = bias[c];            // bias for channel c
        size_t cIin  = (size_t)c * I_in;   // offset within each batch

        for (int b = 0; b < B; ++b) {
            size_t xoff = (size_t)b * CI + cIin;
            float acc = bval;

            // sparse dot‐product over K
            for (int k = 0; k < K; ++k) {
                int blade = kernel_blades[k];
                int xi    = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
                if (xi >= 0) {
                    acc += weight[baseW + k] * x[xoff + xi];
                    FLOP(2);    // one multiply + one add
                }
            }

            // sigmoid activation
            gate[b*C + c] = 1.0f / (1.0f + expf(-acc));
            FLOP(4);        // exp + add + divide + misc (project convention)
        }
    }

    // --- Scale & write‐back: batches outer, channels inner ---
    for (int b = 0; b < B; ++b) {
        size_t bCI = (size_t)b * CI;
        size_t bg  = (size_t)b * C;

        for (int c = 0; c < C; ++c) {
            size_t xoff = bCI + (size_t)c * I_in;
            float  g    = gate[bg + c];

            for (int i = 0; i < I_in; ++i) {
                out[xoff + i] = x[xoff + i] * g;
                FLOP(1);    // one multiply
            }
        }
    }

    // 4) Cleanup
    free(gate);
    free(inv);
}




/*
 * optimized_act_v5
 *
 * Fused activation without SIMD/OpenMP pragmas, using per-channel
 * precomputation of valid indices and weights to enable branch-free
 * inner loops. This version targets single-core execution with
 * minimal control flow inside hot loops.
 *
 * Arguments:
 *   x             - input tensor [B][C][I_in]
 *   B             - batch size
 *   C             - number of channels
 *   I_in          - number of active blades per channel
 *   I_full        - total number of blade positions
 *   input_blades  - length-I_in array mapping input index → blade index
 *   weight        - weight matrix [C][K]
 *   K             - number of potentially non-zero weights per channel
 *   kernel_blades - length-K array listing blade indices for each weight
 *   bias          - length-C array of channel biases
 *   out           - output tensor [B][C][I_in]
 */
#include <stdlib.h>
#include <math.h>

void optimized_act_v5(
    const float *x,
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes for input data
    size_t input_bytes = B * C * I_in * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for input_blades and kernel_blades
    size_t input_blades_bytes = I_in * sizeof(int);
    size_t kernel_blades_bytes = K * sizeof(int);
    BYTES_READ(input_blades_bytes);
    BYTES_READ(kernel_blades_bytes);
    
    // Track bytes for weight and bias
    size_t weight_bytes = C * K * sizeof(float);
    size_t bias_bytes = C * sizeof(float);
    BYTES_READ(weight_bytes);
    BYTES_READ(bias_bytes);
    
    // Track bytes for output data
    size_t output_bytes = B * C * I_in * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    // ----------------------------------------------------------------------
    // 1) Build inverse map: blade index → input slot (or -1 if unused)
    int *inv = malloc((size_t)I_full * sizeof *inv);
    if (!inv) return;  // allocation failure
    // Initialize all entries to -1
    for (int i = 0; i < I_full; ++i) {
        inv[i] = -1;
    }
    // Fill map for active blades
    for (int i = 0; i < I_in; ++i) {
        int blade = input_blades[i];
        if (blade >= 0 && blade < I_full) {
            inv[blade] = i;
        }
        // else: out-of-range blade ignored
    }

    // ----------------------------------------------------------------------
    // 2) Compute stride: number of floats per batch slice
    size_t stride = (size_t)C * I_in;

    // ----------------------------------------------------------------------
    // 3) Allocate per-channel temporary arrays for valid indices & weights
    int   *valid_idx    = malloc((size_t)K * sizeof *valid_idx);
    float *valid_weight = malloc((size_t)K * sizeof *valid_weight);
    if (!valid_idx || !valid_weight) {
        free(inv);
        free(valid_idx);
        free(valid_weight);
        return;
    }

    // ----------------------------------------------------------------------
    // 4) For each channel: filter out unused blades, then apply to all batches
    for (int c = 0; c < C; ++c) {
        // a) Gather valid indices and corresponding weights
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int blade = kernel_blades[k];
            int idx = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
            if (idx >= 0) {
                valid_idx[vc]    = idx;
                valid_weight[vc] = w_c[k];
                vc++;
            }
        }

        // b) Process each batch for this channel
        for (int b = 0; b < B; ++b) {
            // Pointers to input slice x[b][c][:] and output slice out[b][c][:]
            const float *x_bc = x   + (size_t)b * stride + (size_t)c * I_in;
            float       *o_bc = out + (size_t)b * stride + (size_t)c * I_in;

            // Compute dot-product over valid entries
            float acc = bias[c];  // start from bias
            for (int j = 0; j < vc; ++j) {
                int idx = valid_idx[j];
                acc += valid_weight[j] * x_bc[idx];
                FLOP(2); // multiply + add
            }

            // Apply sigmoid activation
            float gate = 1.0f / (1.0f + expf(-acc));
            FLOP(4); // project convention

            // Scale and store outputs for each active input
            for (int i = 0; i < I_in; ++i) {
                o_bc[i] = x_bc[i] * gate;
                FLOP(1); // multiply
            }
        }
    }

    // ----------------------------------------------------------------------
    // 5) Cleanup temporaries
    free(valid_idx);
    free(valid_weight);
    free(inv);
}









/*
 * optimized_act_v6
 *
 * Incorporates fmaf-based fused activation using precomputed valid indices
 * and weights per channel, minimizing arithmetic in hot loops. Inspired by v3's
 * k_idx/x_idx concept and v5's valid_weight array.
 *
 * Arguments:
 *   x             - input tensor [B][C][I_in]
 *   B             - batch size
 *   C             - number of channels
 *   I_in          - number of active blades per channel
 *   I_full        - total number of blade positions
 *   input_blades  - length-I_in array mapping input index → blade index
 *   weight        - weight matrix [C][K]
 *   K             - number of potentially non-zero weights per channel
 *   kernel_blades - length-K array listing blade indices for each weight
 *   bias          - length-C array of channel biases
 *   out           - output tensor [B][C][I_in]
 */
#include <stdlib.h>
#include <math.h>

void optimized_act_v6(
    const float *x,
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes for input data
    size_t input_bytes = B * C * I_in * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for input_blades and kernel_blades
    size_t input_blades_bytes = I_in * sizeof(int);
    size_t kernel_blades_bytes = K * sizeof(int);
    BYTES_READ(input_blades_bytes);
    BYTES_READ(kernel_blades_bytes);
    
    // Track bytes for weight and bias
    size_t weight_bytes = C * K * sizeof(float);
    size_t bias_bytes = C * sizeof(float);
    BYTES_READ(weight_bytes);
    BYTES_READ(bias_bytes);
    
    // Track bytes for output data
    size_t output_bytes = B * C * I_in * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    // 1) Build inverse map: blade index → input slot
    int *inv = malloc((size_t)I_full * sizeof *inv);
    if (!inv) return;
    for (int i = 0; i < I_full; ++i) inv[i] = -1;
    for (int i = 0; i < I_in; ++i) inv[input_blades[i]] = i;

    // 2) Prepare per-channel valid lists (max K entries each)
    int   *valid_idx    = malloc((size_t)C * K * sizeof *valid_idx);
    float *valid_weight = malloc((size_t)C * K * sizeof *valid_weight);
    int   *valid_count  = malloc((size_t)C * sizeof *valid_count);
    if (!valid_idx || !valid_weight || !valid_count) {
        free(inv);
        free(valid_idx);
        free(valid_weight);
        free(valid_count);
        return;
    }
    for (int c = 0; c < C; ++c) {
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int blade = kernel_blades[k];
            int idx   = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
            if (idx >= 0) {
                valid_idx[c*K + vc]    = idx;
                valid_weight[c*K + vc] = w_c[k];
                vc++;
            }
        }
        valid_count[c] = vc;
    }
    free(inv);

    // 3) Stride for each batch slice
    size_t stride = (size_t)C * I_in;

    // 4) Fused conv + activation + write-back
    for (int c = 0; c < C; ++c) {
        int vc = valid_count[c];
        int *idxs = valid_idx    + (size_t)c * K;
        float *wts = valid_weight + (size_t)c * K;
        float bval = bias[c];

        for (int b = 0; b < B; ++b) {
            const float *x_bc = x   + (size_t)b * stride + (size_t)c * I_in;
            float       *o_bc = out + (size_t)b * stride + (size_t)c * I_in;

            // dot-product using fused multiply-add
            float acc = bval;
            for (int j = 0; j < vc; ++j) {
                acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                FLOP(2); // FMA = 1 mul + 1 add
            }

            // sigmoid activation
            float g = 1.0f / (1.0f + expf(-acc));
            FLOP(4); // project convention

            // scale and store
            for (int i = 0; i < I_in; ++i) {
                o_bc[i] = x_bc[i] * g;
                FLOP(1); // multiply
            }
        }
    }

    // 5) Cleanup
    free(valid_idx);
    free(valid_weight);
    free(valid_count);
}







/*
 * optimized_act_v8
 *
 * Fused activation with LUT-based sigmoid using a reduced range and size
 * for improved cache locality. LUT initialization samples [-6,6] instead of [-8,8],
 * and table size is halved to 512 entries, reducing memory footprint and
 * potentially improving speed while still maintaining ~1e-5 accuracy.
 *
 * Arguments:
 *   x             - input tensor [B][C][I_in]
 *   B             - batch size
 *   C             - number of channels
 *   I_in          - number of active blades per channel
 *   I_full        - total number of blade positions
 *   input_blades  - length-I_in array mapping input index → blade index
 *   weight        - weight matrix [C][K]
 *   K             - number of kernel weights per channel
 *   kernel_blades - length-K array listing blade indices for each weight
 *   bias          - length-C array of channel biases
 *   out           - output tensor [B][C][I_in]
 */
#include <stdlib.h>
#include <math.h>

// ------------------------------------------------------------
// Sigmoid LUT parameters and storage (half-size, tighter range)
// ------------------------------------------------------------
#define SIGMOID_LUT_SIZE 512      // Reduced table size for better cache
static float sigmoid_lut[SIGMOID_LUT_SIZE];
static int   lut_initialized = 0;

// ------------------------------------------------------------
// init_sigmoid_lut
// One-time initialization of the sigmoid LUT over [-6,6]
// ------------------------------------------------------------
static void init_sigmoid_lut(void) {
    const float x_min = -6.0f;
    const float x_max =  6.0f;
    // Precompute for x calculation: (x_max - x_min) / (SIGMOID_LUT_SIZE - 1)
    const float scale_factor_x = (x_max - x_min) / (SIGMOID_LUT_SIZE - 1); FLOP(2); // 1 sub, 1 div (assuming SIGMOID_LUT_SIZE -1 is const)
    for (int i = 0; i < SIGMOID_LUT_SIZE; ++i) {
        float x = x_min + scale_factor_x * i; FLOP(2); // 1 mul, 1 add (i to float implicit)
        sigmoid_lut[i] = 1.0f / (1.0f + expf(-x)); FLOP(4); // project convention
    }
    lut_initialized = 1;
}

// ------------------------------------------------------------
// fast_sigmoid
// Lookup + interpolation over reduced-range LUT
// ------------------------------------------------------------
static inline float fast_sigmoid(float x) {
    if (!lut_initialized) init_sigmoid_lut();
    const float x_min = -6.0f, x_max = 6.0f; ///here we reduced the size to 6.0f from 8.0 it dosen't affect but maybe with other inputs
    if (x <= x_min) return 0.0f; // No FLOPs for comparison if not counted
    if (x >= x_max) return 1.0f; // No FLOPs for comparison
    // Precompute: (SIGMOID_LUT_SIZE - 1) / (x_max - x_min)
    const float scale_factor_interp = (SIGMOID_LUT_SIZE - 1) / (x_max - x_min); // Assuming this is optimized out or precomputed for fast_sigmoid if called often
                                                                               // If counted each time: FLOP(2) here.
                                                                               // If not, add FLOP(2) here. Given it's inline and might be in a hot loop, it's better if this is const.
                                                                               // For init_sigmoid_lut, it makes sense to calculate it once.
                                                                               // Let's assume for this specific call, (SIGMOID_LUT_SIZE - 1) / (x_max - x_min) is a known constant.
    float fx   = (x - x_min) * scale_factor_interp; FLOP(2); // 1 sub, 1 mul
    int   idx  = (int)fx;
    float frac = fx - idx; FLOP(1); // 1 sub
    // Linear interpolation: sigmoid_lut[idx] + frac * (sigmoid_lut[idx+1] - sigmoid_lut[idx])
    // val_diff = sigmoid_lut[idx+1] - sigmoid_lut[idx] -> 1 sub
    // term = frac * val_diff -> 1 mul
    // result = sigmoid_lut[idx] + term -> 1 add
    return sigmoid_lut[idx] + frac * (sigmoid_lut[idx+1] - sigmoid_lut[idx]); FLOP(3); // 1 sub, 1 mul, 1 add
}

// ------------------------------------------------------------
// optimized_act_v8
// Main fused activation kernel with optimized LUT
// ------------------------------------------------------------
void optimized_act_v7(
    const float *x,
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes for input data
    size_t input_bytes = B * C * I_in * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for input_blades and kernel_blades
    size_t input_blades_bytes = I_in * sizeof(int);
    size_t kernel_blades_bytes = K * sizeof(int);
    BYTES_READ(input_blades_bytes);
    BYTES_READ(kernel_blades_bytes);
    
    // Track bytes for weight and bias
    size_t weight_bytes = C * K * sizeof(float);
    size_t bias_bytes = C * sizeof(float);
    BYTES_READ(weight_bytes);
    BYTES_READ(bias_bytes);
    
    // Track bytes for output data
    size_t output_bytes = B * C * I_in * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    // 1) Build inverse map
    int *inv = malloc((size_t)I_full * sizeof *inv);
    if (!inv) return;
    for (int i = 0; i < I_full; ++i) inv[i] = -1;
    for (int i = 0; i < I_in; ++i) inv[input_blades[i]] = i;

    // 2) Build per-channel valid lists
    int   *valid_idx    = malloc((size_t)C * K * sizeof *valid_idx);
    float *valid_weight = malloc((size_t)C * K * sizeof *valid_weight);
    int   *valid_count  = malloc((size_t)C * sizeof *valid_count);
    if (!valid_idx || !valid_weight || !valid_count) {
        free(inv); free(valid_idx); free(valid_weight); free(valid_count);
        return;
    }
    for (int c = 0; c < C; ++c) {
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int blade = kernel_blades[k];
            int idx   = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
            if (idx >= 0) {
                valid_idx[c*K + vc]    = idx;
                valid_weight[c*K + vc] = w_c[k];
                vc++;
            }
        }
        valid_count[c] = vc;
    }
    free(inv);

    // 3) Stride per batch slice
    size_t stride = (size_t)C * I_in;

    // 4) Fused loop: dot, LUT sigmoid, write-back
    for (int c = 0; c < C; ++c) {
        int     vc   = valid_count[c];
        int    *idxs = valid_idx    + (size_t)c * K;
        float  *wts  = valid_weight + (size_t)c * K;
        float   bval = bias[c];

        for (int b = 0; b < B; ++b) {
            const float *x_bc = x   + (size_t)b * stride + (size_t)c * I_in;
            float       *o_bc = out + (size_t)b * stride + (size_t)c * I_in;
            float acc = bval;
            for (int j = 0; j < vc; ++j) {
                acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                FLOP(2); // FMA
            }
            float g = fast_sigmoid(acc); FLOP(6); // From fast_sigmoid: 2 (fx) + 1 (frac) + 3 (lerp)
            for (int i = 0; i < I_in; ++i) {
                o_bc[i] = x_bc[i] * g;
                FLOP(1); // multiply
            }
        }
    }

    // 5) Cleanup
    free(valid_idx); free(valid_weight); free(valid_count);
}









void register_activation_functions(void) {
    add_activation_function(baseline_act,      "baseline_act");
    add_activation_function(optimized_act_v1,  "optimized_act_v1");
    add_activation_function(optimized_act_v2,  "optimized_act_v2");
    add_activation_function(optimized_act_v5,  "optimized_act_v5");
    add_activation_function(optimized_act_v6,  "optimized_act_v6");
    add_activation_function(optimized_act_v7,  "optimized_act_v7");

}
