#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include "mv_act.h"

/**
 * Multivector activation (linear) across blade dimension:
 * 
 * SO WHAT WE ARE ESSENTIALLY DOING : 
 *   out[b,c,p,i] = x[b,c,p,i] * sigmoid(
 *       sum_{k=0..K-1} weight[c*K + k] * x[b,c,p, kernel_blades[k]]
 *   )
 * 
 * 
 *
 * Todo : need to update that we also need the input_blades and bias
 * Parameters:
 *   x               - input data [B][C][L][I]
 *   B, C, L, I      - batch size, channels,length,num blades
 *   weight          - conv weight [C][K]
 *   K               - number of kernel blades
 *   kernel_blades   - blade indices [K]
 *   out             - output buffer [B][C][L][I]
 */

void mv_act_forward_baseline(
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
    // 1) Allocate + zero the embedded multivector buffer v[B*C*I_full]
    size_t Vsize = (size_t)B * C * I_full;
    float *v = (float*)malloc(Vsize * sizeof(float));
    memset(v, 0, Vsize * sizeof(float));

    // 2) EMBED: scatter each x[b,c,i] → v[b,c, input_blades[i]]
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        size_t xc = (size_t)b*C*I_in + c*I_in;
        size_t vc = (size_t)b*C*I_full + c*I_full;
        for (int i = 0; i < I_in; ++i) {
          int blade = input_blades[i];
          v[vc + blade] = x[xc + i];
        }
      }
    }

    // 3) CONV+SIGMOID: one gate per (b,c)
    float *gate = (float*)malloc((size_t)B * C * sizeof(float));
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        float acc = bias[c];                   // start with bias
        const float *w = weight + (size_t)c * K;
        size_t vc = (size_t)b*C*I_full + c*I_full;
        for (int k = 0; k < K; ++k) {
          int blade = kernel_blades[k];
          if (blade >= 0 && blade < I_full) {
            acc += w[k] * v[vc + blade];
          }
        }
        gate[b*C + c] = 1.0f / (1.0f + expf(-acc));
      }
    }

    // 4) GET: for each (b,c), multiply embedded slots by gate and gather back
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        float g = gate[b*C + c];
        size_t vc = (size_t)b*C*I_full + c*I_full;
        size_t xo = (size_t)b*C*I_in  + c*I_in;
        for (int i = 0; i < I_in; ++i) {
          int blade = input_blades[i];
          out[xo + i] = v[vc + blade] * g;
        }
      }
    }
    free(gate);
    free(v);
}


void optimization1(
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
    // Build inverse map: blade_index → position in x (or –1 if unused)
    int *inv = (int*)malloc(I_full * sizeof(int));
    for (int i = 0; i < I_full; ++i) {
        inv[i] = -1;
    }
    for (int i = 0; i < I_in; ++i) {
        inv[input_blades[i]] = i;
    }

    // Allocate gate[B*C]
    float *gate = (float*)malloc((size_t)B * C * sizeof(float));

    // 1) Compute each gate[b,c] = sigmoid( bias[c] + Σ_k w[c*K + k] * x[b*C*I_in + c*I_in + inv[kernel_blades[k]]] )
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            size_t baseX = (size_t)b * C * I_in + (size_t)c * I_in;
            size_t baseW = (size_t)c * K;
            float acc = bias[c];
            for (int k = 0; k < K; ++k) {
                int blade = kernel_blades[k];
                int xi    = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
                if (xi >= 0) {
                    acc += weight[baseW + k] * x[baseX + xi];
                }
            }
            gate[b*C + c] = 1.0f / (1.0f + expf(-acc));
        }
    }

    // 2) Write back: out[b,c,i] = x[b,c,i] * gate[b,c]
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            size_t baseX = (size_t)b * C * I_in + (size_t)c * I_in;
            size_t baseO = (size_t)b * C + c;
            float g = gate[baseO];
            for (int i = 0; i < I_in; ++i) {
                out[baseX + i] = x[baseX + i] * g;
            }
        }
    }

    free(gate);
    free(inv);
}

void optimization2(
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
    // Build inverse map once
    int *inv = malloc(I_full * sizeof(int));
    for (int i = 0; i < I_full; ++i) inv[i] = -1;
    for (int i = 0; i < I_in;  ++i) inv[input_blades[i]] = i;

    // Allocate gate[B*C]
    float *gate = malloc((size_t)B * C * sizeof(float));

    // Precompute strides
    int CI = C * I_in;   // stride in x/out
    int CG = C;          // stride in gate

    // --- 1) Conv + Sigmoid ---
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            size_t baseX = (size_t)b * CI + (size_t)c * I_in;
            size_t baseW = (size_t)c * K;
            size_t gidx  = b * CG + c;

            float acc = bias[c];
            for (int k = 0; k < K; ++k) {
                int blade = kernel_blades[k];
                int xi    = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
                if (xi >= 0) {
                    acc += weight[baseW + k] * x[baseX + xi];
                }
            }
            gate[gidx] = 1.0f / (1.0f + expf(-acc));
        }
    }

    // --- 2) Scale & write-out ---
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            size_t baseX = (size_t)b * CI + (size_t)c * I_in;
            size_t gidx  = b * CG + c;
            float  g     = gate[gidx];

            for (int i = 0; i < I_in; ++i) {
                out[baseX + i] = x[baseX + i] * g;
            }
        }
    }

    free(gate);
    free(inv);
}


void optimization3(
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
    // 1) Build inverse‐index map: blade → input index (or –1 if unused)
    int *inv = malloc(I_full * sizeof *inv);
    for (int i = 0; i < I_full; ++i)    inv[i] = -1;
    for (int i = 0; i < I_in;  ++i)    inv[input_blades[i]] = i;

    // 2) Allocate workspace for gates
    float *gate = malloc((size_t)B * C * sizeof *gate);

    // 3) Precompute strides
    int CI = C * I_in;   // jump from x[b] to x[b+1]
    int CG = C;          // jump from gate[b] to gate[b+1]

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
                }
            }

            // sigmoid activation
            gate[b*CG + c] = 1.0f / (1.0f + expf(-acc));
        }
    }

    // --- Scale & write‐back: batches outer, channels inner ---
    for (int b = 0; b < B; ++b) {
        size_t bCI = (size_t)b * CI;
        size_t bg  = (size_t)b * CG;

        for (int c = 0; c < C; ++c) {
            size_t xoff = bCI + (size_t)c * I_in;
            float  g    = gate[bg + c];

            for (int i = 0; i < I_in; ++i) {
                out[xoff + i] = x[xoff + i] * g;
            }
        }
    }

    // 4) Cleanup
    free(gate);
    free(inv);
}



void optimization4(
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
    // 1) Build inverse map: blade -> x-index
    int *inv = malloc(I_full * sizeof *inv);
    for (int i = 0; i < I_full; ++i) inv[i] = -1;
    for (int i = 0; i < I_in;  ++i) inv[input_blades[i]] = i;

    // 2) Filter kernel indices: keep only those with inv[...] >= 0
    int *k_idx = malloc(K * sizeof *k_idx);
    int *x_idx = malloc(K * sizeof *x_idx);
    int Kp = 0;
    for (int k = 0; k < K; ++k) {
        int blade = kernel_blades[k];
        if (blade >= 0 && blade < I_full) {
            int xi = inv[blade];
            if (xi >= 0) {
                k_idx[Kp] = k;
                x_idx[Kp] = xi;
                ++Kp;
            }
        }
    }

    // 3) Precompute stride for x/out
    int CI = C * I_in;

    // 4) Fused conv+sigmoid + scale  
    //    Loop order: c outer (reuses weight/bias), b inner (unit-stride on out)
    for (int c = 0; c < C; ++c) {
        size_t baseW = (size_t)c * K;
        float  bval  = bias[c];
        size_t cIin  = (size_t)c * I_in;

        for (int b = 0; b < B; ++b) {
            size_t xoff = (size_t)b * CI + cIin;

            // dot‐product without branch
            float acc = bval;
            for (int j = 0; j < Kp; ++j) {
                float w = weight[baseW + k_idx[j]];
                float xv = x[xoff + x_idx[j]];
                acc = fmaf(w, xv, acc);          // fused multiply–add
            }

            // sigmoid
            float g = 1.0f / (1.0f + expf(-acc));

            // scale & write out
            for (int i = 0; i < I_in; ++i) {
                out[xoff + i] = x[xoff + i] * g;
            }
        }
    }

    free(inv);
    free(k_idx);
    free(x_idx);
}



void mv_act_forward(
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
    // 1) inverse map
    int *inv = malloc(I_full * sizeof *inv);
    for (int i = 0; i < I_full; ++i) inv[i] = -1;
    for (int i = 0; i < I_in;  ++i) inv[input_blades[i]] = i;

    // 2) filter valid kernels
    int *k_idx = malloc(K * sizeof *k_idx);
    int *x_idx = malloc(K * sizeof *x_idx);
    int Kp = 0;
    for (int k = 0; k < K; ++k) {
        int b = kernel_blades[k];
        if (0 <= b && b < I_full) {
            int xi = inv[b];
            if (xi >= 0) {
                k_idx[Kp] = k;
                x_idx[Kp] = xi;
                ++Kp;
            }
        }
    }

    // 3) stride for x/out
    int CI = C * I_in;

    const __m256 one_v = _mm256_set1_ps(1.0f);
    const __m256 two_v = _mm256_set1_ps(2.0f);

    // 4) main loops: channel outer, batch inner
    for (int c = 0; c < C; ++c) {
        size_t woff = (size_t)c * K;
        float  bval = bias[c];
        size_t cI = (size_t)c * I_in;

        for (int b = 0; b < B; ++b) {
            size_t xoff = (size_t)b * CI + cI;

            // vectorized dot-product
            __m256 accv = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= Kp; j += 8) {
                __m256i iw = _mm256_loadu_si256((__m256i*)(k_idx + j));
                __m256i ix = _mm256_loadu_si256((__m256i*)(x_idx + j));
                __m256 wv = _mm256_i32gather_ps(weight + woff, iw, 4);
                __m256 xv = _mm256_i32gather_ps(x      + xoff, ix, 4);
                accv = _mm256_fmadd_ps(wv, xv, accv);
            }
            float buf[8];
            _mm256_storeu_ps(buf, accv);
            float acc = bval
                      + buf[0] + buf[1] + buf[2] + buf[3]
                      + buf[4] + buf[5] + buf[6] + buf[7];
            for (; j < Kp; ++j) {
                acc = fmaf(weight[woff + k_idx[j]],
                           x[xoff     + x_idx[j]],
                           acc);
            }

            // scalar exp + broadcast
            float e = expf(-acc);
            __m256 expv = _mm256_set1_ps(e);

            // denom = 1 + e, then reciprocal via rcp+Newton step
            __m256 denom = _mm256_add_ps(one_v, expv);
            __m256 rcp0 = _mm256_rcp_ps(denom);
            __m256 r = _mm256_mul_ps(rcp0,
                              _mm256_sub_ps(two_v, _mm256_mul_ps(denom, rcp0)));
            __m256 gv = r;

            // scale & store
            int i = 0;
            for (; i + 8 <= I_in; i += 8) {
                __m256 xv2 = _mm256_loadu_ps(x + xoff + i);
                __m256 yv2 = _mm256_fmadd_ps(xv2, gv, _mm256_setzero_ps());
                _mm256_storeu_ps(out + xoff + i, yv2);
            }
            float g_scalar = _mm256_cvtss_f32(r);
            for (; i < I_in; ++i) {
                out[xoff + i] = x[xoff + i] * g_scalar;
            }
        }
    }

    free(inv);
    free(k_idx);
    free(x_idx);
}
