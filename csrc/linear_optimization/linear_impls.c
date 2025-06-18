#include "linear_bench.h"
#include "../clifford_linear.h"
#include "../flops.h"
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdio.h>
#include <math.h>

// --------------------------------------------------------------------
// Scratch buffer for flattening
// --------------------------------------------------------------------
static float  *scratch_Xf = NULL;
static size_t  scratch_sz = 0;
static inline float* ensure_scratch(size_t needed) {
    if (scratch_sz < needed) {
        free(scratch_Xf);
        scratch_Xf = malloc(needed * sizeof *scratch_Xf);
        scratch_sz = needed;
    }
    return scratch_Xf;
}

// --------------------------------------------------------------------
// 1) baseline_linear
// --------------------------------------------------------------------
void baseline_linear(const CliffordLinear* L, const float* x, int B, float* out) {
    clifford_linear_forward(L, x, B, out);
}

// --------------------------------------------------------------------
// 2) optimized_linear_simple
// --------------------------------------------------------------------
void optimized_linear_simple(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        /* flatten */
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f; if (L->bias) FLOP(1);
                float *Krow = K + (size_t)r * d_flat;

                int d = 0, lim = (int)d_flat - 3;
                for (; d < lim; d += 4) {
                    acc +=
                      Krow[d  ] * Xf[d  ]
                    + Krow[d+1] * Xf[d+1]
                    + Krow[d+2] * Xf[d+2]
                    + Krow[d+3] * Xf[d+3];
                    FLOP(8);
                }
                for (; d < (int)d_flat; ++d) {
                    acc += Krow[d] * Xf[d]; FLOP(2);
                }

                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 3) optimized_linear_v2
// --------------------------------------------------------------------
void optimized_linear_v2(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float a0=0,a1=0,a2=0,a3=0;
                if (L->bias) { a0 = L->bias[r]; FLOP(1); }
                float *Krow = K + (size_t)r * d_flat;

                int d=0, lim = ((int)d_flat/4)*4;
                for (; d < lim; d += 4) {
                    a0 += Krow[d+0]*Xf[d+0];
                    a1 += Krow[d+1]*Xf[d+1];
                    a2 += Krow[d+2]*Xf[d+2];
                    a3 += Krow[d+3]*Xf[d+3];
                    FLOP(8);
                }
                float sum = a0+a1+a2+a3; FLOP(3);
                for (; d < (int)d_flat; ++d) {
                    sum += Krow[d]*Xf[d]; FLOP(2);
                }
                outb[oc * I + i_out] = sum;
            }
        }
    }
}

// --------------------------------------------------------------------
// 4) optimized_linear_v3
// --------------------------------------------------------------------
void optimized_linear_v3(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float a0=0,a1=0,a2=0,a3=0;
                if (L->bias) { a0 = L->bias[r]; FLOP(1); }
                float *Krow = K + (size_t)r * d_flat;

                int d=0, lim = ((int)d_flat/4)*4;
                for (; d < lim; d += 4) {
                    a0 = fmaf(Krow[d+0], Xf[d+0], a0);
                    a1 = fmaf(Krow[d+1], Xf[d+1], a1);
                    a2 = fmaf(Krow[d+2], Xf[d+2], a2);
                    a3 = fmaf(Krow[d+3], Xf[d+3], a3);
                    FLOP(8);
                }
                float sum = a0+a1+a2+a3; FLOP(3);
                for (; d < (int)d_flat; ++d) {
                    sum = fmaf(Krow[d], Xf[d], sum); FLOP(2);
                }
                outb[oc * I + i_out] = sum;
            }
        }
    }
}

// --------------------------------------------------------------------
// 5) optimized_linear_v4
// --------------------------------------------------------------------
void optimized_linear_v4(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f;
                if (L->bias) FLOP(1);

                float *Krow = K + (size_t)r * d_flat;
                const int TILE = 256;
                for (size_t base = 0; base < d_flat; base += TILE) {
                    size_t end = base + TILE;
                    if (end > d_flat) end = d_flat;
                    float a0=0,a1=0,a2=0,a3=0;
                    size_t t = base, lim = base + (((end-base)/4)*4);
                    for (; t < lim; t += 4) {
                        a0 = fmaf(Krow[t+0], Xf[t+0], a0);
                        a1 = fmaf(Krow[t+1], Xf[t+1], a1);
                        a2 = fmaf(Krow[t+2], Xf[t+2], a2);
                        a3 = fmaf(Krow[t+3], Xf[t+3], a3);
                        FLOP(8);
                    }
                    float sum = a0+a1+a2+a3; FLOP(3);
                    for (; t < end; ++t) {
                        sum = fmaf(Krow[t], Xf[t], sum); FLOP(2);
                    }
                    acc += sum; FLOP(1);
                }
                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 6) optimized_linear_v5 (alias of v4 for experimentation)
// --------------------------------------------------------------------
void optimized_linear_v5(
    const CliffordLinear* L,
    const float*          x,
    int                   B,
    float*                out
) {
    optimized_linear_v4(L, x, B, out);
}

// --------------------------------------------------------------------
// 7) optimized_linear_v6_avx
// --------------------------------------------------------------------
static inline float horizontal_sum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s,s);
    s = _mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}
void optimized_linear_v6_avx(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f; if (L->bias) FLOP(1);
                float *Krow = K + (size_t)r * d_flat;

                __m256 vec = _mm256_setzero_ps();
                int d = 0, lim = ((int)d_flat/8)*8;
                for (; d < lim; d += 8) {
                    __m256 kv = _mm256_loadu_ps(Krow + d);
                    __m256 xv = _mm256_loadu_ps(Xf   + d);
                    vec = _mm256_fmadd_ps(kv, xv, vec); FLOP(16);
                }
                float sum = horizontal_sum_m256(vec); FLOP(7);
                acc += sum; FLOP(1);
                for (; d < (int)d_flat; ++d) {
                    acc = fmaf(Krow[d], Xf[d], acc); FLOP(2);
                }
                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 8) optimized_linear_v7_avx_k_accum
// --------------------------------------------------------------------
void optimized_linear_v7_avx_k_accum(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f; if (L->bias) FLOP(1);
                float *Krow = K + (size_t)r * d_flat;

                __m256 va = _mm256_setzero_ps();
                __m256 vb = _mm256_setzero_ps();
                int d=0, lim16=((int)d_flat/16)*16;
                for (; d < lim16; d += 16) {
                    __m256 k0=_mm256_loadu_ps(Krow+d+ 0), x0=_mm256_loadu_ps(Xf+d+ 0);
                    va=_mm256_fmadd_ps(k0,x0,va); FLOP(16);
                    __m256 k1=_mm256_loadu_ps(Krow+d+ 8), x1=_mm256_loadu_ps(Xf+d+ 8);
                    vb=_mm256_fmadd_ps(k1,x1,vb); FLOP(16);
                }
                if (d < ((int)d_flat/8)*8) {
                    __m256 k0=_mm256_loadu_ps(Krow+d), x0=_mm256_loadu_ps(Xf+d);
                    va=_mm256_fmadd_ps(k0,x0,va); FLOP(16);
                    d+=8;
                }
                __m256 vs=_mm256_add_ps(va,vb); FLOP(8);
                float sum = horizontal_sum_m256(vs); FLOP(7);
                acc += sum; FLOP(1);
                for (; d < (int)d_flat; ++d) {
                    acc = fmaf(Krow[d], Xf[d], acc); FLOP(2);
                }
                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 9) optimized_linear_v8_avx_more_unroll
// --------------------------------------------------------------------
void optimized_linear_v8_avx_more_unroll(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f; if (L->bias) FLOP(1);
                float *Krow = K + (size_t)r * d_flat;

                __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
                int d=0, lim32=((int)d_flat/32)*32;
                for (; d < lim32; d += 32) {
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+ 0),
                                      _mm256_loadu_ps(Xf  +d+ 0),a0); FLOP(16);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+ 8),
                                      _mm256_loadu_ps(Xf  +d+ 8),a1); FLOP(16);
                    a2=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+16),
                                      _mm256_loadu_ps(Xf  +d+16),a2); FLOP(16);
                    a3=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+24),
                                      _mm256_loadu_ps(Xf  +d+24),a3); FLOP(16);
                }
                int lim16=((int)d_flat/16)*16;
                if (d < lim16) {
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+ 0),
                                      _mm256_loadu_ps(Xf  +d+ 0),a0); FLOP(16);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+ 8),
                                      _mm256_loadu_ps(Xf  +d+ 8),a1); FLOP(16);
                    d+=16;
                }
                int lim8=((int)d_flat/8)*8;
                if (d < lim8) {
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d),
                                      _mm256_loadu_ps(Xf  +d),a0); FLOP(16);
                    d+=8;
                }
                __m256 vs = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3)); FLOP(24);
                float sum = horizontal_sum_m256(vs); FLOP(7);
                acc += sum; FLOP(1);
                for (; d < (int)d_flat; ++d) {
                    acc = fmaf(Krow[d], Xf[d], acc); FLOP(2);
                }
                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 9) optimized_linear_v9_scalar_no_xf
// --------------------------------------------------------------------
void optimized_linear_v9_scalar_no_xf(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I_in  = L->sig.n_blades;
    const int I_out = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const int K_cols = C_in * I_in;
    const int K_rows = C_out * I_out;

    static float *Kk = NULL;
    static int     last_szK = 0;
    int neededK = K_cols * K_rows;
    if (last_szK != neededK) {
        if (Kk) { free(Kk); Kk = NULL; }
        if (neededK > 0) {
            int nb_unused;
            L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &Kk);
            Kk = (float*)__builtin_assume_aligned(Kk,64);
        }
        last_szK = neededK;
    }

    for (int b = 0; b < B; ++b) {
        const float *xptr = x + (size_t)b * (C_in * I_in);
        float       *optr = out + (size_t)b * (C_out * I_out);

        for (int i_out = 0; i_out < I_out; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int row = i_out * C_out + oc;
                float acc = L->bias ? L->bias[row] : 0.0f; if (L->bias) FLOP(1);
                const float *Krow = Kk + (size_t)row * K_cols;
                for (int ic = 0; ic < C_in; ++ic) {
                    const float *xch = xptr + (size_t)ic * I_in;
                    float c0=0,c1=0,c2=0,c3=0;
                    int lim = (I_in/4)*4, d=0;
                    for (; d < lim; d += 4) {
                        c0 = fmaf(Krow[(d+0)*C_in+ic], xch[d+0], c0);
                        c1 = fmaf(Krow[(d+1)*C_in+ic], xch[d+1], c1);
                        c2 = fmaf(Krow[(d+2)*C_in+ic], xch[d+2], c2);
                        c3 = fmaf(Krow[(d+3)*C_in+ic], xch[d+3], c3);
                        FLOP(8);
                    }
                    float sum = c0+c1+c2+c3; FLOP(3);
                    for (; d < I_in; ++d) {
                        sum = fmaf(Krow[d*C_in+ic], xch[d], sum); FLOP(2);
                    }
                    acc += sum; FLOP(1);
                }
                optr[oc * I_out + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 10) optimized_linear_v10_avx_no_xf
// --------------------------------------------------------------------
void optimized_linear_v10_avx_no_xf(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I_in  = L->sig.n_blades;
    const int I_out = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const int K_cols = C_in * I_in;
    const int K_rows = C_out * I_out;

    static float *Kk = NULL;
    static int     last_szK = 0;
    int neededK = K_cols * K_rows;
    if (last_szK != neededK) {
        if (Kk) { free(Kk); Kk = NULL; }
        if (neededK > 0) {
            int nb_unused;
            L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &Kk);
            Kk = (float*)__builtin_assume_aligned(Kk,64);
        }
        last_szK = neededK;
    }

    if (neededK > 0 && !Kk) {
        for (int i = 0; i < B * C_out * I_out; ++i) out[i] = 0.0f;
        return;
    }

    for (int b = 0; b < B; ++b) {
        const float *xptr = x + (size_t)b * (C_in * I_in);
        float       *optr = out + (size_t)b * (C_out * I_out);

        for (int i_out = 0; i_out < I_out; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int row = i_out * C_out + oc;
                float acc = L->bias ? L->bias[row] : 0.0f; if (L->bias) FLOP(1);
                const float *Krow = Kk + (size_t)row * K_cols;
                for (int ic = 0; ic < C_in; ++ic) {
                    const float *xch = xptr + (size_t)ic * I_in;
                    __m256 vec = _mm256_setzero_ps();
                    int lim = (I_in/8)*8, d = 0;
                    for (; d < lim; d += 8) {
                        __m256 xv = _mm256_loadu_ps(xch + d);
                        __m256i idx = _mm256_setr_epi32(
                            (d+0)*C_in+ic, (d+1)*C_in+ic, (d+2)*C_in+ic, (d+3)*C_in+ic,
                            (d+4)*C_in+ic, (d+5)*C_in+ic, (d+6)*C_in+ic, (d+7)*C_in+ic
                        );
                        __m256 kv = _mm256_i32gather_ps(Krow, idx, 4);
                        vec = _mm256_fmadd_ps(kv, xv, vec); FLOP(16);
                    }
                    float sum = horizontal_sum_m256(vec); FLOP(7);
                    for (; d < I_in; ++d) {
                        sum = fmaf(Krow[d*C_in+ic], xch[d], sum); FLOP(2);
                    }
                    acc += sum; FLOP(1);
                }
                optr[oc * I_out + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// register_linear_functions
// --------------------------------------------------------------------
void register_linear_functions() {
    add_linear_function(baseline_linear,                       "baseline_linear");
    add_linear_function(optimized_linear_simple,             "optimized_simple");
    add_linear_function(optimized_linear_v2,                 "optimized_v2_k_accum");
    add_linear_function(optimized_linear_v3,                 "optimized_v3_fma_k_accum");
    add_linear_function(optimized_linear_v4,                 "optimized_v4_tiled_fma_k_accum");
    add_linear_function(optimized_linear_v5,                 "optimized_v5_dup_v4");
    add_linear_function(optimized_linear_v6_avx,             "optimized_v6_avx");
    add_linear_function(optimized_linear_v7_avx_k_accum,     "optimized_v7_avx_k_accum");
    add_linear_function(optimized_linear_v8_avx_more_unroll, "optimized_v8_avx_more_unroll");
    add_linear_function(optimized_linear_v9_scalar_no_xf,    "optimized_v9_scalar_no_xf");
    add_linear_function(optimized_linear_v10_avx_no_xf,      "optimized_v10_avx_no_xf");
}
