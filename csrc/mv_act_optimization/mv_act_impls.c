#include "mv_act_bench.h"
#include "mv_act_setup.h"
#include "../mv_act.h"
#include "../flops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

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
    mv_act_forward(
      x, B, C, I_in, I_full,
      input_blades, weight, K,
      kernel_blades, bias, out
    );
}

static void optimized_act_v1(
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

    mv_act_forward(
      x, B, C, I_in, I_full,
      input_blades, weight, K,
      kernel_blades, bias, out
    );
}

static void optimized_act_v2(
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

void register_activation_functions(void) {
    add_activation_function(baseline_act,      "baseline_act");
    add_activation_function(optimized_act_v1,  "optimized_act_v1");
    add_activation_function(optimized_act_v2,  "optimized_act_v2");
}
