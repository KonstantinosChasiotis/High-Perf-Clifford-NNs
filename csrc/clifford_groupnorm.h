#ifndef CLIFFORD_GROUPNORM_H
#define CLIFFORD_GROUPNORM_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>

void clifford_groupnorm_forward(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_cov_orig,
    bool scaling,
    const float *weight_orig,
    const float *bias_orig,
    bool training,
    float momentum,
    float eps,
    float *x_norm);

#endif // CLIFFORD_GROUPNORM_H
