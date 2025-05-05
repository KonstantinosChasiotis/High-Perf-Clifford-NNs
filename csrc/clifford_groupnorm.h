#ifndef CLIFFORD_GROUPNORM_H
#define CLIFFORD_GROUPNORM_H

/*
TODO:
      - change the benachmarks script to reflect the new implementaiton
      - attempt a total timing data automation
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include "flops.h"

void clifford_groupnorm( //wrapper around with memory allocation
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

void clifford_groupnorm_alloctemp(
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float **running_mean_temp,
    float **running_cov_temp,
    bool scaling,
    float **weight_temp,
    float **weight_perm_temp,
    float **bias_temp,
    float **mean_temp,
    float **X_temp,
    float **cov_temp,
    float **max_temp);

void clifford_groupnorm_freetemp(
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float **running_mean_temp,
    float **running_cov_temp,
    bool scaling,
    float **weight_temp,
    float **weight_perm_temp,
    float **bias_temp,
    float **mean_temp,
    float **X_temp,
    float **cov_temp,
    float **max_temp);

void clifford_groupnorm_part0(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_mean_temp,
    float *running_cov_orig,
    float *running_cov_temp,
    float *mean_temp,
    float *X_temp,
    float *cov_temp,
    float *max_temp,
    bool training,
    float momentum,
    float eps,
    float *x_norm);

void clifford_groupnorm_lapacke(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    float *cov_temp);

void clifford_groupnorm_part1(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool scaling,
    const float *weight_orig,
    float *weight_temp,
    float *weight_perm_temp,
    const float *bias_orig,
    float *bias_temp,
    float *cov_temp,
    bool training,
    float momentum,
    float eps,
    float *x_norm);

#endif // CLIFFORD_GROUPNORM_H
